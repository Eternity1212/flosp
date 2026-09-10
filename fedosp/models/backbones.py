"""骨干适配层：把 ViT / SwinV2 / ResNet / DINOv2 收敛成同一个接口。

**为什么需要这一层。** 消融 A12 要换骨干证明"方法与骨干解耦"，但三类骨干的结构差异很大：

===========  ==========================  ==============================
骨干          层的组织方式                 浅层特征长什么样
===========  ==========================  ==============================
ViT / DINOv2  ``blocks`` 是平的 24 层      ``(B, 1+N, D)`` token 序列
SwinV2        ``layers[i].blocks[j]``     ``(B, H, W, C)``，且 C 逐 stage 翻倍
ResNet        ``layer1..layer4``          ``(B, C, H, W)`` 卷积特征图
===========  ==========================  ==============================

直接写 ``self.backbone.blocks[:6]`` 只对 ViT 成立，SwinV2 会报
``'SwinTransformerV2' object has no attribute 'blocks'``，ResNet 连 ``img_size`` 参数都不接受。

**这一层做的事**：把每个骨干拆成 ``stages``（一个可迭代的模块列表）+ 一个
``split_at`` 切点，让 :class:`FedOSPNet` 只需要写

    tokens = adapter.embed(x)
    tokens = adapter.run(tokens, 0, split)      # 浅层
    tokens = fsr(tokens)                        # FSR
    tokens = adapter.run(tokens, split, None)   # 深层
    feat   = adapter.pool(tokens)

FSR 那一侧则通过 ``token_layout`` 知道自己拿到的是 token 序列还是特征图，
分别走 ViT 版和 CNN 版（CNN 版其实更接近 FedBCS 原文设定）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger("backbones")

#: 特征在 FSR 眼里的三种排布
LAYOUT_TOKENS = "tokens"      # (B, prefix+N, D)，ViT / DINOv2
LAYOUT_NHWC = "nhwc"          # (B, H, W, C)，SwinV2
LAYOUT_NCHW = "nchw"          # (B, C, H, W)，ResNet


@dataclass
class BackboneSpec:
    """一个骨干的元信息，FedOSPNet 只依赖这里的字段。"""

    name: str
    embed_dim: int            # 深层特征维度（进分类头的那个）
    depth: int                # 可切分的 stage 数
    num_prefix_tokens: int    # ViT 的 CLS/dist token 数；CNN 与 Swin 为 0
    layout: str               # 浅层特征排布，见上面三个常量
    shallow_dim: int          # 浅层特征维度（可能 != embed_dim，Swin 就不等）
    lora_block_regex: str     # LoRA 注入时从模块名抠 stage index 的正则
    default_split: int        # FSR 默认插入位置
    #: ``lora_block_regex`` 能抠出的全部 index，**有序**。
    #: ViT 是 (0..23)；SwinV2 只有 4 个 ``layers``；ResNet 是 1-indexed 的 (1,2,3,4)。
    #: 有了它，``lora_last_n_blocks`` 才能在三类骨干上表达同一个意思（"后百分之几"）。
    lora_groups: Tuple[int, ...] = ()
    #: 该骨干上要替换的模块名（最后一级）。CNN 没有 qkv/proj，得换成卷积层名。
    lora_targets: Tuple[str, ...] = ("qkv", "proj")

    def resolve_split(self, requested: Optional[int], vit_depth: int = 24) -> int:
        """把请求的 FSR 切点换算成本骨干的合法 stage index。

        A12 用同一条命令换骨干，不能要求用户为每个骨干手填切点。
        ``fsr_after_block=6``（ViT-L 24 层的四分之一处）换到只有 4 个 stage 的
        ResNet 上要变成 1，否则直接越界报错。
        """
        if requested is None:
            return self.default_split
        if 1 <= requested < self.depth:
            return requested
        scaled = max(1, min(self.depth - 1, round(self.depth * requested / vit_depth)))
        LOGGER.warning(
            "FSR 切点 %d 超出骨干 %s 的范围 [1, %d]，按比例换算为 %d",
            requested, self.name, self.depth - 1, scaled,
        )
        return scaled

    def resolve_lora_blocks(self, last_n: int) -> Tuple[int, ...]:
        """把 "最后 last_n 个 block" 换算成本骨干的 stage index 集合。

        ViT-L 上 ``last_n=12`` 就是字面的后 12 个 block。换到只有 4 个 stage 的
        SwinV2 上，字面理解会一个都匹配不上（index 只到 3），所以按**比例**换算：
        12/24 = 后 50% → SwinV2 取后 2 个 stage。
        """
        groups = self.lora_groups or tuple(range(self.depth))
        frac = min(1.0, max(last_n, 1) / max(self.depth, 1))
        k = max(1, round(len(groups) * frac))
        return tuple(groups[-k:])


class BackboneAdapter(nn.Module):
    """统一接口的基类。子类实现 ``embed`` / ``run`` / ``pool``。"""

    spec: BackboneSpec

    def __init__(self, model: nn.Module, spec: BackboneSpec) -> None:
        super().__init__()
        self.model = model
        self.spec = spec

    # -------------------------- 必须实现 -------------------------- #
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def run(self, h: torch.Tensor, start: int, end: Optional[int]) -> torch.Tensor:
        raise NotImplementedError

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        """深层特征 → ``(B, embed_dim)``。"""
        raise NotImplementedError

    # -------------------------- 通用实现 -------------------------- #
    @torch.no_grad()
    def probe_shallow_dim(self, split: int, img_size: int) -> int:
        """实测**给定切点处**的通道数，并写回 ``spec.shallow_dim``。

        必须用实际使用的 split 来探测，不能用 ``spec.default_split``：
        Swin 的通道数逐 stage 翻倍，切点差一个 stage 维度就差一倍，
        FSR 的门控维度会和特征维度对不上（曾经真的踩到，见测试
        ``test_backbone_adapters_all_work``）。
        """
        was_training = self.training
        self.eval()
        h = self.embed(torch.zeros(1, 3, img_size, img_size))
        h = self.run(h, 0, split)
        dim = (
            h.shape[-1]
            if self.spec.layout in (LAYOUT_TOKENS, LAYOUT_NHWC)
            else h.shape[1]
        )
        self.train(was_training)
        self.spec.shallow_dim = int(dim)
        return int(dim)

    def pool_shallow(self, h: torch.Tensor) -> torch.Tensor:
        """浅层特征 → ``(B, shallow_dim)``，供浅层风格原型使用。"""
        layout = self.spec.layout
        if layout == LAYOUT_TOKENS:
            return h[:, self.spec.num_prefix_tokens :].mean(dim=1)
        if layout == LAYOUT_NHWC:
            return h.mean(dim=(1, 2))
        return h.mean(dim=(2, 3))

    def layernorm_param_names(self) -> List[str]:
        """哪些参数算「本地 LayerNorm」。

        CNN 骨干没有 LayerNorm，用 BatchNorm —— 那就把 BN 的 affine 当本地参数，
        这正好是原版 FedBN 的做法，语义是一致的。
        """
        names = []
        for mod_name, mod in self.model.named_modules():
            if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                for p_name, _ in mod.named_parameters(recurse=False):
                    names.append(f"{mod_name}.{p_name}" if mod_name else p_name)
        return names


# --------------------------------------------------------------------------- #
class ViTAdapter(BackboneAdapter):
    """timm ``VisionTransformer``（RETFound / DINOv2 / 普通 ViT 都走这条）。"""

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        h = m.patch_embed(x)
        h = m._pos_embed(h)
        if hasattr(m, "patch_drop"):
            h = m.patch_drop(h)
        if hasattr(m, "norm_pre"):
            h = m.norm_pre(h)
        return h

    def run(self, h: torch.Tensor, start: int, end: Optional[int]) -> torch.Tensor:
        for blk in self.model.blocks[start:end]:
            h = blk(h)
        return h

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        h = self.model.norm(h)
        return h[:, 0]  # CLS


class SwinAdapter(BackboneAdapter):
    """timm ``SwinTransformerV2``。

    两个必须处理的差异：

    1. block 藏在 ``layers[i].blocks[j]``，要拍平成一个列表；但 **PatchMerging 在
       ``layers[i].downsample``**，拍平时必须把 downsample 一起串进去，否则维度对不上。
    2. 通道数逐 stage 翻倍（128→256→512→1024），所以 ``shallow_dim != embed_dim``。
       FSR 的门控维度要用 ``shallow_dim``。
    """

    def __init__(self, model: nn.Module, spec: BackboneSpec) -> None:
        super().__init__(model, spec)
        # 把 [downsample?, block, block, ...] 按 stage 顺序拍平。
        # 存普通 list 而不是 nn.ModuleList：这些模块已经通过 self.model 注册过了，
        # 再注册一遍会让参数在 named_parameters() 里出现两条路径，
        # 进而让「本地 LayerNorm」的名单匹配不上。
        flat: List[nn.Module] = []
        for layer in model.layers:
            if getattr(layer, "downsample", None) is not None and not isinstance(
                layer.downsample, nn.Identity
            ):
                flat.append(layer.downsample)
            flat.extend(list(layer.blocks))
        self.flat_stages = flat

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.patch_embed(x)

    def run(self, h: torch.Tensor, start: int, end: Optional[int]) -> torch.Tensor:
        for mod in self.flat_stages[start:end]:
            h = mod(h)
        return h

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        h = self.model.norm(h)
        return h.mean(dim=(1, 2)) if h.dim() == 4 else h.mean(dim=1)


class CNNAdapter(BackboneAdapter):
    """timm ResNet 系列。FSR 在这里回到 FedBCS 原文的 CNN 特征图形式。"""

    def __init__(self, model: nn.Module, spec: BackboneSpec) -> None:
        super().__init__(model, spec)
        # 同 SwinAdapter：用普通 list 避免模块被重复注册
        self.stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        h = m.conv1(x)
        h = m.bn1(h)
        h = m.act1(h)
        return m.maxpool(h)

    def run(self, h: torch.Tensor, start: int, end: Optional[int]) -> torch.Tensor:
        for stage in self.stages[start:end]:
            h = stage(h)
        return h

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=(2, 3))


# --------------------------------------------------------------------------- #
def build_backbone_adapter(
    name: str,
    img_size: int = 224,
    pretrained: bool = False,
    drop_path_rate: float = 0.0,
) -> BackboneAdapter:
    """按名字造骨干并包上适配器。

    Args:
        name: timm 模型名，或 ``'debug_vit'``（内置微型 ViT，供冒烟测试）。
        img_size: 输入边长。CNN 骨干会忽略这个参数（它不需要）。

    Raises:
        ImportError: 没装 timm。
        ValueError: 不认识的骨干家族。
    """
    if name == "debug_vit":
        from .retfound_lora import _DebugViT  # 循环导入放在函数内

        model = _DebugViT(img_size=img_size, embed_dim=64, depth=8, num_heads=4)
        spec = BackboneSpec(
            name=name, embed_dim=64, depth=8, num_prefix_tokens=1,
            layout=LAYOUT_TOKENS, shallow_dim=64,
            lora_block_regex=r"blocks\.(\d+)\.", default_split=4,
            lora_groups=tuple(range(8)), lora_targets=("qkv", "proj"),
        )
        return ViTAdapter(model, spec)

    try:
        import timm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要 timm：pip install 'timm>=0.9.16'\n"
            "（想先跑通流程可以用 backbone='debug_vit'）"
        ) from exc

    lower = name.lower()
    is_cnn = any(k in lower for k in ("resnet", "resnext", "convnext", "efficientnet"))

    kwargs = dict(pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
    if not is_cnn:
        kwargs["img_size"] = img_size
    model = timm.create_model(name, **kwargs)

    if is_cnn:
        if not hasattr(model, "layer1"):
            raise ValueError(f"CNN 骨干 {name} 没有 layer1..layer4，暂不支持")
        dims = [
            model.layer1[-1].bn3.num_features if hasattr(model.layer1[-1], "bn3")
            else model.layer1[-1].bn2.num_features,
            model.num_features,
        ]
        spec = BackboneSpec(
            name=name, embed_dim=int(model.num_features), depth=4, num_prefix_tokens=0,
            layout=LAYOUT_NCHW, shallow_dim=int(dims[0]),
            lora_block_regex=r"layer(\d+)\.", default_split=2,
            # ResNet 的 layer1..layer4 是 1-indexed
            lora_groups=(1, 2, 3, 4),
            # 卷积瓶颈块里的三个卷积，走 LoRAConv2d
            lora_targets=("conv1", "conv2", "conv3"),
        )
        adapter: BackboneAdapter = CNNAdapter(model, spec)

    elif hasattr(model, "layers") and hasattr(model.layers[0], "blocks"):
        # SwinV2：先拍平数 stage 数，再探测浅层维度
        n_flat = 0
        for layer in model.layers:
            ds = getattr(layer, "downsample", None)
            if ds is not None and not isinstance(ds, nn.Identity):
                n_flat += 1
            n_flat += len(layer.blocks)
        spec = BackboneSpec(
            name=name, embed_dim=int(model.num_features), depth=n_flat,
            num_prefix_tokens=0, layout=LAYOUT_NHWC,
            shallow_dim=int(model.embed_dim),  # 先填 stage0 的维度，下面实测修正
            lora_block_regex=r"layers\.(\d+)\.", default_split=max(1, n_flat // 4),
            lora_groups=tuple(range(len(model.layers))),
            # 注意：SwinV2 的 attention 里是 `F.linear(x, weight=self.qkv.weight, ...)`
            # 直接读 .weight（它要手动拼 cosine attention 的 qkv_bias），
            # 所以 qkv 不能被 LoRALinear 包住，否则报 'LoRALinear' has no attribute 'weight'。
            # 改打 attn.proj 与 MLP 的 fc1/fc2，可训练参数量与 ViT 方案同量级。
            lora_targets=("proj", "fc1", "fc2"),
        )
        adapter = SwinAdapter(model, spec)
        # 这里只是给个初值；真正的维度由 FedOSPNet 在确定 split 之后重新探测
        adapter.probe_shallow_dim(spec.default_split, img_size)

    elif hasattr(model, "blocks"):
        spec = BackboneSpec(
            name=name, embed_dim=int(model.embed_dim), depth=len(model.blocks),
            num_prefix_tokens=int(getattr(model, "num_prefix_tokens", 1)),
            layout=LAYOUT_TOKENS, shallow_dim=int(model.embed_dim),
            lora_block_regex=r"blocks\.(\d+)\.", default_split=6,
            lora_groups=tuple(range(len(model.blocks))),
            lora_targets=("qkv", "proj"),
        )
        adapter = ViTAdapter(model, spec)

    else:
        raise ValueError(
            f"不认识骨干 {name} 的结构（既没有 blocks 也没有 layers/layer1）。"
            "请在 backbones.py 里加一个对应的 Adapter。"
        )

    LOGGER.info(
        "骨干 %s：depth=%d embed_dim=%d shallow_dim=%d layout=%s 默认 FSR 切点=%d",
        adapter.spec.name, adapter.spec.depth, adapter.spec.embed_dim,
        adapter.spec.shallow_dim, adapter.spec.layout, adapter.spec.default_split,
    )
    return adapter


def spatial_grid(h: torch.Tensor, layout: str, num_prefix: int) -> Tuple[int, int]:
    """返回浅层特征的空间网格大小，用于报告 FSR 的频率分辨率。"""
    if layout == LAYOUT_TOKENS:
        n = h.shape[1] - num_prefix
        side = int(round(n ** 0.5))
        return side, side
    if layout == LAYOUT_NHWC:
        return int(h.shape[1]), int(h.shape[2])
    return int(h.shape[2]), int(h.shape[3])


__all__ = [
    "LAYOUT_NCHW",
    "LAYOUT_NHWC",
    "LAYOUT_TOKENS",
    "BackboneAdapter",
    "BackboneSpec",
    "CNNAdapter",
    "SwinAdapter",
    "ViTAdapter",
    "build_backbone_adapter",
    "spatial_grid",
]
