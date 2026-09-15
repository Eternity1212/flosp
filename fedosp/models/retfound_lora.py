"""FedOSP 主干模型：冻结 RETFound ViT-L + LoRA + FSR + 双层原型 + 本地 LayerNorm。

对应方案 4.1 / 4.2 / 4.7。

参数分三类，**这个划分是本方案的核心，改动前先想清楚**：

=========================  ==========  =========================================
参数                        是否训练     是否上传服务器
=========================  ==========  =========================================
ViT-L 骨干权重              否          否（全程冻结，~303M）
LoRA A/B                   是          **是**
分类头                      是          **是**
FSR 门控 gate_logit         是          **是**
Visual prompt（仅 B11）     是          **是**
双层原型                    统计更新     **是**（client 等权聚合）
LayerNorm / BatchNorm 的仿射  是          **否（留本地，FedBN 精神）**
=========================  ==========  =========================================

实测（timm 1.0.27，ViT-L/16 @224，LoRA r=8 打最后 12 个 block）：
可训练 **0.70 M / 303.90 M = 0.23%**，单轮上传 **2.31 MB/client**。
对比全量同步的 1.2 GB，通信量降低约 **520 倍**。

骨干无关性由 :mod:`fedosp.models.backbones` 的适配层保证，
所以消融 A12 换 SwinV2 / ResNet50 / DINOv2 不需要改这个文件。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import (
    LAYOUT_TOKENS,
    BackboneAdapter,
    build_backbone_adapter,
)
from .fsr import build_fsr
from .lora import count_parameters, inject_lora
from .prototypes import PrototypeBank

LOGGER = logging.getLogger("retfound_lora")


@dataclass
class FedOSPConfig:
    """一处改、处处生效的模型配置（与方案第 6 节超参表对应）。"""

    backbone: str = "vit_large_patch16_224"
    num_classes: int = 5
    img_size: int = 224
    #: FSR 插在第几个 stage 之后。None = 用骨干的默认切点（ViT 是 6）
    fsr_after_block: Optional[int] = 6
    use_fsr: bool = True                 # A1 消融置 False
    use_shallow_proto: bool = True        # A2 消融置 False
    use_deep_proto: bool = True           # A3 消融置 False
    lora_rank: int = 8                    # A8 扫 {4,8,16,32}
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    #: LoRA 打在最后几个 stage；A8 对比 6 / 12 / 24
    lora_last_n_blocks: int = 12
    lora_targets: Sequence[str] = ("qkv", "proj")
    proto_momentum: float = 0.9
    #: LayerNorm 是否留在本地不上传；A7 消融置 False
    personal_layernorm: bool = True
    pretrained_path: Optional[str] = None
    #: 骨干是否用 timm 自带的 ImageNet 预训练（RETFound 权重没批下来时的临时替代）
    imagenet_pretrained: bool = False
    #: ``"main"`` = 正式实验，随机初始化骨干会**直接报错**；``"smoke"`` = 允许（CI / 调试）。
    #: 这道闸门的存在理由见 docs：上一个项目曾用 fallback 小模型产出过看起来正常的假结果。
    stage: str = "smoke"
    drop_path_rate: float = 0.0
    #: B10 基线：解冻整个骨干做全量微调（会让上传量涨到 ~1.2 GB）
    full_finetune: bool = False
    #: B11 基线：visual prompt tuning 的 prompt token 数；0 = 关闭
    num_prompts: int = 0
    #: 序数输出头（B17 交叉组）。``"none"`` = 标准 K 类 softmax 头；
    #: ``"ordinal_encoding"`` = K-1 个**独立**阈值 logit；
    #: ``"coral"`` = K-1 个阈值**共享同一权重向量**、只有偏置不同（秩单调性由构造保证）。
    #: 非 ``"none"`` 时 ``ForwardOutput.logits`` 是 (B, K-1)，
    #: 用 ``losses.ordinal_logits_to_probs`` 转回 K 类概率后下游指标完全复用。
    ordinal_head: str = "none"


@dataclass
class ForwardOutput:
    """一次前向的全部产物，避免用元组导致下游取错位置。"""

    logits: torch.Tensor            # (B, C)
    deep_feat: torch.Tensor         # (B, embed_dim) 深层池化特征
    shallow_feat: torch.Tensor      # (B, shallow_dim) FSR 后的浅层池化特征
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def load_retfound_weights(model: nn.Module, path: str) -> None:
    """加载 RETFound MAE 权重，容忍 ``model.`` 前缀与缺失的分类头/解码器。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "teacher"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    ckpt = {k.replace("module.", "").replace("model.", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    LOGGER.info(
        "RETFound 权重已加载：missing=%d unexpected=%d（head/decoder 不匹配是正常的）",
        len(missing), len(unexpected),
    )
    critical = [k for k in missing if k.startswith(("patch_embed", "blocks"))]
    if critical:
        raise RuntimeError(
            f"RETFound 权重与骨干结构不匹配，关键层缺失 {len(critical)} 个，例如 {critical[:5]}。"
            "确认下载的是 RETFound_mae_natureCFP（ViT-Large/16）而不是别的变体。"
        )


# --------------------------------------------------------------------------- #
class CoralHead(nn.Module):
    """CORAL 输出头（Cao, Mirjalili & Raschka 2020）：共享权重 + K-1 个独立偏置。

    .. math:: z_k = w^\\top f + b_k,\\qquad k=1,\\dots,K-1

    只有偏置不同，所以任意样本上 $z_1,\\dots,z_{K-1}$ 的**大小顺序完全由 $b$ 决定、
    与样本无关**。这就是 CORAL 的 rank-consistency：不会出现
    $\\sigma(z_2)>\\sigma(z_1)$ 这种"不是 >1 却是 >2"的自相矛盾预测。

    代价是表达力比 K-1 个独立线性头弱（所有阈值共用一个方向），这正是
    ``ordinal_encoding`` 与它的取舍，B17 里两者都跑就是为了量出这个取舍。

    参数量：``embed_dim + (K-1)``，比标准 K 类头（``embed_dim*K + K``）还少。
    """

    def __init__(self, dim: int, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.zeros(1, dim))
        # 偏置初始化成递减序列，让初始预测落在中间等级而不是全 0 或全 4
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        nn.init.trunc_normal_(self.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # (B, dim) @ (dim, 1) -> (B, 1)，再广播加 K-1 个偏置
        return feat @ self.weight.t() + self.bias

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[1]}, thresholds={self.num_classes - 1}, shared_weight=True"


class FedOSPNet(nn.Module):
    """FedOSP 网络：``forward`` 返回 :class:`ForwardOutput`。"""

    def __init__(self, cfg: FedOSPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- 骨干（经适配层统一接口，ViT / Swin / CNN 都走这条）----
        self.adapter: BackboneAdapter = build_backbone_adapter(
            cfg.backbone,
            img_size=cfg.img_size,
            pretrained=cfg.imagenet_pretrained,
            drop_path_rate=cfg.drop_path_rate,
        )
        spec = self.adapter.spec
        self.embed_dim = spec.embed_dim
        self.shallow_dim = spec.shallow_dim
        self.depth = spec.depth
        self.layout = spec.layout
        self.num_prefix = spec.num_prefix_tokens

        self.weight_source = self._init_backbone_weights()

        # 换骨干时切点自动按比例换算（A12 用同一条命令跑三个骨干）
        self.split = spec.resolve_split(cfg.fsr_after_block)
        # 浅层维度必须在**实际切点**处实测：Swin 的通道数逐 stage 翻倍，
        # 用 default_split 探测出来的值在切点不同时会差一倍，FSR 门控就对不上了。
        if spec.layout != LAYOUT_TOKENS:
            self.shallow_dim = self.adapter.probe_shallow_dim(self.split, cfg.img_size)

        # ---- FSR：作用在浅层，通道数用 shallow_dim（Swin 的浅深维度不同）----
        self.fsr = build_fsr(
            cfg.use_fsr, self.shallow_dim,
            layout=spec.layout, num_prefix_tokens=spec.num_prefix_tokens,
        )

        # ---- B11：visual prompt tuning（只对 token 排布有意义）----
        self.prompts: Optional[nn.Parameter] = None
        if cfg.num_prompts > 0:
            if spec.layout != LAYOUT_TOKENS:
                raise ValueError(f"prompt tuning 只支持 token 排布的骨干，{cfg.backbone} 不行")
            self.prompts = nn.Parameter(torch.zeros(1, cfg.num_prompts, self.embed_dim))
            nn.init.trunc_normal_(self.prompts, std=0.02)
            LOGGER.info("已启用 visual prompt tuning：%d 个 prompt token", cfg.num_prompts)

        self.head = self._build_head()

        self.shallow_proto = PrototypeBank(cfg.num_classes, self.shallow_dim, cfg.proto_momentum)
        self.deep_proto = PrototypeBank(cfg.num_classes, self.embed_dim, cfg.proto_momentum)
        if cfg.ordinal_head != "none":
            LOGGER.info(
                "序数输出头 %r 已启用：logits 是 (B, %d)，下游需经 "
                "ordinal_logits_to_probs 转回 %d 类概率",
                cfg.ordinal_head, cfg.num_classes - 1, cfg.num_classes,
            )

        # ---- 本地参数名单（LayerNorm / BatchNorm 的仿射）----
        self._personal_keys = {
            f"adapter.model.{n}" for n in self.adapter.layernorm_param_names()
        }

        if not cfg.full_finetune:
            self._inject_lora()
        else:
            LOGGER.warning("full_finetune=True：骨干解冻，上传量将回到 ~1.2 GB（这是 B10 基线）")
        self._freeze()

        trainable, total = count_parameters(self)
        LOGGER.info(
            "FedOSPNet 就绪：%s depth=%d deep_dim=%d shallow_dim=%d | "
            "可训练 %.2fM / 总 %.2fM (%.2f%%) | 单轮上传 %.2f MB",
            cfg.backbone, self.depth, self.embed_dim, self.shallow_dim,
            trainable / 1e6, total / 1e6, 100 * trainable / max(total, 1),
            self.communication_mb(),
        )

    # ---------------------------- 构建辅助 ---------------------------- #
    def _build_head(self) -> nn.Module:
        """按 ``cfg.ordinal_head`` 造分类头。

        三种形态（B17 交叉组用后两种）：

        ==================  ===========  ==============================================
        ``ordinal_head``    输出维度      结构
        ==================  ===========  ==============================================
        ``none``            ``K``        标准 ``Linear``，配 softmax
        ``ordinal_encoding`` ``K-1``     标准 ``Linear``，K-1 个**独立**阈值
        ``coral``           ``K-1``      **共享权重向量** + K-1 个独立偏置
        ==================  ===========  ==============================================

        CORAL 的共享权重不是实现上的偷懒，而是它秩单调性的**唯一来源**：所有阈值
        logit 都是 $w^\\top f + b_k$，彼此只差常数偏置，于是 $\\sigma(z_k)$ 的排序
        与 $b_k$ 的排序恒等，不可能出现"不是 >1 却是 >2"的自相矛盾（Cao et al. 2020）。
        若退化成普通 ``Linear(dim, K-1)``，这个保证就没了 —— 那就是 ordinal_encoding。
        """
        cfg = self.cfg
        mode = cfg.ordinal_head
        if mode not in ("none", "coral", "ordinal_encoding"):
            raise ValueError(
                f"未知 ordinal_head={mode!r}，可选 none / coral / ordinal_encoding"
            )
        if mode == "coral":
            return CoralHead(self.embed_dim, cfg.num_classes)
        out_dim = cfg.num_classes if mode == "none" else cfg.num_classes - 1
        head = nn.Linear(self.embed_dim, out_dim)
        nn.init.trunc_normal_(head.weight, std=0.01)
        nn.init.zeros_(head.bias)
        return head

    def _init_backbone_weights(self) -> str:
        """加载骨干权重，并返回权重来源标识（会写进 ``result.json`` 供事后审计）。

        ``stage="main"`` 下随机初始化骨干是 **硬错误**：随机骨干照样能跑完 100 轮并产出
        格式完全正常的 ``result.json``，这种结果在事后极难分辨，因此必须在启动时就拦住。
        ``imagenet`` 是方案 R-B 里写明的降级方案，允许但会显著告警并记录来源。
        """
        cfg = self.cfg
        if cfg.pretrained_path:
            load_retfound_weights(self.adapter.model, cfg.pretrained_path)
            return f"retfound:{cfg.pretrained_path}"

        if cfg.backbone == "debug_vit":
            return "random:debug_vit"

        if cfg.imagenet_pretrained:
            LOGGER.warning(
                "骨干用的是 ImageNet 预训练，不是 RETFound。这是方案 R-B 的降级路径，"
                "论文里必须声明，且不能再声称『眼科基础模型』。"
            )
            return "imagenet"

        msg = (
            f"骨干 {cfg.backbone!r} 是**随机初始化**的。stage='main' 下这被视为致命错误：\n"
            "  - 正式实验请传 --pretrained <RETFound_mae_natureCFP 权重路径>\n"
            "  - 若 RETFound 权重尚未获批，显式传 --imagenet-pretrained（方案 R-B 降级路径）\n"
            "  - 仅调试流程请用 --stage smoke 或 --dry-run\n"
            "拦截原因：随机骨干同样能跑完并产出格式正常的 result.json，事后无法分辨。"
        )
        if cfg.stage == "main":
            raise RuntimeError(msg)
        LOGGER.warning("%s\n（当前 stage=%r，仅告警放行）", msg, cfg.stage)
        return "random"

    def _inject_lora(self) -> None:
        spec = self.adapter.spec
        blocks = spec.resolve_lora_blocks(self.cfg.lora_last_n_blocks)
        # 目标层名以骨干为准：ViT/Swin 是 qkv/proj，ResNet 是 conv1/2/3。
        # 只有用户显式改过 cfg.lora_targets 时才覆盖骨干默认值。
        targets = (
            tuple(self.cfg.lora_targets)
            if tuple(self.cfg.lora_targets) != ("qkv", "proj")
            else spec.lora_targets
        )
        replaced = inject_lora(
            self.adapter.model,
            target_suffixes=targets,
            block_filter=blocks,
            block_regex=spec.lora_block_regex,
            r=self.cfg.lora_rank,
            alpha=self.cfg.lora_alpha,
            dropout=self.cfg.lora_dropout,
        )
        if not replaced:
            raise RuntimeError(
                f"骨干 {self.cfg.backbone} 上没有注入任何 LoRA 层"
                f"（targets={targets}, blocks={blocks}, regex={spec.lora_block_regex!r}）。"
                "这会让实验退化成「只训分类头」，与其它骨干不可比 —— 请在 backbones.py 里"
                "为这个骨干补上正确的 lora_targets。"
            )

    def _freeze(self) -> None:
        """按方案 4.1 决定哪些参数训练。"""
        if self.cfg.full_finetune:
            for p in self.parameters():
                p.requires_grad_(True)
            return
        trainable_keys = ("lora_A", "lora_B", "head.", "gate_logit", "prompts")
        for name, p in self.named_parameters():
            is_norm = name in self._personal_keys
            p.requires_grad_(any(k in name for k in trainable_keys) or is_norm)

    # ------------------------------ 前向 ------------------------------ #
    def forward(self, x: torch.Tensor) -> ForwardOutput:
        h = self.adapter.embed(x)

        if self.prompts is not None:
            # VPT-Shallow：在 embed 之后插入可学习 prompt token
            h = torch.cat([h, self.prompts.expand(h.shape[0], -1, -1)], dim=1)

        h = self.adapter.run(h, 0, self.split)

        # ---- FSR：压掉浅层风格，保留相位承载的空间语义 ----
        if self.prompts is not None:
            # prompt token 不参与 FFT（它不在空间网格上），先摘出来
            n_p = self.cfg.num_prompts
            h_main, h_prompt = h[:, :-n_p], h[:, -n_p:]
            h_main = self.fsr(h_main)
            shallow_feat = self.adapter.pool_shallow(h_main)
            h = torch.cat([h_main, h_prompt], dim=1)
        else:
            h = self.fsr(h)
            shallow_feat = self.adapter.pool_shallow(h)

        h = self.adapter.run(h, self.split, None)
        deep_feat = self.adapter.pool(h)
        return ForwardOutput(
            logits=self.head(deep_feat), deep_feat=deep_feat, shallow_feat=shallow_feat
        )

    def class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """把本模型的原始输出统一转成 ``(B, K)`` 类别概率。

        **所有评估路径都必须走这个方法**，不要在外面自己写 ``softmax``。
        序数头（coral / ordinal_encoding）输出的是 K-1 个阈值 logit，直接 softmax
        会得到一个 4 维的、含义完全错误的"概率"，而且不会报错 —— QWK 照样能算出
        一个像样的数字。把转换收敛到这一个方法里，就不存在漏改某条路径的可能。
        """
        if self.cfg.ordinal_head == "none":
            return F.softmax(logits.float(), dim=-1)
        from ..losses import ordinal_logits_to_probs
        return ordinal_logits_to_probs(logits, self.cfg.num_classes)

    @torch.no_grad()
    def shallow_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """取 FSR **之前**的浅层特征，只给论文图 F3（幅度谱可视化）用。"""
        h = self.adapter.embed(x)
        return self.adapter.run(h, 0, self.split)

    @torch.no_grad()
    def update_prototypes(self, out: ForwardOutput, labels: torch.Tensor) -> None:
        if self.cfg.use_deep_proto:
            self.deep_proto.update(out.deep_feat, labels)
        if self.cfg.use_shallow_proto:
            self.shallow_proto.update(out.shallow_feat, labels)

    # -------------------- 联邦通信：参数切分 -------------------- #
    def _is_personal(self, name: str) -> bool:
        """LayerNorm / BatchNorm 的仿射参数留本地（方案 4.7）。"""
        return name in self._personal_keys

    def _is_shareable(self, name: str) -> bool:
        if self.cfg.full_finetune:
            # B10：除了本地 norm 之外全部上传
            return True
        return (
            ".lora_A" in name
            or ".lora_B" in name
            or name.startswith("head.")
            or "gate_logit" in name
            or name == "prompts"
        )

    def shared_state_dict(self) -> Dict[str, torch.Tensor]:
        """要上传给服务器的部分（不含原型，原型走单独通道）。"""
        keep = {}
        for k, v in self.state_dict().items():
            if k.startswith(("shallow_proto.", "deep_proto.")):
                continue
            if self.cfg.personal_layernorm and self._is_personal(k):
                continue
            if self._is_shareable(k):
                keep[k] = v.detach().cpu().clone()
        return keep

    def load_shared_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        _, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            LOGGER.warning("下发参数里有 %d 个本模型不认识的 key，例如 %s",
                           len(unexpected), list(unexpected)[:3])

    def personal_state_dict(self) -> Dict[str, torch.Tensor]:
        """留在本地、不上传的部分（各 client 自己的 LayerNorm / BatchNorm）。"""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if self._is_personal(k)
        }

    def communication_mb(self) -> float:
        """单轮上传量（MB），fp32 口径。写进方案第 9 节的系统指标表 T5。"""
        n = sum(v.numel() for v in self.shared_state_dict().values())
        n += self.shallow_proto.proto.numel() + self.deep_proto.proto.numel()
        return n * 4 / 1024 / 1024

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def param_report(self) -> Dict[str, float]:
        """给论文表 T5 用的一行数据。"""
        trainable, total = count_parameters(self)
        return {
            "backbone": self.cfg.backbone,
            "total_M": round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 3),
            "trainable_pct": round(100 * trainable / max(total, 1), 3),
            "upload_MB_per_round": round(self.communication_mb(), 3),
        }


# --------------------------------------------------------------------------- #
# 微型 ViT：只为在没有 GPU / timm 的机器上跑通全流程冒烟测试
# --------------------------------------------------------------------------- #
class _DebugAttn(nn.Module):
    """命名与 timm ``Attention`` 一致（``qkv`` / ``proj``），保证 LoRA 注入规则通用。"""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class _DebugBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _DebugAttn(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _DebugPatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.grid = img_size // patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _DebugViT(nn.Module):
    """结构接口与 timm VisionTransformer 对齐的最小实现。"""

    def __init__(self, img_size=224, patch=16, embed_dim=64, depth=8, num_heads=4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_prefix_tokens = 1
        self.patch_embed = _DebugPatchEmbed(img_size, patch, embed_dim)
        n = self.patch_embed.grid ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([_DebugBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def _pos_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1) + self.pos_embed


def build_model(cfg: Optional[FedOSPConfig] = None, **kwargs) -> FedOSPNet:
    """便捷入口：``build_model(backbone='debug_vit', use_fsr=False)``。"""
    if cfg is None:
        cfg = FedOSPConfig(**kwargs)
    elif kwargs:
        cfg = FedOSPConfig(**{**asdict(cfg), **kwargs})
    return FedOSPNet(cfg)
