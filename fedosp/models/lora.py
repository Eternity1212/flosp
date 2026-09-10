"""极简 LoRA：只对 ``nn.Linear`` 打补丁，不引入额外依赖。

之所以不直接用 peft：本方案需要精确控制「哪些参数上传、哪些留本地」
（LoRA 上传、LayerNorm 不上传），自己实现一层最省心，也方便审稿时说清楚。

默认打在**最后 12 个 block 的 ``qkv`` 与 ``proj``**（方案 4.1）。
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger("lora")


class LoRALinear(nn.Module):
    """``y = W0 x + b + (alpha/r) * B(A(dropout(x)))``，其中 W0 冻结。"""

    def __init__(
        self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank 必须 > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        # A 用 Kaiming、B 置零 → 初始时 LoRA 分支输出恒为 0，等价于原模型
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * delta

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.r}, scaling={self.scaling:.2f}"
        )


class LoRAConv2d(nn.Module):
    """Conv2d 版 LoRA，供 A12 的 CNN 骨干对照使用。

    没有这个的话 ResNet50 骨干上找不到 ``qkv``/``proj`` 这类 Linear，
    对照就退化成「只训分类头」，和 ViT 训 LoRA 不是同一个量级，A12 就失去意义了。

    分解方式：``lora_A`` 是一个 ``in -> r`` 的卷积（承担原 conv 的 stride/padding/dilation，
    保证 delta 与 base 输出同形状），``lora_B`` 是 ``r -> out`` 的 1x1 卷积。
    """

    def __init__(
        self, base: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.05
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank 必须 > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Conv2d(
            base.in_channels, r, base.kernel_size, base.stride,
            base.padding, base.dilation, groups=1, bias=False,
        )
        self.lora_B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # 初始 delta 恒为 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_channels}, out={self.base.out_channels}, "
            f"k={self.base.kernel_size}, r={self.r}, scaling={self.scaling:.2f}"
        )


def _set_module(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def inject_lora(
    model: nn.Module,
    target_suffixes: Sequence[str] = ("qkv", "proj"),
    block_filter: Sequence[int] = (),
    block_regex: str = r"blocks\.(\d+)\.",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> List[str]:
    """就地把匹配的 ``nn.Linear`` 换成 :class:`LoRALinear`，返回被替换的模块名列表。

    Args:
        target_suffixes: 只替换**最后一级名字**等于其中之一的 Linear。
            这里用精确相等而不是 ``endswith``：``nn.MultiheadAttention`` 内部有个
            ``out_proj``，用 endswith('proj') 会把它一起换掉，然后 MHA 取
            ``out_proj.weight`` 就会崩。
        block_filter: 只在这些 block index 上替换；空元组表示不限制。
        block_regex: 从模块名里抠 block index 的正则。
    """
    pattern = re.compile(block_regex)
    wanted = set(block_filter)
    targets = set(target_suffixes)
    replaced: List[str] = []

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            wrapper = LoRALinear
        elif isinstance(module, nn.Conv2d):
            wrapper = LoRAConv2d
        else:
            continue
        if name.split(".")[-1] not in targets:
            continue
        if wanted:
            m = pattern.search(name)
            if m is None or int(m.group(1)) not in wanted:
                continue
        _set_module(model, name, wrapper(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)

    if not replaced:
        LOGGER.warning(
            "inject_lora 没有替换任何层！检查 target_suffixes=%s / block_regex=%r 是否匹配骨干命名",
            target_suffixes, block_regex,
        )
    else:
        LOGGER.info("LoRA 已注入 %d 层（r=%d, alpha=%d），例如 %s",
                    len(replaced), r, alpha, replaced[:3])
    return replaced


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """只取 LoRA 的 A/B，用于上传。"""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".lora_A" in k or ".lora_B" in k
    }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回 ``(可训练参数量, 总参数量)``。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_all_but(model: nn.Module, keywords: Iterable[str]) -> None:
    """冻结全部参数，只解冻名字里含任一 keyword 的（方案 4.1 的冻结策略）。"""
    keywords = list(keywords)
    n_train = 0
    for name, p in model.named_parameters():
        keep = any(k in name for k in keywords)
        p.requires_grad_(keep)
        n_train += p.numel() if keep else 0
    LOGGER.info("冻结完成，可训练参数 %.2fM（关键词 %s）", n_train / 1e6, keywords)
