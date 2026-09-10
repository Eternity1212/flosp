"""双层原型库：浅层风格原型 + 深层等级原型（方案 4.4）。

两层原型的分工：

* ``shallow``：FSR 之后 patch token 的池化特征的类均值 —— 用来**跨 client 对齐风格**
* ``deep``：CLS 特征的类均值 —— 用来**承载 0→4 的有序几何**

原型都用 EMA 累积（动量 0.9），而不是每步重算。原因：IDRiD 一个 batch 可能只有
1~2 张 grade 1，直接用 batch 均值当原型会剧烈抖动。

原型以 ``buffer`` 形式存在，不是 ``Parameter`` —— 它们由统计量更新、由服务器聚合，
不参与本地梯度下降。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("prototypes")


class PrototypeBank(nn.Module):
    """维护 ``(num_classes, dim)`` 的一组原型，支持 EMA 本地更新与服务器覆写。

    Args:
        num_classes: DR 等级数，5。
        dim: 特征维度。
        momentum: EMA 动量，越大越平滑。
        normalize: 是否在更新与比较前做 L2 归一化（ordinal margin 依赖归一化后的尺度）。
    """

    def __init__(
        self,
        num_classes: int = 5,
        dim: int = 1024,
        momentum: float = 0.9,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.normalize = normalize
        self.register_buffer("proto", torch.zeros(num_classes, dim))
        # seen[c] = 该 client 是否见过类别 c；上传给服务器决定「哪些 client 参与该类平均」
        self.register_buffer("seen", torch.zeros(num_classes, dtype=torch.bool))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update(self, feat: torch.Tensor, labels: torch.Tensor) -> None:
        """用一个 batch 的特征做 EMA 更新。``feat: (B, D)``, ``labels: (B,)``。"""
        f = F.normalize(feat.detach().float(), dim=-1) if self.normalize else feat.detach().float()
        for c in labels.unique():
            c_int = int(c)
            mask = labels == c
            if not bool(mask.any()):
                continue
            batch_mean = f[mask].mean(dim=0)
            if bool(self.seen[c_int]):
                self.proto[c_int].mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
            else:
                # 首次见到该类：直接赋值，避免从 0 向量慢慢爬
                self.proto[c_int].copy_(batch_mean)
                self.seen[c_int] = True
        if self.normalize:
            self.proto.copy_(F.normalize(self.proto, dim=-1))

    @torch.no_grad()
    def load_global(self, global_proto: torch.Tensor) -> None:
        """服务器下发的全局原型覆写本地（只覆写服务器实际有值的类）。"""
        valid = global_proto.abs().sum(dim=-1) > 0
        self.proto[valid] = global_proto[valid].to(self.proto.device, self.proto.dtype)

    def export(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """导出给服务器：``(proto, seen)``。seen 决定 client 等权平均的分母。"""
        return self.proto.detach().cpu().clone(), self.seen.detach().cpu().clone()

    def distances(self, feat: torch.Tensor) -> torch.Tensor:
        """样本到各原型的平方欧氏距离，``(B, C)``。"""
        f = F.normalize(feat, dim=-1) if self.normalize else feat
        return torch.cdist(f, self.proto.to(f.dtype)).pow(2)

    def is_ready(self) -> bool:
        return bool(self.seen.any())

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, dim={self.dim}, "
            f"momentum={self.momentum}, normalize={self.normalize}"
        )


@torch.no_grad()
def aggregate_prototypes(
    protos: Dict[str, torch.Tensor],
    seens: Dict[str, torch.Tensor],
    sizes: Optional[Dict[str, int]] = None,
    mode: str = "client_equal",
) -> torch.Tensor:
    """服务器端原型聚合（方案 4.5，对应消融 A6）。

    Args:
        protos: ``client -> (C, D)``
        seens: ``client -> (C,) bool``，标记该 client 是否有该类样本
        sizes: ``client -> n_k``，只有非 client_equal 模式才需要
        mode: ``client_equal`` / ``sqrt`` / ``sample``

    Returns:
        ``(C, D)`` 全局原型。没有任何 client 拥有的类别保持为 0 向量。

    **``client_equal`` 是本文默认**：EyePACS 的 24,600 张不能把 IDRiD 的 372 张淹没。
    只对「真的有该类样本」的 client 求平均，避免没见过 grade 1 的 client 拿 0 向量拉低均值。
    """
    keys = list(protos.keys())
    if not keys:
        raise ValueError("aggregate_prototypes 收到空的 protos")
    c, d = protos[keys[0]].shape
    out = torch.zeros(c, d, dtype=torch.float32)

    for cls in range(c):
        owners = [k for k in keys if bool(seens[k][cls])]
        if not owners:
            continue
        if mode == "client_equal":
            w = torch.ones(len(owners))
        elif mode == "sample":
            if sizes is None:
                raise ValueError("mode='sample' 需要提供 sizes")
            w = torch.tensor([float(sizes[k]) for k in owners])
        elif mode == "sqrt":
            if sizes is None:
                raise ValueError("mode='sqrt' 需要提供 sizes")
            w = torch.tensor([float(sizes[k]) for k in owners]).sqrt()
        else:
            raise ValueError(f"未知原型聚合模式: {mode}")
        w = w / w.sum()
        stacked = torch.stack([protos[k][cls].float() for k in owners])
        out[cls] = (w[:, None] * stacked).sum(dim=0)

    out = F.normalize(out, dim=-1)
    out[out.isnan()] = 0.0
    n_valid = int((out.abs().sum(dim=-1) > 0).sum())
    LOGGER.debug("原型聚合 mode=%s，有效类别 %d/%d", mode, n_valid, c)
    return out
