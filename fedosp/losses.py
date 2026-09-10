"""损失函数（方案第 5 节）。

.. math::

    L = L_{CBCE} + 0.5 L_{ord} + 0.1 L_{proto-grade}
        + 0.1 L_{proto-style} + 0.05 L_{style} + 0.1 L_{cons}

四个核心项分别对应三个问题：

* ``L_CBCE``     —— EyePACS 里 73.5% 是 grade 0，必须按**本地**分布做类别平衡
* ``L_ord``      —— P2：0-4 是有序等级，把 3 判成 4 的代价应远小于把 0 判成 4
* ``L_proto_grade`` —— P2：让深层原型排成 0→1→2→3→4 的一维有序流形
* ``L_proto_style`` —— P1：把浅层特征拉向全局风格原型，逼出风格无关表征
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("losses")


# --------------------------------------------------------------------------- #
# 1. 类别平衡交叉熵
# --------------------------------------------------------------------------- #
def effective_number_weights(
    counts: Sequence[float],
    beta: float = 0.9999,
    clip: tuple = (0.1, 10.0),
) -> torch.Tensor:
    """Cui et al. 的 effective number 权重：``w_c = (1-beta) / (1-beta^{n_c})``。

    **必须传本地 client 的类别计数**，不是全局计数 —— 这是联邦设定下的正确做法。

    IDRiD 的 grade 1 只有 20 张，``1-beta^20`` 很小会让权重爆到几十倍，
    所以这里做了归一化 + clip，否则本地训练会被那 20 张图带跑。
    """
    counts = np.asarray(counts, dtype=np.float64)
    safe = np.maximum(counts, 1.0)
    w = (1.0 - beta) / (1.0 - np.power(beta, safe))
    w = w / w.sum() * len(w)          # 归一化到均值 1
    w = np.clip(w, clip[0], clip[1])
    w = w / w.sum() * len(w)          # clip 之后再归一一次
    w[counts == 0] = 0.0              # 本地没有的类别不产生梯度
    LOGGER.debug("CBCE 权重 %s（counts=%s）", np.round(w, 3), counts.astype(int))
    return torch.tensor(w, dtype=torch.float32)


class ClassBalancedCE(nn.Module):
    """带 effective-number 权重、可选 label smoothing 的交叉熵。"""

    def __init__(
        self,
        counts: Sequence[float],
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", effective_number_weights(counts, beta))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight.to(logits.dtype),
            label_smoothing=self.label_smoothing,
        )


# --------------------------------------------------------------------------- #
# 2. Ordinal：平方 EMD
# --------------------------------------------------------------------------- #
def squared_emd_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """平方 Earth Mover's Distance：累积分布之间的 L2 距离。

    .. math:: L_{ord} = \\frac{1}{K}\\sum_k \\Big(\\sum_{j\\le k} p_j - \\sum_{j\\le k} y_j\\Big)^2

    直觉：真值为 3 时，把概率放在 4 上只错一格、放在 0 上错三格，
    累积分布的差会自动把这个距离算进去。普通 CE 完全看不到这一层结构。
    """
    p = F.softmax(logits, dim=-1)
    y = F.one_hot(target, num_classes).to(p.dtype)
    return (p.cumsum(-1) - y.cumsum(-1)).pow(2).sum(-1).mean() / num_classes


# --------------------------------------------------------------------------- #
# 3. 原型损失
# --------------------------------------------------------------------------- #
def prototype_align_loss(
    feat: torch.Tensor, target: torch.Tensor, protos: torch.Tensor, valid: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """把样本拉向自己类别的全局原型（FedProto 的基础项）。

    ``valid`` 标记哪些类别的全局原型已经有值；第一轮原型还是 0 向量时会全部跳过。
    """
    if valid is None:
        valid = protos.abs().sum(-1) > 0
    mask = valid.to(feat.device)[target]
    if not bool(mask.any()):
        return feat.new_zeros(())
    f = F.normalize(feat, dim=-1)
    p = protos.to(f.dtype).to(f.device)[target]
    return (f - p).pow(2).sum(-1)[mask].mean()


def ordinal_prototype_loss(
    feat: torch.Tensor,
    target: torch.Tensor,
    protos: torch.Tensor,
    margin: float = 0.5,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """**Ordinal margin**：本文与普通 FedProto 的真正区别（方案 4.4）。

    要求样本到任意其它等级原型的距离，比到自己等级原型的距离，
    至少多出正比于等级差的间隔：

    .. math::
        \\sum_{c \\ne y_i} \\max\\big(0,\\; m|c-y_i| - (d(f_i,p_c) - d(f_i,p_{y_i}))\\big)

    结果是原型空间自然排成 0→1→2→3→4 的一维有序流形（论文图 F4 的 t-SNE）。
    """
    if valid is None:
        valid = protos.abs().sum(-1) > 0
    valid = valid.to(feat.device)
    if int(valid.sum()) < 2:  # 至少要有两个类的原型才谈得上"间隔"
        return feat.new_zeros(())

    f = F.normalize(feat, dim=-1)
    p = F.normalize(protos.to(f.dtype).to(f.device), dim=-1)
    d = torch.cdist(f, p).pow(2)                               # (B, C)

    grades = torch.arange(p.shape[0], device=f.device)
    d_pos = d.gather(1, target[:, None])                       # (B, 1)
    need = margin * (grades[None, :] - target[:, None]).abs().to(d.dtype)
    violation = (need - (d - d_pos)).clamp(min=0)              # (B, C)

    mask = (grades[None, :] != target[:, None]) & valid[None, :] & valid.to(d.device)[target][:, None]
    if not bool(mask.any()):
        return feat.new_zeros(())
    return (violation * mask).sum() / mask.sum()


# --------------------------------------------------------------------------- #
# 4. 风格不变性与一致性
# --------------------------------------------------------------------------- #
def style_invariance_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """同一张图的两个「风格视图」经 FSR 之后应当一致 —— FSR 门控的直接监督信号。"""
    a = F.normalize(feat_a, dim=-1)
    b = F.normalize(feat_b, dim=-1)
    return (a - b).pow(2).sum(-1).mean()


def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """两视图 logits 的对称 KL，稳定本地训练（对 IDRiD 这种 372 张的小 client 尤其重要）。"""
    pa = F.log_softmax(logits_a / tau, dim=-1)
    pb = F.log_softmax(logits_b / tau, dim=-1)
    kl_ab = F.kl_div(pa, pb, log_target=True, reduction="batchmean")
    kl_ba = F.kl_div(pb, pa, log_target=True, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba) * tau * tau


def edl_mse_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int = 5, anneal: float = 1.0
) -> torch.Tensor:
    """Evidential Deep Learning 的 MSE 型损失，供 **B9 FedUAA-style 基线**使用。

    证据 ``e = softplus(logits)``，Dirichlet 参数 ``alpha = e + 1``，
    不确定性 ``u = K / S``（``S = sum(alpha)``）。FedUAA 正是用这个 u 去调聚合权重。

    Args:
        anneal: KL 正则的退火系数，随轮次从 0 线性升到 1（原文做法）。
    """
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    y = F.one_hot(target, num_classes).to(p.dtype)

    err = (y - p).pow(2).sum(dim=1)
    var = (p * (1 - p) / (s + 1)).sum(dim=1)

    # KL(Dir(alpha_tilde) || Dir(1))，把非真值类的证据压回 1
    alpha_t = y + (1 - y) * alpha
    k = num_classes
    kl = (
        torch.lgamma(alpha_t.sum(1))
        - torch.lgamma(torch.tensor(float(k), device=logits.device))
        - torch.lgamma(alpha_t).sum(1)
        + ((alpha_t - 1) * (torch.digamma(alpha_t) - torch.digamma(alpha_t.sum(1, keepdim=True)))).sum(1)
    )
    return (err + var + anneal * kl).mean()


def edl_uncertainty(logits: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """``u = K / S``，越大越不确定。FedUAA-style 用 ``1 - u`` 当聚合权重。"""
    alpha = F.softplus(logits) + 1.0
    return num_classes / alpha.sum(dim=1)


def fedprox_term(model: nn.Module, global_params: Dict[str, torch.Tensor], mu: float) -> torch.Tensor:
    """FedProx 的近端项（基线 B4）：``mu/2 * ||w - w_global||^2``。"""
    total = None
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in global_params:
            continue
        g = global_params[name].to(p.device, p.dtype)
        term = (p - g).pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return 0.5 * mu * total


# --------------------------------------------------------------------------- #
# 5. 组合
# --------------------------------------------------------------------------- #
@dataclass
class LossWeights:
    """方案第 6 节超参表里的 λ。消融时把对应项置 0 即可。"""

    ord: float = 0.5              # A5 置 0
    proto_grade: float = 0.1      # A3 置 0
    proto_style: float = 0.1      # A2 置 0
    style: float = 0.05
    cons: float = 0.1
    ordinal_margin: float = 0.5   # A4 置 0 → 退化成普通原型拉近


class FedOSPLoss(nn.Module):
    """把六项组装起来，并把各项数值原样返回，方便记 TensorBoard / 排查是哪一项炸了。

    Args:
        class_counts: **本地** 类别计数。
        weights: 各项权重。
        num_classes: 5。
    """

    def __init__(
        self,
        class_counts: Sequence[float],
        weights: Optional[LossWeights] = None,
        num_classes: int = 5,
        beta: float = 0.9999,
    ) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.num_classes = num_classes
        self.cbce = ClassBalancedCE(class_counts, beta=beta)

    def forward(
        self,
        out,                                   # ForwardOutput
        target: torch.Tensor,
        deep_protos: Optional[torch.Tensor] = None,
        shallow_protos: Optional[torch.Tensor] = None,
        out_aug=None,                          # ForwardOutput | None
    ):
        parts: Dict[str, torch.Tensor] = {}
        zero = out.logits.new_zeros(())

        parts["cbce"] = self.cbce(out.logits, target)
        parts["ord"] = (
            squared_emd_loss(out.logits, target, self.num_classes) if self.w.ord > 0 else zero
        )

        if deep_protos is not None and self.w.proto_grade > 0:
            align = prototype_align_loss(out.deep_feat, target, deep_protos)
            margin = (
                ordinal_prototype_loss(
                    out.deep_feat, target, deep_protos, margin=self.w.ordinal_margin
                )
                if self.w.ordinal_margin > 0
                else zero
            )
            parts["proto_grade"] = align + margin
            parts["proto_margin_only"] = margin.detach()
        else:
            parts["proto_grade"] = zero

        if shallow_protos is not None and self.w.proto_style > 0:
            parts["proto_style"] = prototype_align_loss(out.shallow_feat, target, shallow_protos)
        else:
            parts["proto_style"] = zero

        if out_aug is not None:
            parts["style"] = (
                style_invariance_loss(out.shallow_feat, out_aug.shallow_feat)
                if self.w.style > 0 else zero
            )
            parts["cons"] = (
                consistency_loss(out.logits, out_aug.logits) if self.w.cons > 0 else zero
            )
        else:
            parts["style"] = zero
            parts["cons"] = zero

        total = (
            parts["cbce"]
            + self.w.ord * parts["ord"]
            + self.w.proto_grade * parts["proto_grade"]
            + self.w.proto_style * parts["proto_style"]
            + self.w.style * parts["style"]
            + self.w.cons * parts["cons"]
        )
        if not torch.isfinite(total):
            LOGGER.error(
                "loss 出现 NaN/Inf！各项：%s",
                {k: float(v) for k, v in parts.items()},
            )
        parts["total"] = total
        return total, {k: float(v.detach()) for k, v in parts.items()}


__all__ = [
    "ClassBalancedCE",
    "FedOSPLoss",
    "LossWeights",
    "consistency_loss",
    "edl_mse_loss",
    "edl_uncertainty",
    "effective_number_weights",
    "fedprox_term",
    "ordinal_prototype_loss",
    "prototype_align_loss",
    "squared_emd_loss",
    "style_invariance_loss",
]
