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
# 2b. 其余三种序数范式（B17 交叉组 / A5 消融用）
# --------------------------------------------------------------------------- #
#: 支持的序数损失范式。``emd`` 是本文默认；后三个用于 B17「现成 FL 方法 × 序数损失」
#: 交叉组，其中 ``binomial`` 与 ``ordinal_encoding`` **正是 Corbetta MIDL 2025 用的两种**，
#: 实现它们的额外好处是直接建立与 MIDL'25 的可比性（设计文档 §3.3）。
ORDINAL_LOSSES = ("none", "emd", "binomial", "coral", "ordinal_encoding", "exp_mse")

#: 需要 K-1 维输出头的范式（见 ``FedOSPConfig.ordinal_head``）
ORDINAL_HEAD_LOSSES = ("coral", "ordinal_encoding")


def expectation_mse_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int = 5
) -> torch.Tensor:
    r"""Expectation MSE（``exp_MSE``）：预测分布**期望**与真值的平方误差。

    .. math:: L = \Big(\sum_{c} c\,p_c - y\Big)^2

    出处：Stelter, **Corbetta**, Lakbir, Beets-Tan, Cruz, Cardoso, **Silva**,
    *Preserving Ordinality in Diabetic Retinopathy Grading through a
    Distribution-Based Loss Function*, **NLDL 2026**, PMLR 307:405–414。
    代码 ``github.com/Trustworthy-AI-UU-NKI/Ordinal-DR-Grading``。

    **为什么必须实现它**：这是 Corbetta/Silva 组 2026 年 1 月发表的新损失，
    任务与本文完全相同（5 级序数 DR 分级），数据集与我们重叠三个
    （APTOS / IDRiD / DDR），并且他们在五个公开 DR 数据集上报告它
    **优于 CE 与其它序数损失**。不把它放进 B17 交叉组，
    "FedOSP 的序数处理更好"这个说法就缺了当前最强的对照。

    **一个必须注意的性质**：它只约束分布的**一阶矩**，对形状完全不敏感。
    取 :math:`y=2, K=5`：

    ==========================  ======  =========  ==========
    预测分布                      均值    exp_MSE    平方 EMD
    ==========================  ======  =========  ==========
    ``[0, 0, 1, 0, 0]``          2.00     0.000      0.000
    ``[0, .5, .5, 0, 0]``        1.50     0.250      0.050
    ``[.5, 0, 0, 0, .5]``        2.00     **0.000**  0.200
    ``[.2, .2, .2, .2, .2]``     2.00     **0.000**  0.080
    ==========================  ======  =========  ==========

    极端双峰与完全均匀分布的均值恰好都是 2，于是 ``exp_MSE`` 给它们**零惩罚** ——
    原文称它 "promotes unimodal predictions"，在这两个反例上并不成立。
    平方 EMD 约束整条累积分布，能区分形状。

    不过要公允：他们仓库里有 ``--lamda`` 超参，说明实际用法应是
    ``CE + λ·exp_MSE``，CE 会把概率质量压到真值类上，从而补掉这个退化。
    本文的损失结构恰好是 ``CB-CE + λ_ord·L_ord``，所以
    ``--ord-type exp_mse`` 就是对他们formulation 的忠实复现，而非削弱版。
    """
    p = F.softmax(logits, dim=-1)
    grades = torch.arange(num_classes, device=p.device, dtype=p.dtype)
    return ((p * grades).sum(-1) - target.to(p.dtype)).pow(2).mean()


def binomial_unimodal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 5,
    tau: float = 1.0,
) -> torch.Tensor:
    """Binomial 单峰软标签 CE（Beckham & Pal 2017；Corbetta MIDL'25 的范式之一）。

    把硬标签换成一个**以真值为中心的二项分布**软标签：

    .. math::
        q_k = \\binom{K-1}{k}\\,p^{k}(1-p)^{K-1-k},\\qquad p = \\frac{y}{K-1}

    然后做软标签交叉熵 :math:`-\\sum_k q_k\\log \\hat p_k`。

    为什么这样能编码序数性：$q$ 是**单峰**且随 $\\vert k-y\\vert$ 衰减的，所以模型被
    要求"把概率质量放在真值附近"，而不是像 one-hot 那样对所有错误一视同仁。
    $y=0$ 与 $y=K-1$ 时退化为 one-hot（$p=0$ 或 $1$），这是该范式的已知性质。

    Args:
        logits: ``(B, K)``
        target: ``(B,)`` 整数等级
        tau: 软标签温度，``q ∝ q^{1/tau}``。``<1`` 更尖锐，``>1`` 更平坦。

    与 ``squared_emd_loss`` 的区别：EMD 惩罚**累积分布**的偏差（对整体位置敏感），
    binomial 规定了**目标分布的形状**（对单峰性敏感）。两者不等价，A5 里分开消融。

    ⚠ **一个非显然但必须知道的性质**：软标签 CE 的最小值在 $\\hat p=q$ 处，**不在
    one-hot 处**。实测 $y=2$ 时最优损失 = $H(q)=1.4075$，而一个"完美自信"的
    peak@2 预测损失是 6.25 —— 比均匀预测（1.609）还差。

    也就是说**这个范式内在地压制预测置信度**：它不允许模型把全部质量压在真值上。
    后果有两个，报结果时要注意：

    1. 它天然倾向于**更低的 ECE**（置信度被拉低），所以在校准指标上占便宜。拿它跟
       普通 CE 比 ECE 是不公平的，必须同时看 QWK/MAE 才能判断是真校准好还是只是不自信。
    2. argmax 预测仍然正确（$q$ 的峰在 $y$），所以 QWK/accuracy 不受这个性质影响。
    """
    k = num_classes
    dev = logits.device
    # log C(K-1, j)，用 lgamma 避免大数溢出
    j = torch.arange(k, device=dev, dtype=torch.float32)
    log_binom = (
        torch.lgamma(torch.tensor(float(k), device=dev))
        - torch.lgamma(j + 1.0)
        - torch.lgamma(float(k) - j)
    )
    p = (target.float() / (k - 1)).clamp(0.0, 1.0).unsqueeze(-1)      # (B, 1)
    # 用 log 空间算，且把 0*log(0) 显式处理成 0
    log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
    log_1mp = torch.where(p < 1, (1.0 - p).log(), torch.zeros_like(p))
    log_q = log_binom + j * log_p + (float(k) - 1.0 - j) * log_1mp    # (B, K)
    # p=0/1 时上式在 j!=target 处应为 -inf；上面的 where 会把它算成 0，这里修掉
    degenerate = (p.squeeze(-1) <= 0) | (p.squeeze(-1) >= 1)
    if bool(degenerate.any()):
        onehot = F.one_hot(target, k).to(log_q.dtype)
        log_q = torch.where(
            degenerate.unsqueeze(-1),
            torch.where(onehot > 0, torch.zeros_like(log_q),
                        torch.full_like(log_q, float("-inf"))),
            log_q,
        )
    q = F.softmax(log_q / max(tau, 1e-6), dim=-1) if tau != 1.0 else log_q.exp()
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    return -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def ordinal_levels(target: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """把等级标签展开成 ``K-1`` 个二元「是否超过阈值 k」标签。

    ``y=3, K=5`` → ``[1, 1, 1, 0]``，即 ``y>0, y>1, y>2, y>3``。
    """
    ks = torch.arange(num_classes - 1, device=target.device).unsqueeze(0)
    return (target.unsqueeze(-1) > ks).to(torch.float32)


def ordinal_binary_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 5,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``K-1`` 个二元分类器的 BCE —— **CORAL 与 Ordinal-Encoding 共用的损失**。

    .. math:: L = -\\sum_{k=1}^{K-1}\\big[y^{(k)}\\log\\sigma(z_k)
              + (1-y^{(k)})\\log(1-\\sigma(z_k))\\big]

    两种范式的差别**只在网络结构上，不在损失上**（见 ``FedOSPConfig.ordinal_head``）：

    * ``ordinal_encoding``：``K-1`` 个**独立**的 logit。灵活，但不保证
      $\\sigma(z_1)\\ge\\sigma(z_2)\\ge\\dots$，可能出现"不是 >1 但是 >2"的自相矛盾。
    * ``coral``：所有阈值**共享同一个权重向量**、只有偏置不同（Cao et al. 2020），
      于是 $z_k$ 之间只差常数，**秩单调性由构造保证**，不会自相矛盾。

    Args:
        logits: ``(B, K-1)``
        class_weights: ``(K-1,)`` 各阈值的重要性权重。DR 分级里 referable 阈值
            （k=1，即 grade>=2）临床上更重要，可以在这里加权。
    """
    levels = ordinal_levels(target, num_classes)
    loss = F.binary_cross_entropy_with_logits(
        logits, levels, reduction="none"
    )                                                     # (B, K-1)
    if class_weights is not None:
        loss = loss * class_weights.to(loss.device, loss.dtype).unsqueeze(0)
    return loss.sum(-1).mean()


def ordinal_logits_to_probs(
    logits: torch.Tensor, num_classes: int = 5
) -> torch.Tensor:
    """把 ``(B, K-1)`` 的阈值 logit 转成 ``(B, K)`` 的类别概率。

    这个函数是**接口边界**：转换之后，下游所有指标（QWK / ECE / DeLong / T7 /
    referable AUROC）都不需要为序数头做任何特殊处理，直接复用同一套评估代码。

    .. math::
        P(y=0)=1-s_1,\\quad P(y=k)=s_k-s_{k+1},\\quad P(y=K-1)=s_{K-1}

    其中 $s_k=\\sigma(z_k)=P(y>k-1)$。``ordinal_encoding`` 下 $s$ 未必单调，
    差值可能为负，所以做 ``clamp_min(0)`` 再归一化 —— 这是该范式的已知代价，
    也正是 CORAL 用共享权重去消除它的原因。
    """
    s = torch.sigmoid(logits.float())                     # (B, K-1)
    ones = torch.ones_like(s[:, :1])
    zeros = torch.zeros_like(s[:, :1])
    upper = torch.cat([ones, s], dim=1)                   # P(y > k-1), k=0..K-1
    lower = torch.cat([s, zeros], dim=1)                  # P(y > k)
    probs = (upper - lower).clamp_min(0.0)
    return probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)


def ordinal_loss_by_type(
    ord_type: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 5,
) -> torch.Tensor:
    """按名字派发序数损失。``logits`` 的形状必须与范式匹配（K 或 K-1 列）。"""
    if ord_type == "none":
        return logits.new_zeros(())
    if ord_type == "emd":
        return squared_emd_loss(logits, target, num_classes)
    if ord_type == "binomial":
        return binomial_unimodal_ce(logits, target, num_classes)
    if ord_type == "exp_mse":
        return expectation_mse_loss(logits, target, num_classes)
    if ord_type in ORDINAL_HEAD_LOSSES:
        return ordinal_binary_ce(logits, target, num_classes)
    raise ValueError(f"未知序数损失 {ord_type!r}，可选 {ORDINAL_LOSSES}")


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


def squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""成对平方欧氏距离，``(B, D) x (C, D) -> (B, C)``。

    展开成矩阵乘法而不用 :func:`torch.cdist`，原因不是性能而是**可移植性**：

    ``aten::_cdist_backward`` 在 MPS 上**没有实现**，反向传播会直接抛
    ``NotImplementedError``，必须靠 ``PYTORCH_ENABLE_MPS_FALLBACK=1`` 回落 CPU 才能跑。
    也就是说在 Apple Silicon 上，任何用到原型损失的策略（fedosp / fedproto）
    不设这个环境变量就**完全无法运行**。用 matmul 展开后原生可跑，不依赖环境变量。

    .. math::
        \lVert a_i - b_j \rVert^2 = \lVert a_i \rVert^2 + \lVert b_j \rVert^2
                                     - 2\,a_i \cdot b_j

    浮点误差可能让结果出现极小的负值（实测与 ``cdist`` 最大差 7e-7），
    所以 ``clamp_min(0)`` —— 距离为负会让下游的 ``sqrt`` 或 margin 比较出错。
    """
    b = b.to(dtype=a.dtype, device=a.device)
    a2 = a.pow(2).sum(-1, keepdim=True)                # (B, 1)
    b2 = b.pow(2).sum(-1)                              # (C,)
    return (a2 + b2.unsqueeze(0) - 2.0 * (a @ b.t())).clamp_min(0)


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
    d = squared_distances(f, p)                                # (B, C)

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


def threshold_consistency_loss(
    logits_a: torch.Tensor, logits_b: torch.Tensor
) -> torch.Tensor:
    """阈值式范式下的两视图一致性：K-1 个**独立伯努利**的对称 BCE。

    不能用 :func:`consistency_loss`：那个函数对 logits 做 softmax，
    而阈值 logit 之间不构成一个概率分布（它们是 K-1 个独立的"是否超过 k"），
    softmax 出来的东西没有任何含义 —— 而且照样能算出一个有限的数，不会报错。
    """
    sa, sb = torch.sigmoid(logits_a), torch.sigmoid(logits_b)
    return 0.5 * (
        F.binary_cross_entropy(sa, sb.detach(), reduction="mean")
        + F.binary_cross_entropy(sb, sa.detach(), reduction="mean")
    )


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
    #: 序数范式（A5 消融 / B17 交叉组）。见 :data:`ORDINAL_LOSSES`。
    #: ``coral`` / ``ordinal_encoding`` 需要模型同时设 ``cfg.ordinal_head``。
    ord_type: str = "emd"


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
        if self.w.ord_type not in ORDINAL_LOSSES:
            raise ValueError(
                f"未知 ord_type={self.w.ord_type!r}，可选 {ORDINAL_LOSSES}"
            )
        #: 阈值式范式（coral / ordinal_encoding）下模型输出只有 K-1 列，
        #: 这时 **CB-CE 无法定义**（它需要 K 类 logit），序数 BCE 本身就是全部的
        #: 分类损失。若仍对 4 列算 CB-CE，不会报错但类别语义完全错位。
        self.threshold_paradigm = self.w.ord_type in ORDINAL_HEAD_LOSSES
        if self.threshold_paradigm:
            LOGGER.info(
                "ord_type=%s 为阈值式范式：CB-CE 关闭，分类损失全部由 K-1 个阈值 BCE 承担。"
                "类别不平衡改由阈值重要性权重承担。",
                self.w.ord_type,
            )
            # 把 CB-CE 的类别权重折算成 K-1 个阈值的权重：阈值 k 的权重取它两侧
            # 类别权重的均值，这样"稀有等级附近的阈值"仍然被加权。
            cw = self.cbce.weight.detach().float()
            self.register_buffer(
                "threshold_weights", (cw[:-1] + cw[1:]) / 2.0, persistent=False
            )

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

        if self.threshold_paradigm:
            # 阈值式：唯一的分类损失就是 K-1 个二元 BCE，权重固定为 1
            # （不乘 self.w.ord，否则整体损失尺度会随消融配置漂移）
            parts["cbce"] = zero
            parts["ord"] = ordinal_binary_ce(
                out.logits, target, self.num_classes,
                class_weights=getattr(self, "threshold_weights", None),
            )
            ord_weight = 1.0
        else:
            parts["cbce"] = self.cbce(out.logits, target)
            parts["ord"] = (
                ordinal_loss_by_type(
                    self.w.ord_type, out.logits, target, self.num_classes
                )
                if self.w.ord > 0 else zero
            )
            ord_weight = self.w.ord

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
            if self.w.cons <= 0:
                parts["cons"] = zero
            elif self.threshold_paradigm:
                parts["cons"] = threshold_consistency_loss(out.logits, out_aug.logits)
            else:
                parts["cons"] = consistency_loss(out.logits, out_aug.logits)
        else:
            parts["style"] = zero
            parts["cons"] = zero

        total = (
            parts["cbce"]
            + ord_weight * parts["ord"]
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
