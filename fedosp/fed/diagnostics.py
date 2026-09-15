"""联邦聚合的统计诊断：随机效应方差分解与有效客户端数。

这个模块是 FedOSP 贡献 C2 的理论内核。它回答一个问题：

    **服务器在聚合各 client 的类原型时，最优权重是什么？**

FedProto 用「按样本量加权」，方案 v1.0 用「client 等权」。两者都是拍脑袋的。
把原型估计写成随机效应模型之后，最优权重有闭式解，而上面两种做法恰好是它的
两个极限特例。

模型
----
client ``k`` 上类别 ``c`` 的本地原型：

.. math::
    p_k = \\mu + b_k + e_k,\\quad
    b_k \\sim \\mathcal{N}(0, \\tau^2 I),\\quad
    e_k \\sim \\mathcal{N}(0, v_k I)

* ``mu``   —— 真正想估计的跨中心一致的类语义
* ``b_k``  —— client 的**域偏置**（相机、人群、标注协议），方差 ``tau^2``，与样本量无关
* ``e_k``  —— **有限样本噪声**，方差 ``v_k = s_k^2 / n_k``，随样本量下降

服务器估计 :math:`P=\\sum_k w_k p_k`（:math:`\\sum w_k = 1`）对任意 ``w`` 都是无偏的，故

.. math::
    \\mathrm{MSE}(P) = \\sum_k w_k^2 (\\tau^2 + v_k)

在 :math:`\\sum w_k=1` 下最小化，由 Cauchy–Schwarz 得**逆方差（精度）加权**：

.. math::
    w_k^\\star \\propto \\frac{1}{\\tau^2 + v_k}

两个极限特例
------------
=========================  ==========================  =======================
条件                        ``w_k*`` 退化为              对应已有方法
=========================  ==========================  =======================
``tau^2 = 0``（无域偏移）    :math:`w_k \\propto n_k`     FedProto（按样本量加权）
``tau^2 >> v_k``（域偏移主导） :math:`w_k \\to 1/K`        client 等权（v1.0 做法）
=========================  ==========================  =======================

所以 A6 消融不再是「三个拍脑袋选项的比较」，而是沿 ``tau^2`` 这一条理论曲线的扫描。

``tau^2`` 用 DerSimonian–Laird 矩估计（随机效应元分析的标准做法，1986），
只需要 client 额外上传每类的 ``n_kc`` 与类内特征方差 —— C=5 时是 10 个标量，
相对 2.31 MB 的参数上传完全可忽略。

有效客户端数
------------
:math:`n_{\\mathrm{eff}} = 1/\\sum_k w_k^2`（客户端维度上的 Kish 有效样本量）。
由 Cauchy–Schwarz :math:`n_{\\mathrm{eff}} \\le K`，等号仅当权重均匀。

它的用处是把「大客户端淹没小客户端」从一句定性描述变成一个可算的数。对
EyePACS/DDR/APTOS/IDRiD 这个 4 院联邦，按样本量加权时 :math:`n_{\\mathrm{eff}}=1.75`；
再叠加「按 epoch 训练」这个第二乘子会跌到 **1.15** —— 这个联邦在统计上几乎
等于只训了 EyePACS 一家。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("diagnostics")

#: tau^2 估计的下限保护。全 0 会让精度加权在某些类上退化成纯 sample-weighted，
#: 这本身是正确行为，所以不设人为下限；这里只是数值保护。
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. 随机效应方差分解
# --------------------------------------------------------------------------- #
def dersimonian_laird_tau2(
    values: np.ndarray,
    sampling_vars: np.ndarray,
) -> float:
    """用 DerSimonian–Laird 矩估计法估计 between-client 方差 ``tau^2``。

    Args:
        values: ``(K, D)`` 各 client 的原型（D 维），或 ``(K,)`` 标量观测。
        sampling_vars: ``(K,)`` 各 client 的**每维**抽样方差 ``v_k = s_k^2 / n_k``。

    Returns:
        ``tau^2`` 的非负估计。K < 2 时返回 0（无法区分组间与组内变异）。

    实现说明：原始 DL 公式针对标量效应量。这里的观测是 D 维向量，做法是把
    Q 统计量按维度平均（等价于假设各维共享同一个 ``tau^2``，与模型里
    ``b_k ~ N(0, tau^2 I)`` 的各向同性假设一致）。
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if values.shape[0] == 1 and values.ndim == 2 and values.shape[1] > 1:
        # (K,) 被 atleast_2d 变成了 (1, K)，这里还原成 (K, 1)
        pass
    v = np.asarray(sampling_vars, dtype=np.float64).ravel()
    if values.shape[0] != v.shape[0]:
        values = values.T
    k, d = values.shape
    if k < 2:
        return 0.0

    # 固定效应权重（只考虑抽样噪声）
    w_fe = 1.0 / np.maximum(v, _EPS)
    p_fe = (w_fe[:, None] * values).sum(axis=0) / w_fe.sum()

    # Q 统计量：按维度平均，使其在 H0(tau^2=0) 下的期望仍是 (K-1)
    q = float((w_fe[:, None] * (values - p_fe) ** 2).sum() / d)

    # DL 的分母
    denom = w_fe.sum() - (w_fe**2).sum() / w_fe.sum()
    if denom <= _EPS:
        return 0.0
    return float(max(0.0, (q - (k - 1)) / denom))


def precision_weights(
    values: np.ndarray,
    sampling_vars: np.ndarray,
    tau2: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """随机效应模型下的最优（逆方差）聚合权重。

    Args:
        values: ``(K, D)`` 各 client 的原型。
        sampling_vars: ``(K,)`` 每维抽样方差 ``v_k``。
        tau2: 显式指定 ``tau^2``（A6 的 tau^2 扫描用）。``None`` 则用 DL 估计。

    Returns:
        ``(weights, tau2)``，``weights`` 已归一化到和为 1。

    退化行为（这是它作为理论的核心价值）：

    * ``tau2 == 0``      → ``w_k ∝ 1/v_k ∝ n_k / s_k^2``，即按样本量加权
    * ``tau2 -> inf``    → ``w_k → 1/K``，即 client 等权
    """
    v = np.asarray(sampling_vars, dtype=np.float64).ravel()
    if tau2 is None:
        tau2 = dersimonian_laird_tau2(values, v)
    w = 1.0 / np.maximum(tau2 + v, _EPS)
    return w / w.sum(), float(tau2)


# --------------------------------------------------------------------------- #
# 2. 有效客户端数
# --------------------------------------------------------------------------- #
def effective_client_count(weights: Sequence[float]) -> float:
    """``n_eff = 1 / sum(w^2)``，衡量聚合实际用上了几个 client。

    权重会先归一化，所以传入未归一化的原始权重也可以。

    >>> round(effective_client_count([0.25] * 4), 2)   # client 等权
    4.0
    >>> round(effective_client_count([24600, 6260, 2560, 372]), 2)   # 按样本量
    1.75
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return 0.0
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.maximum((w**2).sum(), _EPS))


def compound_effective_client_count(
    weights: Sequence[float], steps: Sequence[float]
) -> float:
    """把**本地步数**这个第二乘子也算进去的有效客户端数。

    client 对全局模型的实际影响 ``∝ w_k * S_k``：参数增量的幅度随本地步数增长，
    所以「按 epoch 训练」会在聚合权重之外再叠加一层不平衡。

    >>> # 按 epoch + 按样本量 = 朴素 FedAvg 的默认配方
    >>> round(compound_effective_client_count(
    ...     [24600, 6260, 2560, 372], [769, 196, 80, 12]), 2)
    1.15
    >>> # sqrt 步数 + client 等权
    >>> round(compound_effective_client_count(
    ...     [1, 1, 1, 1], [200, 101, 65, 25]), 2)
    2.78
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    s = np.asarray(steps, dtype=np.float64).ravel()
    if w.size != s.size:
        raise ValueError(f"weights 与 steps 长度不一致：{w.size} vs {s.size}")
    if w.sum() > 0:
        w = w / w.sum()
    return effective_client_count(w * s)


def aggregation_diagnostics(
    client_names: Sequence[str],
    param_weights: Sequence[float],
    local_steps: Sequence[float],
    proto_weights: Optional[Dict[int, Sequence[float]]] = None,
    tau2_per_class: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    """汇总一轮的聚合诊断，写进 ``result.json`` 的 ``diagnostics`` 字段。

    Args:
        client_names: client 名字，顺序与权重一致。
        param_weights: LoRA/head 的聚合权重。
        local_steps: 各 client 本轮的本地步数。
        proto_weights: ``类别 -> 该类的原型聚合权重``（精度加权模式下逐类不同）。
        tau2_per_class: ``类别 -> tau^2 估计``。

    Returns:
        可直接 json 序列化的 dict。
    """
    out: Dict[str, object] = {
        "clients": list(client_names),
        "param_weights": [float(w) for w in param_weights],
        "local_steps": [float(s) for s in local_steps],
        "n_eff_param": effective_client_count(param_weights),
        "n_eff_compound": compound_effective_client_count(param_weights, local_steps),
        "n_clients": len(client_names),
    }
    if proto_weights:
        out["n_eff_proto_per_class"] = {
            str(c): effective_client_count(w) for c, w in sorted(proto_weights.items())
        }
        vals = list(out["n_eff_proto_per_class"].values())  # type: ignore[union-attr]
        out["n_eff_proto_mean"] = float(np.mean(vals)) if vals else 0.0
    if tau2_per_class:
        out["tau2_per_class"] = {str(c): float(t) for c, t in sorted(tau2_per_class.items())}
        out["tau2_mean"] = float(np.mean(list(tau2_per_class.values())))
    return out


__all__ = [
    "aggregation_diagnostics",
    "compound_effective_client_count",
    "dersimonian_laird_tau2",
    "effective_client_count",
    "precision_weights",
]
