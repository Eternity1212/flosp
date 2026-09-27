"""统计显著性检验与实验设计（方案第 9 节）。

审稿人最常问的两句是"这 0.8 个点的提升是真的还是噪声"和"你们几个 seed 够吗"。
这个模块把两类问题的工具放在一起：

=========================  ==============================================================
函数                        用在哪
=========================  ==============================================================
:func:`delong_test`        比两个 AUROC（同一批样本上的配对比较，不用 bootstrap）
:func:`wilcoxon_clients`   跨 client 配对比 QWK —— 只有 4 个 client，必须用非参数检验
:func:`holm_bonferroni`    一次比 8 个基线要校正多重比较，否则假阳性率远高于 0.05
:func:`paired_bootstrap`   通用兜底：任何指标的配对 bootstrap 差值置信区间
:func:`pairs_needed`       **事前**：检出指定效应需要多少个配对 seed（精确非中心 t）
=========================  ==============================================================

**为什么 4 个 client 不能用 t 检验**：n=4 时正态性假设站不住，Wilcoxon 符号秩是标准做法。
但 n=4 时 Wilcoxon 的最小可能 p 值是 0.125（双侧），**永远达不到 0.05**。
这不是 bug，是样本量的硬限制 —— 所以论文里 client 级比较应该报效应量和方向一致性
（4/4 个 client 都提升），把 p 值当辅助信息，主检验放在样本级（DeLong / bootstrap）。

**为什么 ** :func:`pairs_needed` **放在这里而不是 metrics.py**：
``metrics.py`` 装的是"给定一批预测算出一个数"（QWK / ECE / Brier），
本模块装的是"给定一批数做推断"。样本量是推断的**事前**一侧——它和
:func:`holm_bonferroni` 共用同一套 α、和配对检验共用同一个 σ_d，
放在一起才能保证"设计时假定的检验"和"分析时实际做的检验"是同一个。
放进 ``metrics.py`` 则要么和 QWK 挤在一起语义不搭，要么诱使调用方
在脚本里各写一份——后者正是本项目已经犯过的错（见函数 docstring）。
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("stats")


@dataclass
class TestResult:
    """一次检验的结果。``significant`` 用校正后的 p 值判定。"""

    name: str
    statistic: float
    p_value: float
    effect: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_corrected: Optional[float] = None
    n: Optional[int] = None
    note: str = ""

    @property
    def significant(self) -> bool:
        p = self.p_corrected if self.p_corrected is not None else self.p_value
        return bool(p < 0.05)

    def to_row(self) -> Dict[str, object]:
        return {
            "comparison": self.name,
            "effect": round(self.effect, 4),
            "ci95": (
                f"[{self.ci_low:.4f}, {self.ci_high:.4f}]"
                if self.ci_low is not None else ""
            ),
            "statistic": round(self.statistic, 4),
            "p": f"{self.p_value:.2e}",
            "p_holm": f"{self.p_corrected:.2e}" if self.p_corrected is not None else "",
            "sig": "*" if self.significant else "",
            "n": self.n or "",
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# DeLong：配对 AUROC 比较
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    """处理并列值的中位秩，DeLong 协方差估计需要它。"""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.zeros(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) 的快速 DeLong 实现。

    Args:
        scores: ``(k, n)``，k 个模型在同一批 n 个样本上的预测分数。
        labels: ``(n,)`` 的 0/1 标签。

    Returns:
        ``(aucs, covariance)``，形状分别是 ``(k,)`` 与 ``(k, k)``。
    """
    pos = labels == 1
    m, n_neg = int(pos.sum()), int((~pos).sum())
    if m == 0 or n_neg == 0:
        raise ValueError("DeLong 需要正负样本都存在")

    k = scores.shape[0]
    x, y = scores[:, pos], scores[:, ~pos]

    tx = np.array([_midrank(x[i]) for i in range(k)])
    ty = np.array([_midrank(y[i]) for i in range(k)])
    tz = np.array([_midrank(np.concatenate([x[i], y[i]])) for i in range(k)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n_neg)
    v01 = (tz[:, :m] - tx) / n_neg              # 正样本方向的结构成分
    v10 = 1.0 - (tz[:, m:] - ty) / m            # 负样本方向的结构成分

    s01 = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    cov = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n_neg
    return aucs, cov


def delong_test(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    name: str = "A vs B",
) -> TestResult:
    """DeLong 配对 AUROC 检验。

    比 bootstrap 更合适的场合：两个模型在**同一批样本**上预测。
    它直接用 AUROC 的渐近正态性算协方差，不需要重采样，也没有随机性。

    Args:
        labels: 0/1 二分类标签（本文用 referable DR，即 grade>=2）。
        scores_a / scores_b: 两个模型的连续分数（referable 的概率）。

    Returns:
        ``effect`` 是 ``auc_a - auc_b``；``ci_low/ci_high`` 是差值的 95% CI。
    """
    from scipy import stats as sps

    y = np.asarray(labels, dtype=int)
    s = np.vstack([np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)])
    if s.shape[1] != len(y):
        raise ValueError(f"分数长度 {s.shape[1]} 与标签长度 {len(y)} 不一致")

    aucs, cov = _fast_delong(s, y)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])

    if var <= 0:
        return TestResult(
            name=name, statistic=0.0, p_value=1.0, effect=diff, n=len(y),
            note="两模型预测几乎完全一致（方差<=0），无法检验",
        )

    se = float(np.sqrt(var))
    z = diff / se
    p = float(2 * (1 - sps.norm.cdf(abs(z))))
    return TestResult(
        name=name, statistic=z, p_value=p, effect=diff,
        ci_low=diff - 1.96 * se, ci_high=diff + 1.96 * se, n=len(y),
        note=f"AUC {aucs[0]:.4f} vs {aucs[1]:.4f}",
    )


# --------------------------------------------------------------------------- #
# Wilcoxon：跨 client 配对比较
# --------------------------------------------------------------------------- #
def wilcoxon_clients(
    ours: Dict[str, float],
    baseline: Dict[str, float],
    name: str = "ours vs baseline",
) -> TestResult:
    """跨 client 的配对 Wilcoxon 符号秩检验。

    Args:
        ours / baseline: ``{client_name: metric}``，只用两者共有的 client。

    Returns:
        ``effect`` 是平均差值；``note`` 里带方向一致性（几个 client 里几个变好）。

    Note:
        只有 4 个 client 时双侧 p 的下界是 0.125，**永远不会 < 0.05**。
        所以论文里请以「4/4 个 client 全部提升」这个方向一致性为主证据，
        p 值只作辅助。这一点函数会在 note 里主动提醒。
    """
    from scipy import stats as sps

    keys = sorted(set(ours) & set(baseline))
    if len(keys) < 2:
        raise ValueError(f"至少要有 2 个共有 client，实际 {len(keys)} 个")

    a = np.array([ours[k] for k in keys], dtype=float)
    b = np.array([baseline[k] for k in keys], dtype=float)
    d = a - b
    n_better = int((d > 0).sum())

    if np.allclose(d, 0):
        return TestResult(
            name=name, statistic=0.0, p_value=1.0, effect=0.0, n=len(keys),
            note="所有 client 上完全相同",
        )

    try:
        stat, p = sps.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError as exc:  # 全为零差值等退化情况
        return TestResult(
            name=name, statistic=0.0, p_value=1.0, effect=float(d.mean()),
            n=len(keys), note=f"Wilcoxon 无法计算：{exc}",
        )

    note = f"{n_better}/{len(keys)} 个 client 提升"
    if len(keys) <= 5:
        note += f"；注意 n={len(keys)} 时 p 的理论下界是 {2 ** -(len(keys) - 1):.3f}，请以方向一致性为主证据"

    return TestResult(
        name=name, statistic=float(stat), p_value=float(p), effect=float(d.mean()),
        n=len(keys), note=note,
    )


# --------------------------------------------------------------------------- #
# 配对 bootstrap：通用兜底
# --------------------------------------------------------------------------- #
def paired_bootstrap(
    metric_fn,
    labels: Sequence[int],
    pred_a: Sequence,
    pred_b: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
    name: str = "A vs B",
) -> TestResult:
    """任意指标的配对 bootstrap 差值检验（QWK 用这个，因为它没有解析方差）。

    每次重采样**用同一组下标**同时评估两个模型，这样才是配对的，
    能吃掉样本难易度带来的共同方差，比独立 bootstrap 敏感得多。

    Args:
        metric_fn: ``f(labels, preds) -> float``，例如 ``quadratic_weighted_kappa``。
        n_boot: 重采样次数。2000 次对 95% CI 足够。

    Returns:
        ``p_value`` 是双侧的 bootstrap p（差值分布跨过 0 的比例）。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    n = len(y)

    observed = float(metric_fn(y, a) - metric_fn(y, b))
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:  # 退化重采样，跳过
            diffs[i] = np.nan
            continue
        diffs[i] = metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx])

    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < n_boot * 0.5:
        LOGGER.warning("[%s] 超过一半的重采样退化，结果不可靠", name)

    # 双侧 bootstrap p：差值分布中落在 0 另一侧的比例
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return TestResult(
        name=name, statistic=float(observed / (diffs.std() + 1e-12)),
        p_value=float(min(1.0, p)), effect=observed,
        ci_low=float(np.percentile(diffs, 2.5)),
        ci_high=float(np.percentile(diffs, 97.5)),
        n=n, note=f"配对 bootstrap n_boot={len(diffs)}",
    )


# --------------------------------------------------------------------------- #
# 多重比较校正
# --------------------------------------------------------------------------- #
def holm_bonferroni(results: Sequence[TestResult], alpha: float = 0.05) -> List[TestResult]:
    """Holm-Bonferroni 逐步降低法，就地写回每个结果的 ``p_corrected``。

    为什么必须做：本文一次要和 8 个基线比。即使每个比较都用 α=0.05，
    8 次独立比较里至少一次假阳性的概率是 ``1-0.95^8 = 34%``。
    Holm 比朴素 Bonferroni 的检验力更高，且同样严格控制 family-wise error rate。

    Args:
        results: 同一个 family 内的检验结果（例如"我方 vs 全部基线"）。

    Returns:
        原列表（顺序不变），每项的 ``p_corrected`` 已填好。
    """
    m = len(results)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: results[i].p_value)
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (m - rank) * results[i].p_value)
        # 保证校正后 p 值单调不减，这是 Holm 的性质
        running = max(running, adjusted)
        results[i].p_corrected = running

    n_sig = sum(r.significant for r in results)
    LOGGER.info(
        "Holm-Bonferroni 校正 %d 个比较（alpha=%.2f）：%d 个显著", m, alpha, n_sig
    )
    return list(results)


def compare_all(
    ours: Dict[str, Dict[str, float]],
    baselines: Dict[str, Dict[str, float]],
    metric: str = "qwk",
) -> List[TestResult]:
    """一次性把我方对全部基线做 Wilcoxon + Holm 校正。

    Args:
        ours: ``{client: {metric: value}}``。
        baselines: ``{baseline_name: {client: {metric: value}}}``。

    Returns:
        已完成 Holm 校正的结果列表，按校正后 p 值升序。
    """
    ours_flat = {c: v[metric] for c, v in ours.items() if metric in v}
    results = []
    for bl_name, bl in baselines.items():
        bl_flat = {c: v[metric] for c, v in bl.items() if metric in v}
        try:
            results.append(wilcoxon_clients(ours_flat, bl_flat, name=f"FedOSP vs {bl_name}"))
        except ValueError as exc:
            LOGGER.warning("跳过 %s：%s", bl_name, exc)
    holm_bonferroni(results)
    return sorted(results, key=lambda r: r.p_corrected or 1.0)


# --------------------------------------------------------------------------- #
# 样本量：事前设计。**全仓库唯一实现，不要在脚本里另写一份。**
# --------------------------------------------------------------------------- #
def pairs_needed(delta: float, sd: float, power: float = 0.80,
                 alpha: float = 0.05, n_max: int = 2000) -> int:
    """检出配对差 ``delta`` 所需的配对数（双侧 ``alpha``，功效 ``power``）。

    **必须用精确的非中心 t，不能用正态近似。** 常见的解析式
    :math:`n=(z_{1-\\alpha/2}+z_{1-\\beta})^2\\sigma_d^2/\\Delta^2`（系数 7.849）
    有两处偏差：临界值是 :math:`t_{n-1}` 而不是 :math:`z`，且 :math:`s` 是估计量
    （所以检验统计量服从**非中心 t**，不是正态）。两处都朝**低估**方向走 ——
    也就是说它会告诉你"跑这么多 seed 就够了"，而实际功效达不到 0.8。

    在 :math:`\\sigma_d=0.0272`（Messidor-2 QWK 配对差的实测 SD）上，
    两者的差恰好是**每一格都少 2 个配对**：

    ====== =========== ===========
    Δ       z 近似       精确 t
    ====== =========== ===========
    0.010     59          **61**
    0.020     15          **17**
    0.030      7           **9**
    ====== =========== ===========

    绝对量级不变，但小 n 时相对误差最大（0.030 那一格差 29%）。

    这个坑本项目踩过三次，每一次都比上一次更难看：

    1. **B0 闸门的功效表**。commit ``8625569`` 把 ``measure_power.py`` 改成了
       蒙特卡洛，并在 commit message 里写下"n=3 时临界值是 t₂=4.303 而非
       z=1.96，差 2.2 倍、平方后 4.8 倍样本量"。**教训只进了 commit message，
       没进代码**——没有任何函数或测试阻止别处再写一遍那个近似。
    2. **方差分解脚本与台账 §8.1 的样本量表**，两处都还留着 ``2.8016**2``。
    3. **只修了一半**：第 2 条修好之后，``scripts/benchmark_table.py``（生成
       论文主表"所需配对数"那一列的脚本）里的同一行近似没跟着改，于是台账写
       61、脚本打印 59。修一处、漏一处，正是本项目台账 §11 记了 15 次的失效模式。

    所以现在它在这里：**装进包里的唯一实现**，脚本一律 import，
    并由 ``tests/test_pipeline.py`` 的两条反向测试守住——一条钉死
    61/17/9 并要求精确值严格大于近似值，另一条扫描 ``scripts/`` 与
    ``fedosp/`` 的源码，断言那个魔数在可执行代码里一次都不出现。

    Args:
        delta: 要检出的真实配对差（绝对值，与 ``sd`` 同单位）。
        sd: 配对差的标准差 :math:`\\sigma_d`。注意**不是**单臂的边际 SD——
            本项目这两个数分别是 0.0272 与 0.033。样本量按 :math:`\\sigma^2` 走，
            代错方向相反但一样要命：Δ=0.010 处 61 会变成 88（多算 44%）。
        power: 目标功效，默认 0.80。
        alpha: 双侧显著性水平，默认 0.05。
        n_max: 搜索上界，同时也是"算不出来"时的返回值。

    Returns:
        所需配对数；``delta`` 非正或 ``sd`` 非正时返回 ``n_max``
        （而不是一个看起来合理的小数字——静默返回 0/1 会让调用方以为不用跑）。
    """
    from scipy import stats as sps

    if delta <= 0 or sd <= 0:
        return n_max
    for n in range(3, n_max + 1):
        t_crit = sps.t.ppf(1 - alpha / 2, n - 1)
        ncp = delta * math.sqrt(n) / sd
        with warnings.catch_warnings():
            # df=3 时 scipy 的 boost 后端会报 "divide by zero in _nct_sf"，
            # 但它自己 clip 到 [0,1]，返回值仍然正确（实测 df=3 的功效远小于
            # 0.8，循环照常往下走）。只是噪声，不是数值错误。
            warnings.simplefilter("ignore", RuntimeWarning)
            if sps.nct.sf(t_crit, n - 1, ncp) >= power:
                return n
    return n_max


__all__ = [
    "TestResult",
    "compare_all",
    "delong_test",
    "holm_bonferroni",
    "paired_bootstrap",
    "pairs_needed",
    "wilcoxon_clients",
]
