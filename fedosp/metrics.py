"""评估指标（方案第 9 节）。

主指标是 **QWK**，并且必须同时报三个口径：

* 每个 client 单独的 QWK
* ``macro-over-client``：对 client 取算术平均（**不是按样本加权**）
* ``worst-client``：最差 client 的 QWK —— 这是本文的核心卖点指标

为什么不看按样本加权的平均：EyePACS 占训练数据 68%，加权平均基本等于只看 EyePACS，
四种方法看起来会差不多，正是方案第 12 节 R2 描述的失败模式。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger("metrics")

NUM_CLASSES = 5
REFERABLE_THRESHOLD = 2  # grade >= 2 记为 referable DR


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = NUM_CLASSES) -> float:
    """QWK，纯 numpy 实现，避免对 sklearn 版本的依赖差异。"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return float("nan")

    o = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(o, (y_true, y_pred), 1)

    i, j = np.meshgrid(np.arange(num_classes), np.arange(num_classes), indexing="ij")
    w = (i - j) ** 2 / (num_classes - 1) ** 2

    hist_t = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    hist_p = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    e = np.outer(hist_t, hist_p)
    e = e / e.sum() * o.sum()

    denom = (w * e).sum()
    if denom == 0:
        return 0.0
    return float(1.0 - (w * o).sum() / denom)


def grade_mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)).mean())


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = NUM_CLASSES) -> float:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    f1s = []
    for c in range(num_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def per_class_recall(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = NUM_CLASSES) -> List[float]:
    """公平性指标：重点看 tail grade（3 和 4）的 recall。"""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    out = []
    for c in range(num_classes):
        n = int((y_true == c).sum())
        out.append(float(((y_pred == c) & (y_true == c)).sum() / n) if n else float("nan"))
    return out


def roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """二分类 AUROC，用秩公式算（等价于 Mann-Whitney U），可正确处理并列值。"""
    y = np.asarray(y_true, int)
    s = np.asarray(score, float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 并列值取平均秩
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def referable_auroc(y_true: Sequence[int], probs: np.ndarray) -> float:
    """referable DR（grade >= 2）的 AUROC，用于与 RETFound / FedUAA 的数字对齐。"""
    y = (np.asarray(y_true, int) >= REFERABLE_THRESHOLD).astype(int)
    score = np.asarray(probs)[:, REFERABLE_THRESHOLD:].sum(axis=1)
    return roc_auc(y, score)


def expected_calibration_error(y_true: Sequence[int], probs: np.ndarray, n_bins: int = 15) -> float:
    """ECE（15 bins），报可靠性。联邦模型常常过自信，这项要如实报。"""
    probs = np.asarray(probs, float)
    y = np.asarray(y_true, int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def evaluate_predictions(y_true: Sequence[int], probs: np.ndarray) -> Dict[str, float]:
    """单个 client / 单个数据集上的全套指标。"""
    probs = np.asarray(probs, float)
    y_pred = probs.argmax(axis=1)
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "referable_auroc": referable_auroc(y_true, probs),
        "ece": expected_calibration_error(y_true, probs),
        "mae": grade_mae(y_true, y_pred),
        "acc": float((np.asarray(y_true, int) == y_pred).mean()),
        "n": int(len(y_true)),
    }


def aggregate_over_clients(per_client: Dict[str, Dict[str, float]], key: str = "qwk") -> Dict[str, float]:
    """把各 client 的指标收敛成主表要报的三个数。

    Returns:
        ``{macro_<key>, worst_<key>, std_<key>, weighted_<key>}``。
        ``weighted_`` 只放附录，主表用 macro 和 worst。
    """
    vals, weights = [], []
    for client, m in per_client.items():
        v = m.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            LOGGER.warning("client %s 的 %s 是 NaN，已跳过", client, key)
            continue
        vals.append(float(v))
        weights.append(float(m.get("n", 1)))
    if not vals:
        return {f"macro_{key}": float("nan"), f"worst_{key}": float("nan"),
                f"std_{key}": float("nan"), f"weighted_{key}": float("nan")}
    vals_a = np.asarray(vals)
    w = np.asarray(weights)
    return {
        f"macro_{key}": float(vals_a.mean()),
        f"worst_{key}": float(vals_a.min()),
        f"std_{key}": float(vals_a.std()),
        f"weighted_{key}": float((vals_a * w).sum() / w.sum()),
    }


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """QWK 的 bootstrap 置信区间（方案 9.2 要求的统计口径）。"""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        stats[b] = quadratic_weighted_kappa(y_true[idx], y_pred[idx])
    return {
        "point": quadratic_weighted_kappa(y_true, y_pred),
        "lo": float(np.percentile(stats, 100 * alpha / 2)),
        "hi": float(np.percentile(stats, 100 * (1 - alpha / 2))),
    }


#: 文献锚点（方案 9.3）：集中式基线必须先对上这几个数再进联邦实验
LITERATURE_ANCHORS: Dict[str, Dict[str, float]] = {
    "retfound_finetune_auroc": {"aptos": 0.943, "idrid": 0.822, "messidor2": 0.884},
    "feduaa_auc": {
        "aptos": 0.9445, "ddr": 0.9044, "eyepacs": 0.8379,
        "messidor2": 0.8012, "idrid": 0.8299, "mean": 0.8636,
    },
    "centralized_messidor2_qwk": {"messidor2": 0.78},
}


def check_against_anchors(
    per_client: Dict[str, Dict[str, float]],
    anchor: str = "retfound_finetune_auroc",
    key: str = "referable_auroc",
    tol: float = 0.05,
) -> Dict[str, str]:
    """把实测值跟文献锚点比一遍，偏离过大就是实现有 bug（方案第 14 节自查）。"""
    ref = LITERATURE_ANCHORS.get(anchor, {})
    report: Dict[str, str] = {}
    for client, expected in ref.items():
        got = per_client.get(client, {}).get(key)
        if got is None or np.isnan(got):
            report[client] = "MISSING"
            continue
        delta = got - expected
        status = "OK" if abs(delta) <= tol else "OFF"
        report[client] = f"{status} got={got:.4f} ref={expected:.4f} delta={delta:+.4f}"
        LOGGER.info("[anchor:%s] %-10s %s", anchor, client, report[client])
    return report
