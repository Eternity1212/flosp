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


def referable_auroc_se(y_true: Sequence[int], auc: Optional[float] = None) -> float:
    r"""referable AUROC 的 Hanley–McNeil 标准误。

    .. math::
        \mathrm{SE} = \sqrt{\frac{A(1-A) + (n_1-1)(Q_1-A^2) + (n_0-1)(Q_2-A^2)}{n_1 n_0}}

    其中 :math:`Q_1=A/(2-A)`、:math:`Q_2=2A^2/(1+A)`。

    **为什么必须有这个量。** 锚点校验原来用一个**固定**容差（方案写 ±0.02，
    代码默认 0.05）去比所有数据集，而各 client 的测试集规模差 68 倍
    （IDRiD 103 张 vs EyePACS 7000 张）。后果是同一个容差在两端完全失效：

    ==========  ========  ==========  ====================================
    数据集        n_test    SE          ±0.02 的含义
    ==========  ========  ==========  ====================================
    IDRiD          103     0.040       仅 0.48 个 SE → **实现完全正确也约
                                       63% 概率判 FAIL**
    EyePACS       7000     0.007       2.9 个 SE → 合理
    ==========  ========  ==========  ====================================

    所以容差必须随测试集规模缩放，判据应当是"偏离几个 SE"而不是"偏离几个点"。
    """
    y = (np.asarray(y_true, int) >= REFERABLE_THRESHOLD).astype(int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 < 1 or n0 < 1:
        return float("nan")          # 单类测试集上 AUROC 本身无定义
    A = float(auc) if auc is not None else 0.5
    A = min(max(A, 1e-6), 1 - 1e-6)
    q1 = A / (2 - A)
    q2 = 2 * A * A / (1 + A)
    var = (A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0)
    return float(np.sqrt(max(var, 0.0)))


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
    auroc = referable_auroc(y_true, probs)
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "referable_auroc": auroc,
        # AUROC 的标准误。锚点校验用它把固定容差换成随规模缩放的容差
        # （IDRiD n=103 时 SE≈0.040，EyePACS n=7000 时 SE≈0.007，差 5.8 倍）
        "referable_auroc_se": referable_auroc_se(y_true, auroc),
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


# --------------------------------------------------------------------------- #
# 序数几何检验（T6 / T7）—— 创新点 C1 的直接证据
# --------------------------------------------------------------------------- #
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关，纯 numpy（含并列秩的平均处理）。"""
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # 并列值取平均秩
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        # 一侧完全无变异（例如正交单形的所有成对距离相等）→ 秩相关在数学上未定义。
        # 这里返回 0 而不是 nan：这个退化情形正是 T6 要检出的「无序数信息」，
        # 它必须得 0 分并参与后续平均，返回 nan 会让整份汇总表变成 nan。
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def prototype_geometry_metrics(
    proto: np.ndarray,
    num_classes: int = NUM_CLASSES,
    valid_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, float]:
    """**T6**：全局原型的几何是否真的承载了 0→4 的序数结构。

    命题 1 要求的不只是「各类原型互相分开」，而是**原型间距离随等级差单调递增**。
    一个把 5 个类均匀撒在球面上的编码器（标准 CE 的结果）会让所有成对距离都差不多，
    T6 上就表现为 ``rho ~ 0``；而序数几何应该给出 ``rho -> 1``。

    Args:
        proto: ``(C, D)`` 全局原型。
        valid_mask: ``(C,)`` 哪些类真的有原型。**建议显式传入**（服务器端的 ``seen``
            掩码就是它）。不传时退化为「零向量 = 未见过」的启发式 —— 在真实流程里
            原型经过 L2 归一化所以这个启发式是安全的，但对未归一化的原型
            （例如合成测试里落在原点的类）会误剔。

    Returns:
        * ``spearman_rho``  —— 成对距离 vs ``|c - y|`` 的 Spearman 相关（**T6 主指标**）
        * ``linear_r2``     —— 距离对 ``|c - y|`` 做线性回归的 R^2（几何是否近似等距排列）
        * ``adjacent_violations`` —— 单调性违反率，0 为理想，**0.5 = 全是并列（无信息）**
        * ``dist_ratio_far_near`` —— 最远等级差距离 / 相邻等级距离，均匀几何下 ~1
        * ``n_valid_classes``

    退化几何（所有成对距离相等，即标准 CE 倾向学到的正交单形）会得到
    ``rho=0, R2=0, violations=0.5, ratio=1`` —— 这是 T6 的零假设基准。
    """
    proto = np.asarray(proto, dtype=np.float64)
    if valid_mask is not None:
        valid = np.where(np.asarray(valid_mask, dtype=bool))[0]
    else:
        valid = np.where(np.abs(proto).sum(axis=1) > 0)[0]
    out = {
        "spearman_rho": float("nan"),
        "linear_r2": float("nan"),
        "adjacent_violations": float("nan"),
        "dist_ratio_far_near": float("nan"),
        "n_valid_classes": float(len(valid)),
    }
    if len(valid) < 3:
        return out

    p = proto[valid]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    gap = np.abs(valid[:, None] - valid[None, :]).astype(np.float64)

    iu = np.triu_indices(len(valid), k=1)
    dv, gv = d[iu], gap[iu]

    # 退化判定必须用**相对**容差：正交单形的距离 std 是 2e-16 量级的浮点噪声，
    # 若只判 `ss_tot > 0` 会让 R^2 在噪声上做回归，算出 -4.5 这种无意义的数。
    scale = max(float(np.abs(dv).mean()), 1e-30)
    degenerate = float(dv.std()) / scale < 1e-9

    if degenerate:
        out.update(spearman_rho=0.0, linear_r2=0.0,
                   adjacent_violations=0.5, dist_ratio_far_near=1.0)
        return out

    out["spearman_rho"] = _spearman(gv, dv)

    # 线性回归 R^2
    a = np.vstack([gv, np.ones_like(gv)]).T
    coef, *_ = np.linalg.lstsq(a, dv, rcond=None)
    resid = dv - a @ coef
    ss_tot = ((dv - dv.mean()) ** 2).sum()
    out["linear_r2"] = float(1.0 - (resid**2).sum() / ss_tot)

    # 单调性违反：gap 更大的对，距离却不更大。并列记半个违反（标准 tie 处理），
    # 这样「全部距离相等」得 0.5 而不是 1.0 —— 无信息，而非完全反序。
    viol = tot = 0.0
    for i in range(len(dv)):
        for j in range(len(dv)):
            if gv[i] > gv[j]:
                tot += 1.0
                if dv[i] < dv[j]:
                    viol += 1.0
                elif dv[i] == dv[j]:
                    viol += 0.5
    out["adjacent_violations"] = float(viol / tot) if tot else float("nan")

    near = dv[gv == 1.0]
    far = dv[gv == gv.max()]
    if near.size and far.size and near.mean() > 0:
        out["dist_ratio_far_near"] = float(far.mean() / near.mean())
    return out


def far_error_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """**T7**：误判的**质量**——错了的时候，是错到隔壁还是错到天边。

    QWK 已经惩罚远端误判，但它是一个被先验分布归一化过的聚合数，看不出误判结构。
    T7 直接报「错超过 2 个等级」的比例。临床上 grade 0 被判成 grade 4（或反之）
    才是真正危险的错误，这个指标是 C1 序数几何最该改善的东西。

    Returns:
        * ``far_error_rate``   —— :math:`P(|\\hat c - y| \\ge 2)`（**T7 主指标**）
        * ``severe_error_rate`` —— :math:`P(|\\hat c - y| \\ge 3)`
        * ``far_error_share``  —— 远端误判占**全部误判**的比例（错误结构，与准确率解耦）
        * ``mean_error_gap``   —— 误判样本的平均等级差
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.size == 0:
        return {k: float("nan") for k in
                ("far_error_rate", "severe_error_rate", "far_error_share", "mean_error_gap")}
    gap = np.abs(yt - yp)
    wrong = gap >= 1
    return {
        "far_error_rate": float((gap >= 2).mean()),
        "severe_error_rate": float((gap >= 3).mean()),
        "far_error_share": float((gap >= 2).sum() / wrong.sum()) if wrong.any() else 0.0,
        "mean_error_gap": float(gap[wrong].mean()) if wrong.any() else 0.0,
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


#: 锚点判 OFF 的阈值，单位是**标准误**而非绝对点数。2.5 SE 双侧约 p=0.012。
ANCHOR_Z = 2.5

#: 容差下限。即使测试集极大（SE→0），文献数字本身也有协议差异
#: （预处理、增广、epoch 数、是否集成），不该要求完全吻合。
ANCHOR_FLOOR = 0.02


def check_against_anchors(
    per_client: Dict[str, Dict[str, float]],
    anchor: str = "retfound_finetune_auroc",
    key: str = "referable_auroc",
    tol: Optional[float] = None,
    z: float = ANCHOR_Z,
    floor: float = ANCHOR_FLOOR,
) -> Dict[str, str]:
    r"""把实测值跟文献锚点比一遍，偏离过大说明实现有 bug（方案 §4 自查）。

    **容差随测试集规模缩放**，而不是所有数据集用同一个固定值：

    .. math:: \mathrm{tol} = \max(\text{floor},\; z \cdot \mathrm{SE})

    这个改动是被一次真实的误判逼出来的。原来是固定容差（方案写 ±0.02，
    代码默认 0.05），而各 client 测试集规模差 68 倍：

    ==========  ========  =======  =========  ==========  ===============
    数据集        n_test    参考      实测       偏差         偏差 / SE
    ==========  ========  =======  =========  ==========  ===============
    IDRiD          103     0.822    0.7676     −0.0544     **1.35**（正常）
    APTOS          732     0.943    0.9753     +0.0323     **3.32**（异常）
    EyePACS       7000     0.838    0.9006     +0.0627     **9.00**（异常）
    ==========  ========  =======  =========  ==========  ===============

    固定容差 0.05 会把这三个判成"IDRiD 挂、另两个过" —— **恰好判反了**。
    IDRiD 只偏 1.35 SE（n=103 时 SE≈0.040，95% CI 宽达 ±0.081），
    统计上无法判定为异常；而 APTOS/EyePACS 分别偏 3.3 和 9.0 个 SE，
    才是真正需要查的（用 0.23% 参数的 LoRA 超过全量微调的 ViT-L 不合理，
    通常意味着指标定义或测试集构成与文献不一致）。

    Args:
        tol: 传了就退回**固定容差**模式（仅为向后兼容，不建议用于正式校验）。
        z: 容差取几个标准误。默认 2.5（双侧约 p=0.012）。
        floor: 容差下限，见 :data:`ANCHOR_FLOOR`。

    Returns:
        ``{client: "OK|OFF|MISSING ..."}``，字符串里带 SE 与偏差的 SE 倍数，
        便于事后判断到底是"噪声"还是"实现问题"。
    """
    ref = LITERATURE_ANCHORS.get(anchor, {})
    report: Dict[str, str] = {}
    for client, expected in ref.items():
        m = per_client.get(client, {})
        got = m.get(key)
        if got is None or np.isnan(got):
            report[client] = "MISSING"
            continue
        delta = got - expected
        se = m.get(f"{key}_se")
        if tol is not None:                      # 固定容差（向后兼容）
            limit, detail = tol, f"tol={tol:.3f}(fixed)"
        elif se is None or np.isnan(se) or se <= 0:
            limit, detail = max(floor, 0.05), "tol=no-SE-fallback"
        else:
            limit = max(floor, z * se)
            detail = f"se={se:.4f} |delta|={abs(delta) / se:.2f}se tol={limit:.3f}(={z}se)"
        status = "OK" if abs(delta) <= limit else "OFF"
        report[client] = (
            f"{status} got={got:.4f} ref={expected:.4f} delta={delta:+.4f} n={m.get('n')} {detail}"
        )
        LOGGER.info("[anchor:%s] %-10s %s", anchor, client, report[client])
    return report
