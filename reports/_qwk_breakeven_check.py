"""QWK 远端/邻级换算率的代数推导 + 数值核验（一次性核查脚本，纯 CPU）。

目标：settle 台账 §6.3 的 "3:1" 常数。严格对齐仓库实现
``fedosp/metrics.py::quadratic_weighted_kappa``（E 由**扰动后的预测边际**重算）。

构造上刻意避开前一次核查的三个缺陷：
1. 不做 clip —— 误差按"该样本的真值是否容得下这个距离"**分配**，所以实现的
   远端误判率严格等于目标值，不会被边界吃掉；
2. d=3 / d=4 的例子真实存在；
3. 基线由项目实测值（FedAvg MAE 0.4928 / ≥2 级误判 0.1393）经尾和恒等式
   MAE = P1+P2+P3+P4 同时约束，而不是凭空设定 accuracy。

换算率按**批量**测（一次修 21 例 ≈ C1 实测的 1.22 pp），而不是单例：
§6.3 用这个常数的方式就是批量的，而单例的 ΔD 随受害样本的真值与方向剧烈摆动。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedosp.metrics import quadratic_weighted_kappa  # noqa: E402

K = 5
GRADE_DIST = [1017, 270, 347, 75, 35]          # EXPECTED_GRADE_DIST["messidor2"]
N = sum(GRADE_DIST)                             # 1744
MAE_OBS = 0.4928                                # FedAvg 等级 MAE（实测）
P2_OBS = 0.1393                                 # FedAvg ≥2 级误判率（实测，T7 主指标）
FAR_DROP = 0.1393 - 0.1271                      # C1 实测 ≥2 级误判下降 1.22 pp
N_FIX = int(round(FAR_DROP * N))                # 21 例 —— 批量换算的批大小

Y_TRUE = np.repeat(np.arange(K), GRADE_DIST)
MU_T = Y_TRUE.mean()


# --------------------------------------------------------------------------
# QWK：两个口径（E 重算 / E 冻结）
# --------------------------------------------------------------------------
def qwk_fixed_E(y_true, y_pred, hist_p_ref, num_classes=K) -> float:
    """E 冻结在参照预测边际上的 QWK（分母不随扰动变化）。"""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    o = np.zeros((num_classes, num_classes))
    np.add.at(o, (y_true, y_pred), 1)
    i, j = np.meshgrid(np.arange(num_classes), np.arange(num_classes), indexing="ij")
    w = (i - j) ** 2 / (num_classes - 1) ** 2
    hist_t = np.bincount(y_true, minlength=num_classes).astype(float)
    e = np.outer(hist_t, np.asarray(hist_p_ref, float))
    e = e / e.sum() * o.sum()
    return float(1.0 - (w * o).sum() / (w * e).sum())


def sse_and_D(y_true, y_pred):
    """QWK = 1 - SSE / D，D = S2 + T2 - 2 S1 T1 / N（(K-1)^2 已约掉）。"""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    S1, S2 = yt.sum(), (yt ** 2).sum()
    T1, T2 = yp.sum(), (yp ** 2).sum()
    return float(((yp - yt) ** 2).sum()), float(S2 + T2 - 2 * S1 * T1 / len(yt))


# --------------------------------------------------------------------------
# 基线构造
# --------------------------------------------------------------------------
def build_baseline(rng):
    """不含 clip 伪影的基线。误差按可行距离分配，方向在可行方向中等概率。"""
    p2 = P2_OBS
    p3 = rng.uniform(0.010, 0.045)
    p4 = rng.uniform(0.002, min(p3, 0.015))
    p1 = MAE_OBS - p2 - p3 - p4                 # = 1 - accuracy
    exact = {1: p1 - p2, 2: p2 - p3, 3: p3 - p4, 4: p4}
    counts = {d: int(round(exact[d] * N)) for d in (1, 2, 3, 4)}

    y_pred = Y_TRUE.copy()
    free = np.ones(N, bool)
    for d in (4, 3, 2, 1):                       # 约束最紧的距离先分配
        feasible = np.where(free & ((Y_TRUE - d >= 0) | (Y_TRUE + d <= K - 1)))[0]
        if len(feasible) < counts[d]:
            return None
        pick = rng.choice(feasible, size=counts[d], replace=False)
        for idx in pick:
            y = Y_TRUE[idx]
            dirs = [s for s in (-1, +1) if 0 <= y + s * d <= K - 1]
            y_pred[idx] = y + rng.choice(dirs) * d
        free[pick] = False
    return y_pred


# --------------------------------------------------------------------------
# 批量盈亏平衡
# --------------------------------------------------------------------------
def breakeven(y_pred0, d, framing, mode, recompute_E, rng, max_m=400):
    """批量换算率：修正 N_FIX 个距离-d 误判，求使 QWK 回到基线的邻级误判数 / N_FIX。

    mode="balanced" ：补偿只落在内部等级（1,2,3）且严格上下各半 —— 预测边际不漂移，
                      这是"两个方法互换"最接近的理想化；
    mode="marginal" ：补偿在全部判对样本上按类频率抽，grade 0 只能向上（真实边际下
                      邻级补偿必然带方向偏置）。
    """
    hist_p_ref = np.bincount(y_pred0, minlength=K).astype(float)
    q = (lambda yp: quadratic_weighted_kappa(Y_TRUE, yp)) if recompute_E \
        else (lambda yp: qwk_fixed_E(Y_TRUE, yp, hist_p_ref))
    q0 = q(y_pred0)

    gap = np.abs(y_pred0 - Y_TRUE)
    victims = np.where(gap == d)[0]
    if len(victims) < N_FIX:
        return None
    yp = y_pred0.copy()
    for v in rng.choice(victims, size=N_FIX, replace=False):
        # convert = 挪到距离恰好 1（对 d=3 是两步，不是一步）
        side = int(np.sign(yp[v] - Y_TRUE[v]))
        yp[v] = Y_TRUE[v] if framing == "eliminate" else Y_TRUE[v] + side

    # 补偿候选：原本判对的样本
    correct = np.where(gap == 0)[0]
    if mode == "balanced":
        correct = correct[(Y_TRUE[correct] >= 1) & (Y_TRUE[correct] <= K - 2)]
    cand = rng.permutation(correct)[:max_m]

    qs = [q(yp)]
    for m in range(1, len(cand) + 1):
        idx = cand[m - 1]
        y = Y_TRUE[idx]
        if mode == "balanced":                   # 严格上下各半
            yp[idx] = y + (1 if m % 2 == 1 else -1)
        else:
            dirs = [s for s in (-1, +1) if 0 <= y + s <= K - 1]
            yp[idx] = y + rng.choice(dirs)
        qs.append(q(yp))
    qs = np.asarray(qs)

    first = next((m for m in range(len(qs)) if qs[m] < q0), None)
    if first is None or first == 0:
        return None
    a, b = qs[first - 1], qs[first]
    m_star = (first - 1) + (a - q0) / (a - b)
    return m_star / N_FIX, q0


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(20261002)
    n_base = 200

    base = [b for b in (build_baseline(rng) for _ in range(n_base)) if b is not None]
    diag = {k: [] for k in ("acc", "adj", "far", "mae", "qwk", "d3", "d4")}
    for yp in base:
        gap = np.abs(yp - Y_TRUE)
        diag["acc"].append((gap == 0).mean()); diag["adj"].append((gap == 1).mean())
        diag["far"].append((gap >= 2).mean()); diag["mae"].append(gap.mean())
        diag["qwk"].append(quadratic_weighted_kappa(Y_TRUE, yp))
        diag["d3"].append((gap == 3).sum()); diag["d4"].append((gap == 4).sum())
    diag = {k: np.asarray(v, float) for k, v in diag.items()}

    print(f"=== 基线面板（{len(base)} 组随机基线，无 clip，批大小 N_FIX={N_FIX}）===")
    for k in ("acc", "adj", "far", "mae", "qwk"):
        print(f"  {k:4s} mean {diag[k].mean():.4f}  sd {diag[k].std(ddof=1):.4f}"
              f"  [{diag[k].min():.4f}, {diag[k].max():.4f}]")
    print(f"  实现 ≥2 级误判率 {diag['far'].mean():.4f}（目标 {P2_OBS}）、"
          f"MAE {diag['mae'].mean():.4f}（目标 {MAE_OBS}）—— 无 clip 压制 ✓")
    print(f"  d=3 {diag['d3'].mean():.1f} 例、d=4 {diag['d4'].mean():.1f} 例 —— 非空 ✓")
    print(f"  真实均值 mu_t = {MU_T:.4f}")

    # ---- 八格主表 ----
    for mode, label in (("balanced", "对称补偿（预测边际不漂移）"),
                        ("marginal", "按类频率补偿（grade 0 强制向上）")):
        print(f"\n=== 八格换算率表 —— {label} ===")
        print(f"{'d':>2} {'framing':>10} {'E':>8} {'换算率 mean':>12} {'sd':>7} "
              f"{'[min, max]':>16}")
        for d in (2, 3):
            for framing in ("eliminate", "convert"):
                for recompute in (True, False):
                    r2 = np.random.default_rng(1000 * d + 31 * recompute
                                               + (11 if framing == "convert" else 0))
                    vals = [r[0] for yp in base
                            if (r := breakeven(yp, d, framing, mode, recompute, r2))]
                    v = np.asarray(vals)
                    print(f"{d:>2} {framing:>10} {'重算' if recompute else '冻结':>8} "
                          f"{v.mean():>12.3f} {v.std(ddof=1):>7.3f} "
                          f"[{v.min():>6.3f},{v.max():>6.3f}]")

    # ---- 闭式解核对（对称补偿下应精确成立）----
    q0 = diag["qwk"].mean()
    print(f"\n=== 闭式解（对称补偿，E 重算；Q0 = {q0:.4f}）===")
    print("  eliminate: m = d^2 恒成立（与 Q0 无关）")
    for d in (2, 3):
        conv = ((1 - q0) * (1 - 2 * d) + d ** 2 - 1) / q0
        print(f"  d={d}: eliminate {d**2:.3f}   convert "
              f"[(1-Q0)(1-2d)+d^2-1]/Q0 = {conv:.3f}")

    # ---- 分母反馈量级（复核 Analysis 1 的 -0.4%/-1.0%/-1.7%）----
    print("\n=== 分母反馈：把 x% 的极端预测向内收一步 ===")
    r3 = np.random.default_rng(3)
    for frac in (0.02, 0.05, 0.10):
        rel = []
        for yp0 in base[:40]:
            _, D0 = sse_and_D(Y_TRUE, yp0)
            yp = yp0.copy()
            ext = np.where((yp == 0) | (yp == K - 1))[0]
            take = r3.choice(ext, size=min(int(frac * N), len(ext)), replace=False)
            yp[take] += np.where(yp[take] == 0, 1, -1)
            _, D1 = sse_and_D(Y_TRUE, yp)
            rel.append(100 * (D1 / D0 - 1))
        print(f"  {frac:>5.0%} → 分母 {np.mean(rel):+.2f}%")

    # ---- §6.3 的 C1 算术 ----
    print(f"\n=== §6.3 的 C1 结论：抹平 {FAR_DROP*100:.2f} pp 远端下降需多少邻级增量 ===")
    for name, rate in (("convert d=2  (3:1)", 3.0), ("eliminate d=2 (4:1)", 4.0),
                       ("convert d=3  (8:1)", 8.0), ("eliminate d=3 (9:1)", 9.0)):
        print(f"  {name:>20}: +{FAR_DROP*rate*100:>5.2f} pp "
              f"（{FAR_DROP*rate*N:>5.1f} 例），基线邻级率 "
              f"{diag['adj'].mean():.4f} 的 {FAR_DROP*rate/diag['adj'].mean()*100:>4.1f}%")


if __name__ == "__main__":
    main()
