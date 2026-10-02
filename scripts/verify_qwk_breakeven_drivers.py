"""核验闭式解与代码一致，并分解 E 重算时换算率的三个驱动量。

闭式解（对 QWK = 1 - SSE/D 精确，不是一阶近似）：
    QWK 不变  <=>  dSSE = (1 - Q0) * dD
因此批量换算率
    m/n = [ (1-Q0)*dD_fix/n - dSSE_fix/n ] / [ 1 - (1-Q0)*delta_adj ]
其中 delta_adj 是每新增一次邻级误判带来的 dD。

基线构造与批量换算率都复用同目录的 ``verify_qwk_breakeven.py``。

运行::

    python scripts/verify_qwk_breakeven_drivers.py   # 纯 CPU，约 18 s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedosp.metrics import quadratic_weighted_kappa  # noqa: E402

from verify_qwk_breakeven import (  # noqa: E402
    K, N, N_FIX, Y_TRUE, MU_T, build_baseline, breakeven,
)


def dD(p_old, p_new, y):
    """单次预测改动对 D = S2 + T2 - 2 S1 T1 / N 的贡献（精确、可加）。"""
    return (p_new ** 2 - p_old ** 2) - 2 * MU_T * (p_new - p_old)


def main():
    rng = np.random.default_rng(7)
    base = [b for b in (build_baseline(rng) for _ in range(60)) if b is not None]

    print(f"mu_t = {MU_T:.4f}   N_FIX = {N_FIX}")
    print("\n=== 驱动量分解 + 闭式解 vs 数值（E 重算）===")
    print(f"{'d':>2} {'framing':>10} {'mode':>9} {'dSSE/例':>8} {'dD_fix/例':>10} "
          f"{'δ_adj':>7} {'闭式':>7} {'数值':>7} {'差':>7}")

    for d in (2, 3):
        for framing in ("eliminate", "convert"):
            for mode in ("balanced", "marginal"):
                r = np.random.default_rng(99)
                dsse_l, ddfix_l, dadj_l, num_l = [], [], [], []
                for yp0 in base:
                    q0 = quadratic_weighted_kappa(Y_TRUE, yp0)
                    gap = np.abs(yp0 - Y_TRUE)
                    vict = np.where(gap == d)[0]
                    if len(vict) < N_FIX:
                        continue
                    pick = r.choice(vict, size=N_FIX, replace=False)
                    s_sse = s_dd = 0.0
                    for v in pick:
                        p_old, y = yp0[v], Y_TRUE[v]
                        # convert = 挪到距离恰好 1（对 d=3 是两步，不是一步）
                        side = int(np.sign(p_old - y))
                        p_new = y if framing == "eliminate" else y + side
                        s_sse += (p_new - y) ** 2 - (p_old - y) ** 2
                        s_dd += dD(p_old, p_new, y)

                    # delta_adj：补偿模型下每例邻级误判的平均 dD
                    corr = np.where(gap == 0)[0]
                    if mode == "balanced":
                        corr = corr[(Y_TRUE[corr] >= 1) & (Y_TRUE[corr] <= K - 2)]
                    samp = r.permutation(corr)[:400]
                    ds = []
                    for m, idx in enumerate(samp, start=1):
                        y = Y_TRUE[idx]
                        if mode == "balanced":
                            p_new = y + (1 if m % 2 == 1 else -1)
                        else:
                            dirs = [s for s in (-1, +1) if 0 <= y + s <= K - 1]
                            p_new = y + r.choice(dirs)
                        ds.append(dD(y, p_new, y))
                    delta_adj = float(np.mean(ds))

                    dsse_l.append(s_sse / N_FIX)
                    ddfix_l.append(s_dd / N_FIX)
                    dadj_l.append(delta_adj)
                    r2 = np.random.default_rng(99)
                    res = breakeven(yp0, d, framing, mode, True, r2)
                    num_l.append(res[0] if res else np.nan)

                dsse, ddfix, dadj = np.mean(dsse_l), np.mean(ddfix_l), np.mean(dadj_l)
                q0 = float(np.mean([quadratic_weighted_kappa(Y_TRUE, b) for b in base]))
                closed = ((1 - q0) * ddfix - dsse) / (1 - (1 - q0) * dadj)
                num = np.nanmean(num_l)
                print(f"{d:>2} {framing:>10} {mode:>9} {dsse:>8.2f} {ddfix:>10.2f} "
                      f"{dadj:>7.3f} {closed:>7.3f} {num:>7.3f} {closed-num:>+7.3f}")

    # ---- 完全对称理想化（修正侧与补偿侧都不漂移预测边际）----
    # 修正侧对称时 dD_fix/例 = -d^2（eliminate）/ 1-d^2（convert），恰好等于 dSSE/例；
    # 补偿侧对称时 δ_adj = 1 = dSSE/例。两边都"ΔD 与 ΔSSE 同值"，于是
    #   m = [(1-Q)*dD_fix - dSSE] / [1-(1-Q)*δ_adj] = dSSE*(1-(1-Q))/Q = dSSE
    # 即换算率精确等于 d^2 / d^2-1，**与 Q0 完全无关**。
    print("\n=== 完全对称理想化：换算率与 Q0 无关（精确）===")
    print(f"{'Q0':>7} {'elim d=2':>9} {'elim d=3':>9} {'conv d=2':>9} {'conv d=3':>9}")
    for q0 in (0.55, 0.6283, 0.70, 0.78):
        row = [((1 - q0) * (-d ** 2) + d ** 2) / q0 for d in (2, 3)]
        row += [((1 - q0) * (1 - d ** 2) + d ** 2 - 1) / q0 for d in (2, 3)]
        print(f"{q0:>7.4f} " + " ".join(f"{v:>9.3f}" for v in row))

    # ---- 复核 Analysis 1 的 5.6 / 10.5 ----
    print("\n=== Analysis 1 的 5.6 / 10.5 落在哪个格子 ===")
    for d, claim in ((2, 5.6), (3, 10.5)):
        r2 = np.random.default_rng(99)
        vals = [res[0] for yp0 in base
                if (res := breakeven(yp0, d, "eliminate", "balanced", True, r2))]
        v = np.asarray(vals)
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"  d={d} eliminate/E 重算/对称补偿: {v.mean():.3f} ± {v.std(ddof=1):.3f} "
              f"95% [{lo:.2f}, {hi:.2f}] —— Analysis 1 声称 {claim}: "
              f"{'落在区间内' if lo <= claim <= hi else '落在区间外'}")


if __name__ == "__main__":
    main()
