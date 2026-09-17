"""B0 前置闸门之二：从已跑完的 run 里测出统计功效参数，判断 seed 数够不够。

为什么必须在正式矩阵之前做
--------------------------
本文主指标是 **worst-client QWK**，而 worst client 几乎必然是 IDRiD（372 张训练图）。
2026-09-17 的集中式审计第一次测出它的 seed 噪声：**标准差 0.0386**。

把这个数字和设计文档里 C1 的诚实效应量（QWK **+0.02~0.04**，见 §2.5）放一起，
达到 80% 功效所需的 seed 数（配对检验，双侧 α=0.05）：

**3 个 seed 的实际功效（蒙特卡洛，Δ=0.03，σ=0.0386）：**

==========  ========  ========  ========  ========
seed 数      ρ=0.5     ρ=0.75    ρ=0.9
==========  ========  ========  ========  ========
**3**       **12.8%**  **20.1%**  **38.4%**
5            26.9%     46.2%     82.3%
7            40.9%     68.6%     96.7%
10           59.0%     87.1%     99.8%
15           80.1%     97.6%     100%
==========  ========  ========  ========  ========

矩阵里 49 行配置用的是 3 个 seed —— **即使在 ρ=0.9 的最好情况下也只有 38% 功效**。
注意不要用基于正态分位数的解析公式估这个（见 :func:`power_at` 的说明）：
n=3 时临界值来自 :math:`t_2=4.303` 而非 :math:`z=1.96`，解析式会严重低估所需 seed 数。
若实际效应量落在 +0.02 那端，或两臂只是中等相关，就会跑出
"方向对但 p=0.09" 这种最难处理的结果，只能加 seed 重跑整个矩阵。

而 ρ 是**可以事先测出来的**：跑两臂各 3 个 seed，算 worst-client 指标的配对相关。
6 个 run、不到一天，就能决定要不要加 seed。这比跑完 176 个 run 再返工便宜得多。

⚠️ **3 个 seed 估出的标准差本身极不可靠**（df=2，标准差的 95% CI 约为
真值的 0.52~6.3 倍），所以本脚本给出的建议一律取**保守上界**。

用法::

    # 两臂各 3 个 seed 的 run 目录
    python scripts/measure_power.py \\
        --arm-a runs/base_fedavg_seed0 runs/base_fedavg_seed1 runs/base_fedavg_seed2 \\
        --arm-b runs/m_fedosp_seed0 runs/m_fedosp_seed1 runs/m_fedosp_seed2

    # 换指标（默认 worst_qwk）
    python scripts/measure_power.py --metric macro_qwk --arm-a ... --arm-b ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: 双侧 α=0.05 与 power=80% 对应的正态分位数
Z_ALPHA, Z_BETA = 1.959964, 0.841621

#: 设计文档 §2.5 给出的 C1 诚实效应量区间，外加一个更保守的下界
TARGET_DELTAS = (0.015, 0.02, 0.03, 0.04)


def read_metric(run_dir: Path, metric: str) -> Optional[float]:
    """从 ``result.json`` 取一个标量指标。

    先找 ``test_summary``（macro_qwk / worst_qwk / std_qwk / weighted_qwk），
    再退回 ``external``（未见中心指标），最后试 ``test_per_client`` 的同名键。
    """
    f = run_dir / "result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    for block in ("test_summary", "external"):
        v = (r.get(block) or {}).get(metric)
        if isinstance(v, (int, float)):
            return float(v)
    # worst-client 也可以从 per-client 里现算，便于换指标（如 worst mae）
    per = r.get("test_per_client") or {}
    vals = [m.get(metric) for m in per.values() if isinstance(m.get(metric), (int, float))]
    if vals and metric.startswith("worst_"):
        return float(min(vals))
    return None


def sd_ci(sd: float, n: int) -> Tuple[float, float]:
    r"""标准差的 95% 置信区间（卡方法）。

    :math:`(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}`，所以
    :math:`\sigma \in [s\sqrt{(n-1)/\chi^2_{0.975}},\; s\sqrt{(n-1)/\chi^2_{0.025}}]`。

    n=3 时区间宽到 **0.52~6.29 倍** —— 这就是为什么 3 个 seed 估出的
    标准差不能当准数用，本脚本的建议一律基于**上界**。
    """
    chi2 = {  # {df: (chi2_0.025, chi2_0.975)}
        1: (0.000982, 5.0239), 2: (0.05064, 7.3778), 3: (0.2158, 9.3484),
        4: (0.4844, 11.143), 5: (0.8312, 12.833), 6: (1.2373, 14.449),
        7: (1.6899, 16.013), 8: (2.1797, 17.535), 9: (2.7004, 19.023),
    }.get(n - 1)
    if chi2 is None or sd <= 0:
        return (sd, sd)
    lo, hi = chi2
    return (sd * math.sqrt((n - 1) / hi), sd * math.sqrt((n - 1) / lo))


#: 配对 t 检验的双侧 0.05 临界值，按自由度 n-1。
T_CRIT = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 17: 2.110, 19: 2.093, 24: 2.064, 29: 2.045,
}


def power_at(n: int, sd_diff: float, delta: float, trials: int = 20000,
             seed: int = 0) -> float:
    """配对 t 检验在给定 n / σ_d / Δ 下的**实际**功效（蒙特卡洛）。

    ★ **不要用基于正态分位数的解析公式**（:math:`n=(z_{\\alpha/2}+z_\\beta)^2\\sigma^2/\\Delta^2`）。
    它在小 n 下会严重低估所需 seed 数，因为真正的临界值来自 t 分布：
    n=3 时 :math:`t_{2,0.975}=4.303`，而 z 只有 1.96 —— 差 2.2 倍，
    平方后是 **4.8 倍**的样本量差距。

    实测对比（σ=0.0386 即 IDRiD 的 seed 标准差，Δ=0.03）：

    ==========  ==================  ==================
    ρ            解析(z) 说需要        模拟实际功效 @3 seed
    ==========  ==================  ==================
    0.5          13 个                **12.8%**
    0.8          6 个                 ~38%（ρ=0.9）
    ==========  ==================  ==================

    也就是说"3 个 seed 在 ρ≥0.8 时勉强够"这个结论是**错的** ——
    即使 ρ=0.9，3 个 seed 也只有 38% 功效。
    """
    import numpy as np

    crit = T_CRIT.get(n - 1)
    if crit is None or sd_diff <= 0 or delta <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    d = delta + sd_diff * rng.normal(0, 1, (trials, n))
    m = d.mean(1)
    s = d.std(1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (s / math.sqrt(n))
    return float((np.abs(t) > crit).mean())


def required_seeds(sd_diff: float, delta: float, target: float = 0.80) -> int:
    """达到 ``target`` 功效所需的 seed 数（用模拟逐个试，不用解析公式）。"""
    if sd_diff <= 0 or delta <= 0:
        return 2
    for n in sorted(k + 1 for k in T_CRIT):
        p = power_at(n, sd_diff, delta)
        if not math.isnan(p) and p >= target:
            return n
    return 30  # 超出表范围，直接给个"很多"的信号


def mean_sd(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-a", nargs="+", type=Path, required=True,
                    help="基线臂的 run 目录（按 seed 顺序）")
    ap.add_argument("--arm-b", nargs="+", type=Path, required=True,
                    help="对比臂（通常是 FedOSP）的 run 目录，**seed 顺序必须与 A 一致**")
    ap.add_argument("--metric", default="worst_qwk",
                    help="主指标，默认 worst_qwk（本文主指标）")
    ap.add_argument("--name-a", default="A(基线)")
    ap.add_argument("--name-b", default="B(FedOSP)")
    args = ap.parse_args(argv)

    a = [read_metric(d, args.metric) for d in args.arm_a]
    b = [read_metric(d, args.metric) for d in args.arm_b]
    miss = [str(d) for d, v in list(zip(args.arm_a, a)) + list(zip(args.arm_b, b)) if v is None]
    if miss:
        print(f"✗ 以下 run 读不到指标 {args.metric!r}：")
        for m in miss:
            print(f"    {m}")
        return 1
    if len(a) != len(b):
        print(f"✗ 两臂 run 数不等（{len(a)} vs {len(b)}）。配对分析要求 seed 一一对应。")
        return 1
    n = len(a)
    if n < 2:
        print("✗ 至少需要 2 个 seed 才能估方差。")
        return 1

    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    diffs = [y - x for x, y in zip(a, b)]
    md, sd_d = mean_sd(diffs)

    # 两臂相关性：配对差方差 = sa^2 + sb^2 - 2*rho*sa*sb
    rho = float("nan")
    if sa > 0 and sb > 0:
        rho = (sa * sa + sb * sb - sd_d * sd_d) / (2 * sa * sb)
        rho = max(-1.0, min(1.0, rho))

    print(f"=== 指标：{args.metric}（{n} 个 seed 配对）===\n")
    print(f"{'seed':>6s} {args.name_a:>12s} {args.name_b:>12s} {'差(B-A)':>10s}")
    for i, (x, y) in enumerate(zip(a, b)):
        print(f"{i:>6d} {x:>12.4f} {y:>12.4f} {y - x:>+10.4f}")
    print(f"{'均值':>6s} {ma:>12.4f} {mb:>12.4f} {md:>+10.4f}")
    print(f"{'标准差':>6s} {sa:>12.4f} {sb:>12.4f} {sd_d:>10.4f}")

    lo, hi = sd_ci(sd_d, n)
    print(f"\n配对差标准差 σ_d = {sd_d:.4f}，95% CI = [{lo:.4f}, {hi:.4f}]")
    print(f"  ⚠ n={n} 时这个区间宽达 {hi / max(lo, 1e-9):.1f} 倍 —— σ_d 本身很不准，")
    print(f"     所以下面同时给「按点估」和「按上界」两套建议，**以上界为准**。")
    print(f"两臂相关性 ρ ≈ {rho:.3f}" if not math.isnan(rho) else "两臂相关性 ρ 无法估计")

    if md != 0 and sd_d > 0:
        t = md / (sd_d / math.sqrt(n))
        print(f"\n当前观测效应 Δ = {md:+.4f}，配对 t = {t:+.2f}（df={n - 1}）")
        crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1)
        if crit:
            print(f"  双侧 0.05 的临界值 t_{n - 1} = {crit:.3f} → "
                  f"{'已显著' if abs(t) > crit else '**未达显著**'}")

    print(f"\n=== 达到 80% 功效所需的 seed 数（配对 t 检验，双侧 α=0.05）===")
    print(f"{'目标效应 Δ':>12s} {'按点估 σ_d':>12s} {'按上界 σ_d':>12s} "
          f"{f'当前 {n} seed 的功效':>20s}")
    for d in TARGET_DELTAS:
        n_pt = required_seeds(sd_d, d)
        n_hi = required_seeds(hi, d)
        p_now = power_at(n, hi, d)          # 用上界算当前功效，保守
        flag = "✓" if p_now >= 0.8 else ("△" if p_now >= 0.5 else "✗")
        print(f"{d:>12.3f} {n_pt:>12d} {n_hi:>12d} "
              f"{f'{p_now:.0%} {flag}':>20s}")
    print("  （功效列按 σ_d 的 95% 上界算，取保守值；≥80% 才算够）")

    print("\n=== 怎么用这张表 ===")
    print("  · 设计文档 §2.5 给 C1 的诚实效应量是 **+0.02~0.04**，所以看那两行。")
    print("  · 若「按上界」那列 ≤3 → 3 个 seed 可以放行整个矩阵。")
    print("  · 若 >3 → 在**跑正式矩阵之前**把 seeds 调上去，代价远低于事后返工：")
    print("      sed -i '' 's/0;1;2/0;1;2;3;4/' configs/experiment_matrix.csv")
    print("  · worst-client 之外也建议看 Messidor-2 外测（1744 张，测试集抽样误差")
    print("    比 IDRiD 的 103 张小 4 倍，功效高得多）。设计文档要求两个主指标")
    print("    **同时**显著，瓶颈在 worst-client 这一侧。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
