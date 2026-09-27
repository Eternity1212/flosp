#!/usr/bin/env python3
"""把外测指标的 seed 间方差拆成"选择方差"与"训练方差"。

**要回答的问题**：我们实测 Messidor-2 QWK 的 seed 间 SD 为 0.033，一直把它
当作训练随机性。但模型选择是在约 100 轮上对验证 ``macro_qwk`` 取 **argmax**，
而实测最佳轮散布在 37/53/60/77/99（对平台期均匀分布的 KS 检验 p=0.982）。
验证集不大（IDRiD 只有 103 张），每轮验证噪声约 0.02 量级 ——
在平坦曲线上对 100 个候选取 argmax，是典型的 winner's curse。

那么 0.033 里有多少其实来自"选哪一轮"，而不是"训得怎样"？

**做法**：用 ``--eval-external-every`` 记下平台期各轮的外测指标，则

.. math::

    \\mathrm{Var}_{\\text{total}} = \\underbrace{\\mathrm{Var}_{\\text{选择}}}_{\\text{run 内、平台各轮之间}}
                                  + \\underbrace{\\mathrm{Var}_{\\text{训练}}}_{\\text{run 之间、平台均值的差异}}

前者在**单次 run 内部**就能测，不需要跨 seed 重复；后者是各 run 平台均值的方差。
这是标准的单因素方差分量估计（run 为随机效应）。

**为什么这个拆分重要**：两种结果都有价值。
选择项占大头 → 改选择规则（平台内权重平均 / 平滑后再取 argmax）能压掉大部分
噪声，这是可落地的方法贡献；训练项占大头 → 噪声不可约，是更强的阴性结论。

**数据从哪来**：独立矩阵 ``configs/variance_decomposition.csv``
（``diag_varsel_fedavg`` / ``diag_varsel_fedproto``，各 5 个 seed，共 11.8 GPU·h）。
基准矩阵的 A/B 两层也带了 ``--eval-external-every 5``，跑完横评同样有数据，
但那是 131.9 GPU·h 的计划，这个测量不该被绑在上面。
协作方的冻结面板**不带**这个开关，供不了这份数据，且**不得为此改动**。

**判据已预注册**：``reports/预注册_方差分解_2026-09-27.md``，
选择占比 >60% 支持方法贡献、<30% 判噪声不可约、30–60% 两个都报。
**阈值不得在看到结果之后修改。**

用法::

    python scripts/variance_decomposition.py --prefix diag_varsel
    python scripts/variance_decomposition.py --prefix bench_fedavg
    python scripts/variance_decomposition.py --prefix diag_varsel --metric ext_referable_auroc
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def window_start(n_points: int, frac: float = 0.5, min_pts: int = 5) -> int:
    """后段窗口的起点下标：取最后 ``frac`` 比例的采样点。

    **为什么是固定窗口而不是自动检测平台**：作者先后写过三版"自动找平台"的
    启发式（相对峰值阈值 → 均值平滑 → 中位数平滑 + 噪声感知阈值），每一版都
    在合成数据上被找出系统性偏差：

    * 原始曲线取 max：单个噪声尖峰把阈值抬到只有它够得着，平台被判到第 93 点；
    * 均值平滑：窗口整体被尖峰抬高（0.70 的平台抬到 0.725），照样判错；
      且 ``np.convolve(mode="same")`` 的零填充让尾部从 0.70 塌到 0.445；
    * 中位数平滑 + "峰值−zσ"：平滑峰值本身被噪声抬高（0.70 → 0.7269），
      而残差 MAD 又低估噪声（0.0113 vs 真实 0.020），两个偏差叠加，
      阈值站到平台之上，一条 35 轮进平台的曲线被判到第 76 轮。

    这些偏差都朝同一个方向——截短窗口、低估 run 内方差——而 run 内方差正是
    本脚本要估的量，低估它会把结论错误地推向"训练主导"，也就是把一个可能
    成立的方法贡献误判成阴性。

    带三个可调参数、需要反复打补丁的启发式，正是这个项目反复栽跟头的东西
    （台账 §11 记了 12 个实现缺陷，其中 8 个是静默失败）。所以改用固定窗口：
    没有可调阈值、结果可直接复算、窗口写进论文即可审计。
    收敛速度的差异改由 :func:`trend_pvalue` **显式检查并报告**，
    而不是藏在一个自动判定里。
    """
    return max(0, min(int(n_points * (1 - frac)), max(0, n_points - min_pts)))


def trend_pvalue(x: List[int], y: List[float]) -> float:
    """窗口内是否还有残余趋势（Spearman）。

    固定窗口的代价是"可能把未收敛的一段算进来"。与其用启发式回避，
    不如直接测出来：若 p 小、说明该 run 在窗口内仍在变好或变坏，
    它的 run 内方差里混了趋势项，需要在论文里单独说明。
    """
    if len(x) < 4:
        return float("nan")
    from scipy import stats
    return float(stats.spearmanr(x, y).pvalue)


def pairs_needed(delta: float, sd: float, power: float = 0.80,
                 alpha: float = 0.05, n_max: int = 2000) -> int:
    """检出配对差 ``delta`` 所需的配对数（双侧 ``alpha``，功效 ``power``）。

    **必须用精确的非中心 t，不能用正态近似。** 常见的解析式
    :math:`n=(z_{1-\\alpha/2}+z_{1-\\beta})^2\\sigma_d^2/\\Delta^2`（系数 7.849）
    有两处偏差：临界值是 :math:`t_{n-1}` 而不是 :math:`z`，且 :math:`s` 是估计量。
    两处都朝**低估**方向走。

    这个坑本项目踩过两次。第一次是 B0 闸门的功效表（commit ``8625569``
    已把 ``measure_power.py`` 改成蒙特卡洛，并在 commit message 里写下
    "n=3 时临界值是 t₂=4.303 而非 z=1.96，差 2.2 倍、平方后 4.8 倍样本量"）；
    第二次是本函数的前身和台账 §8.1 的样本量表，都还留着 ``2.8016**2``。

    在 :math:`\\sigma_d=0.0272` 上，两者的差恰好是**每一格都少 2 个配对**：

    ====== =========== ===========
    Δ       z 近似       精确 t
    ====== =========== ===========
    0.010     59          **61**
    0.020     15          **17**
    0.030      7           **9**
    ====== =========== ===========

    绝对量级不变，但小 n 时相对误差最大（0.030 那一格差 29%）。

    Returns:
        所需配对数；``delta`` 非正或 ``sd`` 非正时返回 ``n_max``。
    """
    import warnings

    from scipy import stats

    if delta <= 0 or sd <= 0:
        return n_max
    for n in range(3, n_max + 1):
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        ncp = delta * math.sqrt(n) / sd
        with warnings.catch_warnings():
            # df=3 时 scipy 的 boost 后端会报 "divide by zero in _nct_sf"，
            # 但它自己 clip 到 [0,1]，返回值仍然正确（实测 df=3 的功效远小于
            # 0.8，循环照常往下走）。只是噪声，不是数值错误。
            warnings.simplefilter("ignore", RuntimeWarning)
            if stats.nct.sf(t_crit, n - 1, ncp) >= power:
                return n
    return n_max


def load(runs_dir: Path, prefix: str, metric: str, val_key: str = "macro_qwk"
         ) -> Dict[str, List[Tuple[int, float, float]]]:
    """返回 {run_tag: [(round, val, ext), ...]}，只保留有 ext 的轮。"""
    out: Dict[str, List[Tuple[int, float, float]]] = {}
    for rj in sorted(runs_dir.glob("*/result.json")):
        if not rj.parent.name.startswith(prefix):
            continue
        res = json.loads(rj.read_text(encoding="utf-8"))
        if (res.get("provenance") or {}).get("tier") == "pilot":
            continue
        pts = [(r["round"], r.get(val_key), r.get(metric))
               for r in res.get("history", [])
               if r.get(metric) is not None and r.get(val_key) is not None]
        if len(pts) >= 3:
            out[rj.parent.name] = pts
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--prefix", default="bench_", help="只看这个前缀的 run")
    ap.add_argument("--metric", default="ext_qwk", help="history 里的外测指标字段")
    ap.add_argument("--window-frac", type=float, default=0.5,
                    help="用最后这个比例的采样点做统计（默认后 50%%）。"
                         "固定窗口、无自动判定，窗口内是否仍有趋势会单独报出")
    ap.add_argument("--total-sd", type=float, default=0.033,
                    help="已知的 seed 间总 SD，用于对照（Messidor QWK 实测 0.033）")
    args = ap.parse_args()

    runs = load(args.runs_dir, args.prefix, args.metric)
    if len(runs) < 2:
        print(f"至少需要 2 个带 history.{args.metric} 的 run（当前 {len(runs)} 个）。\n"
              f"这些数据由 --eval-external-every 5 产生。最便宜的一条路是专用矩阵\n"
              f"（11.8 GPU·h / 10 run，不依赖 131.9 GPU·h 的横评是否跑完）：\n"
              f"  python scripts/scheduler.py --matrix configs/variance_decomposition.csv \\\n"
              f"         --stage bench --gpus 0,1,2,3 --jobs-per-gpu 2\n"
              f"  python scripts/variance_decomposition.py --prefix diag_varsel")
        return 1

    print(f"\n指标 {args.metric}　窗口 后 {args.window_frac:.0%}　{len(runs)} 个 run\n")
    print(f"{'run':24s} {'采样':>4s} {'窗口起':>6s} {'窗口内均值':>10s} {'窗口内SD':>9s} "
          f"{'argmax选中':>10s} {'趋势p':>7s}")
    print("─" * 82)

    within_ss, within_df, means, picked, trending = 0.0, 0, [], [], []
    for tag, pts in sorted(runs.items()):
        rounds = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        exts = [p[2] for p in pts]
        i0 = window_start(len(pts), args.window_frac)
        ext_w = np.array(exts[i0:], float)
        if len(ext_w) < 2:
            print(f"{tag:24s} 窗口内采样点不足，跳过")
            continue
        # argmax-on-val 会选中的那一轮对应的外测值
        j = int(np.argmax(vals))
        p_trend = trend_pvalue(rounds[i0:], list(ext_w))
        if p_trend == p_trend and p_trend < 0.05:
            trending.append(tag)
        means.append(ext_w.mean())
        picked.append(exts[j])
        within_ss += ((ext_w - ext_w.mean()) ** 2).sum()
        within_df += len(ext_w) - 1
        print(f"{tag:24s} {len(pts):>4d} {rounds[i0]:>6d} {ext_w.mean():>10.4f} "
              f"{ext_w.std(ddof=1):>9.4f} {'r'+str(rounds[j]):>10s} {p_trend:>7.3f}")

    if trending:
        print(f"\n⚠ {len(trending)} 个 run 在窗口内仍有显著趋势："
              f"{', '.join(trending[:4])}{' …' if len(trending) > 4 else ''}")
        print("  它们的窗口内方差混了趋势项（未收敛），会**高估**选择方差。"
              "论文里需单独说明，或改用更靠后的窗口复算。")

    if within_df == 0 or len(means) < 2:
        print("\n有效数据不足。")
        return 1

    sd_within = math.sqrt(within_ss / within_df)          # 选择项
    sd_between = float(np.std(means, ddof=1))             # 训练项（平台均值之差）
    sd_picked = float(np.std(picked, ddof=1))             # 实际协议下的 seed 间 SD
    total = math.hypot(sd_within, sd_between)

    print("\n" + "═" * 82)
    print(f"{'选择方差  SD_within（窗口内轮次间）':40s} {sd_within:>8.4f}")
    print(f"{'训练方差  SD_between（run 间窗口均值）':40s} {sd_between:>8.4f}")
    print(f"{'合成      sqrt(within²+between²)':40s} {total:>8.4f}")
    print(f"{'实测      argmax 协议下的 seed 间 SD':40s} {sd_picked:>8.4f}")
    print(f"{'参照      已知 Messidor QWK seed SD':40s} {args.total_sd:>8.4f}")

    share = sd_within ** 2 / max(1e-12, total ** 2)
    print(f"\n★ 选择占总方差的 {share:.0%}")
    if share > 0.6:
        print("  → **选择主导**。改选择规则（窗口内权重平均 / 平滑后取 argmax）")
        print("     有望压掉大部分噪声，这是可落地的方法贡献。")
        print(f"     若选择项归零，seed SD 可降至 {sd_between:.4f}"
              f"（降 {1 - sd_between / max(1e-12, total):.0%}），")
        need0 = pairs_needed(0.010, total)
        need1 = pairs_needed(0.010, sd_between)
        print(f"     检出 Δ=0.010 所需配对数从 {need0} 降到 {need1}。")
    elif share < 0.3:
        print("  → **训练主导**。选择规则不是主要矛盾，SWA 类做法帮不上。")
        print("     这本身是更强的阴性结论：噪声不可约，方法比较在这个规模下不可行。")
    else:
        print("  → 两者相当，单改选择规则只能压掉部分噪声。")

    # argmax 是否真的选到了比平台均值更好的轮 —— winner's curse 的直接证据
    lift = float(np.mean(np.array(picked) - np.array(means)))
    print(f"\nargmax 选中轮的外测值 − 窗口均值 = {lift:+.4f}")
    if lift < 0.002:
        print("  验证集上的 argmax 并没有换来更好的外测表现 ——")
        print("  说明它主要在拟合验证噪声，而不是在挑真正更好的模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
