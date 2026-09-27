#!/usr/bin/env python3
"""把 ``runs/bench_*`` 汇总成论文的主表：配对差 + 置信区间 + 可分辨下限。

**为什么必须配对**：本项目实测 Messidor-2 QWK 的 seed 间边际 SD 为 0.033，
而同 seed 配对差的 SD 只有 0.027。更要紧的是，不同方法在**同一个 seed** 上
会一起偏高或偏低（共享数据划分与初始化），所以独立两样本检验会把这部分
共同波动算进误差里，白白损失功效。所有比较一律按 seed 配对。

**为什么要报"可分辨下限"**：这篇论文的主张不是"某方法更好"，而是
"在可达的样本量下这些方法互相分辨不开"。所以每一行除了点估计，
还要给出该行实际能排除多大的差异——没有这一列，零结果会被误读成功效不足。

用法::

    python scripts/benchmark_table.py                      # 默认对 bench_fedavg
    python scripts/benchmark_table.py --ref bench_fedproto # 换参照臂
    python scripts/benchmark_table.py --metric messidor_auroc --markdown
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 样本量的唯一实现在包里。这一列（"检出指定差异所需配对数"）是论文表 4 的来源，
# 曾经用正态近似算，每一格都少 2 个配对。见 pairs_needed 的 docstring。
from fedosp.stats import pairs_needed  # noqa: E402

#: 主端点在 result.json 里的取值路径。改端点只要在这里加一行。
#: ``external`` 是 ``evaluate_predictions`` 的**扁平**返回值（未见中心只有一个，
#: 所以不按数据集名再嵌一层）；``test_summary`` 来自 ``aggregate_over_clients``。
METRICS = {
    "messidor_qwk": ("external", "qwk"),
    "messidor_auroc": ("external", "referable_auroc"),
    "messidor_brier": ("external", "brier"),
    "messidor_ece": ("external", "ece"),
    "worst_qwk": ("test_summary", "worst_qwk"),
    "macro_qwk": ("test_summary", "macro_qwk"),
}
#: 越小越好的指标，打印时方向要翻过来说明
LOWER_IS_BETTER = {"messidor_brier", "messidor_ece"}

# t 分布双侧 97.5% 分位，键是自由度 n-1。样本量小，不能用 1.96。
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        14: 2.145, 19: 2.093, 29: 2.045}


def t_crit(df: int) -> float:
    if df in T975:
        return T975[df]
    if df <= 0:
        return float("nan")
    return 1.96 + 2.4 / df  # df>29 时的够用近似


def dig(obj, path: Tuple[str, ...]):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj if isinstance(obj, (int, float)) else None


def load_runs(runs_dir: Path, metric: str) -> Dict[str, Dict[int, float]]:
    """扫描 runs/，返回 {exp_id: {seed: 指标值}}。

    只收 ``tier == "main"`` 的结果。pilot 产物混进正式表是这个项目
    早先踩过的坑，这里直接在入口拦掉。
    """
    path = METRICS[metric]
    out: Dict[str, Dict[int, float]] = defaultdict(dict)
    skipped_pilot = 0
    for rj in sorted(runs_dir.glob("*/result.json")):
        m = re.fullmatch(r"(.+)_seed(\d+)", rj.parent.name)
        if not m:
            continue
        exp_id, seed = m.group(1), int(m.group(2))
        try:
            res = json.loads(rj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ! {rj} 解析失败，跳过")
            continue
        # tier 在 provenance 下，不在顶层。取错位置会让 pilot 结果全部混进正式表
        # 而且毫无提示 —— 正是这个项目反复出过的那类静默失败。
        if (res.get("provenance") or {}).get("tier") == "pilot":
            skipped_pilot += 1
            continue
        v = dig(res, path)
        if v is not None:
            out[exp_id][seed] = float(v)
    if skipped_pilot:
        print(f"  （跳过 {skipped_pilot} 个 tier=pilot 的结果，它们不可用于正式表）")
    return out


def paired(a: Dict[int, float], b: Dict[int, float]) -> Tuple[List[int], List[float]]:
    """取两臂共有的 seed，返回 (seed 列表, a-b 差值列表)。"""
    common = sorted(set(a) & set(b))
    return common, [a[s] - b[s] for s in common]


def summarize(d: List[float]) -> Dict[str, float]:
    n = len(d)
    if n < 2:
        return {"n": n, "mean": d[0] if d else float("nan"), "sd": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "t": float("nan"),
                "half": float("nan"), "pos": float(n)}
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    half = t_crit(n - 1) * se
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half,
            "t": mean / se if se > 0 else float("nan"), "half": half,
            "pos": sum(1 for x in d if x > 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--metric", default="messidor_qwk", choices=sorted(METRICS))
    ap.add_argument("--ref", default="bench_fedavg", help="参照臂 exp_id")
    ap.add_argument("--prefix", default="bench_", help="只统计这个前缀的实验")
    ap.add_argument("--markdown", action="store_true", help="输出 Markdown 表格")
    args = ap.parse_args()

    data = load_runs(args.runs_dir, args.metric)
    data = {k: v for k, v in data.items() if k.startswith(args.prefix)}
    if args.ref not in data:
        print(f"找不到参照臂 {args.ref}。现有：{sorted(data) or '（无）'}")
        return 1

    ref = data[args.ref]
    lower_better = args.metric in LOWER_IS_BETTER
    print(f"\n端点 {args.metric}"
          f"{'（越小越好）' if lower_better else '（越大越好）'}"
          f"　参照臂 {args.ref}（n={len(ref)}, 均值 {sum(ref.values())/len(ref):.4f}）\n")

    rows = []
    for exp in sorted(data):
        if exp == args.ref:
            continue
        seeds, d = paired(data[exp], ref)
        if not d:
            continue
        s = summarize(d)
        s["exp"] = exp.replace(args.prefix, "")
        s["abs"] = sum(data[exp][x] for x in seeds) / len(seeds)
        rows.append(s)
    rows.sort(key=lambda r: r["mean"], reverse=not lower_better)

    if args.markdown:
        print("| 方法 | n | 绝对值 | 相对参照 | 95% CI | 同向 | 本行可排除 |")
        print("|---|---:|---:|---:|---|---:|---|")
        for r in rows:
            sig = "" if (r["lo"] <= 0 <= r["hi"]) else " **\\***"
            print(f"| {r['exp']} | {r['n']} | {r['abs']:.4f} | {r['mean']:+.4f}{sig} "
                  f"| [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['pos']:.0f}/{r['n']} "
                  f"| >{r['half']:.3f} |")
    else:
        print(f"{'方法':14s} {'n':>2s} {'绝对值':>8s} {'相对参照':>9s} "
              f"{'95% CI':>20s} {'同向':>6s} {'可排除':>8s}")
        print("─" * 78)
        for r in rows:
            sig = "*" if not (r["lo"] <= 0 <= r["hi"]) else " "
            print(f"{r['exp']:14s} {r['n']:>2d} {r['abs']:>8.4f} {r['mean']:>+9.4f}{sig}"
                  f" [{r['lo']:>+7.4f},{r['hi']:>+7.4f}] {r['pos']:>3.0f}/{r['n']:<2d} "
                  f"  >{r['half']:.3f}")

    if rows:
        ns = [r["n"] for r in rows]
        pooled_sd = math.sqrt(sum(r["sd"] ** 2 * (r["n"] - 1) for r in rows if r["n"] > 1)
                              / max(1, sum(r["n"] - 1 for r in rows if r["n"] > 1)))
        crossing = sum(1 for r in rows if r["lo"] <= 0 <= r["hi"])
        print(f"\n合并配对 SD = {pooled_sd:.4f}（{len(rows)} 个对比）")
        print(f"CI 跨零（与参照臂不可区分）：{crossing}/{len(rows)}")
        print("\n以该 SD，检出指定差异所需配对数（双侧 α=0.05，功效 0.8，精确非中心 t）：")
        for delta in (0.005, 0.010, 0.015, 0.020, 0.030):
            need = pairs_needed(delta, pooled_sd)
            print(f"   Δ={delta:.3f} → {need:>4d} 对"
                  + ("   ← 文献典型声称区间" if delta in (0.010, 0.015, 0.020) else ""))
        if min(ns) < 5:
            print(f"\n⚠ 最小 n={min(ns)}，CI 会很宽。正式表建议每臂至少 5 个 seed。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
