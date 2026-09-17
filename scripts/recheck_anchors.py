"""用已保存的 per-sample 预测**零 GPU 成本**复算锚点，并在多种指标定义下对比。

为什么需要这个
--------------
锚点对不上时，第一反应往往是"重跑几个 seed / 多训几个 epoch"。但那要花 GPU 时间，
而且**答不了最常见的那个问题：指标定义是不是一致的**。

2026-09-17 的真实案例：APTOS 实测 referable(≥2) 二分类 AUROC **0.9753**，
而 RETFound 全量微调的参考值是 0.943 —— 我们只训练 0.23% 的参数却高出
**3.3 个标准误**，这不合理。头号嫌疑是文献报的其实是 **5 类 macro one-vs-rest
AUROC**（系统性更低，因为稀有等级拖后腿）。

这个假设**不需要任何训练就能验证**：只要 ``predictions.npz`` 在，
换个指标重算一遍即可。如果 macro OvR 落到 0.943 附近，问题就定性了。

用法::

    # 单个 run
    python scripts/recheck_anchors.py runs/s_anchor_aptos

    # 批量
    python scripts/recheck_anchors.py runs/s_anchor_*

    # 换判据严格程度
    python scripts/recheck_anchors.py runs/s_anchor_idrid --z 2.0

输出对每个 client 给出：两种 AUROC 定义、各自的标准误、偏离文献锚点的 SE 倍数，
以及在当前判据下是 OK 还是 OFF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedosp.metrics import (  # noqa: E402
    ANCHOR_FLOOR,
    ANCHOR_Z,
    LITERATURE_ANCHORS,
    macro_ovr_auroc,
    quadratic_weighted_kappa,
    referable_auroc,
    referable_auroc_se,
)


def load_predictions(run_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """从 ``predictions.npz`` 读出 ``{client: {y, probs}}``。"""
    npz = run_dir / "predictions.npz"
    if not npz.exists():
        return {}
    data = np.load(npz)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.endswith("__y"):
            out.setdefault(key[:-3], {})["y"] = data[key]
        elif key.endswith("__probs"):
            out.setdefault(key[:-7], {})["probs"] = data[key]
        elif key in ("y", "probs"):                 # run_central 单集模式
            out.setdefault(run_dir.name, {})[key] = data[key]
    return {k: v for k, v in out.items() if "y" in v and "probs" in v}


def analyse(client: str, y: np.ndarray, probs: np.ndarray, z: float) -> Dict[str, object]:
    ref = LITERATURE_ANCHORS["retfound_finetune_auroc"].get(client)
    bin_auc = referable_auroc(y, probs)
    ovr_auc = macro_ovr_auroc(y, probs)
    se = referable_auroc_se(y, bin_auc)
    n_pos = int((np.asarray(y) >= 2).sum())
    row: Dict[str, object] = {
        "client": client,
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": int(len(y)) - n_pos,
        "qwk": quadratic_weighted_kappa(y, np.asarray(probs).argmax(1)),
        "referable_auroc": bin_auc,
        "macro_ovr_auroc": ovr_auc,
        "se": se,
        "ref": ref,
    }
    if ref is not None and not np.isnan(se) and se > 0:
        tol = max(ANCHOR_FLOOR, z * se)
        row["delta_bin"] = bin_auc - ref
        row["sigma_bin"] = abs(bin_auc - ref) / se
        row["verdict_bin"] = "OK" if abs(bin_auc - ref) <= tol else "OFF"
        if not np.isnan(ovr_auc):
            row["delta_ovr"] = ovr_auc - ref
            row["sigma_ovr"] = abs(ovr_auc - ref) / se
            row["verdict_ovr"] = "OK" if abs(ovr_auc - ref) <= tol else "OFF"
    return row


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run 目录（可多个 / 可用通配）")
    ap.add_argument("--z", type=float, default=ANCHOR_Z,
                    help=f"容差取几个标准误，默认 {ANCHOR_Z}")
    args = ap.parse_args(argv)

    rows: List[Dict[str, object]] = []
    for run_dir in args.runs:
        if not run_dir.is_dir():
            continue
        preds = load_predictions(run_dir)
        if not preds:
            print(f"⚠ {run_dir}：没有 predictions.npz —— 该 run 早于此功能，"
                  f"需重跑一次才能零成本复算（之后就不用了）")
            continue
        for client, d in sorted(preds.items()):
            rows.append({"run": run_dir.name, **analyse(client, d["y"], d["probs"], args.z)})

    if not rows:
        print("没有可分析的预测文件。")
        return 1

    hdr = (f"{'run':22s} {'client':10s} {'n':>5s} {'阳/阴':>9s} {'QWK':>7s} "
           f"{'refer.AUROC':>12s} {'macroOvR':>9s} {'SE':>7s} {'文献':>7s} "
           f"{'二分类偏差':>12s} {'OvR偏差':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ref = r.get("ref")
        b = (f"{r['delta_bin']:+.4f}/{r['sigma_bin']:.1f}σ {r['verdict_bin']}"
             if "delta_bin" in r else "—")
        o = (f"{r['delta_ovr']:+.4f}/{r['sigma_ovr']:.1f}σ {r['verdict_ovr']}"
             if "delta_ovr" in r else "—")
        print(f"{str(r['run'])[:22]:22s} {r['client']:10s} {r['n']:>5d} "
              f"{r['n_pos']:>4d}/{r['n_neg']:<4d} {r['qwk']:>7.4f} "
              f"{r['referable_auroc']:>12.4f} {r['macro_ovr_auroc']:>9.4f} "
              f"{r['se']:>7.4f} {(f'{ref:.4f}' if ref else '—'):>7s} {b:>12s} {o:>12s}")

    print("\n怎么读这张表")
    print("  · σ = 偏差 / 标准误。**小于 2σ 基本是噪声**，不要据此改数据或停工。")
    print("    小测试集的 σ 很容易被误读：IDRiD n=103 时 SE≈0.040，")
    print("    固定 ±0.02 的容差只有 0.48σ，实现完全正确也约 63% 概率误报 FAIL。")
    print("  · 偏差**为正**（实测高于文献）比为负更可疑：用 0.23% 参数的 LoRA")
    print("    超过全量微调的 ViT-L 不合理，通常是指标定义或测试集构成不一致。")
    print("  · ★ 对比最后两列：若 `OvR偏差` 明显比 `二分类偏差` 更接近 0，")
    print("    说明**文献报的是 5 类 macro OvR 而我们在比二分类**，")
    print("    这不是模型问题，改指标定义即可，不需要重训。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
