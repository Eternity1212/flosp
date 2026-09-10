#!/usr/bin/env python3
"""生成论文用图 F2–F7（矢量 PDF + 位图 PNG）。

=====  ======================================================================
图      内容与作用
=====  ======================================================================
F2     四个 client 的等级分布 + 样本量对比 —— 直观展示"标签偏移 + 64:1 规模差"
F3     FSR 前后的浅层幅度谱 —— **证明 P1 的关键图**，展示风格被压平
F4     深层原型的 t-SNE / PCA —— **证明 P2 的关键图**，展示原型排成有序流形
F5     标签效率曲线 —— 10%/25%/50%/100% 下我方与基线的差距
F6     收敛曲线 + 累计通信量双轴 —— 证明"少通信也能更快更好"
F7     per-client QWK 箱线图 —— **证明 P3 的关键图**，展示 worst-client 被抬起
=====  ======================================================================

用法::

    python scripts/make_figures.py --runs runs --out figures            # 全部
    python scripts/make_figures.py --figs F3 F7                        # 指定
    python scripts/make_figures.py --figs F3 --ckpt runs/m_fedosp_seed0/best.pt

F3 与 F4 需要模型权重和数据，没有就自动跳过并说明原因。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
LOGGER = logging.getLogger("figures")

TRAIN_CLIENTS = ["eyepacs", "aptos", "ddr", "idrid"]
HELDOUT = "messidor2"

#: 图里显示用的正式名称（别用 .capitalize()，那会写出 Eyepacs / Ddr）
DISPLAY_NAME = {
    "eyepacs": "EyePACS", "aptos": "APTOS", "ddr": "DDR",
    "idrid": "IDRiD", "messidor2": "Messidor-2",
}

#: 对比图里要画哪些方法、按什么顺序。**必须与 experiment_matrix.csv 的 exp_id 一致**，
#: 否则一个基线都匹配不上，图里只剩我方一条曲线。
COMPARE_ORDER = [
    ("base_fedavg", "FedAvg"),
    ("base_fedbn", "FedBN"),
    ("base_fedproto", "FedProto"),
    ("base_feduaa", "FedUAA-style"),
    ("m_fedosp", "FedOSP (ours)"),
]
LABEL_OF = dict(COMPARE_ORDER)

GRADE_NAMES = ["0 None", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]
GRADE_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

# 期刊投稿常用设置：矢量字体可编辑、字号够大
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # TrueType，投稿系统要求字体可嵌入
    "ps.fonttype": 42,
})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    LOGGER.info("✓ %s → %s.{pdf,png}", name, out / name)


def load_runs(root: Path) -> Dict[str, List[dict]]:
    """按 exp_id 分组读 result.json（这里只要 json，不要 npz）。"""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for f in sorted(root.rglob("result.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_dir"] = str(f.parent)
            eid = d.get("args", {}).get("exp_id") or f.parent.name.rsplit("_seed", 1)[0]
            grouped[str(eid)].append(d)
        except Exception as exc:
            LOGGER.warning("跳过 %s：%s", f, exc)
    return grouped


# --------------------------------------------------------------------------- #
def fig_f2(manifest: Path, out: Path) -> bool:
    """F2：client 的等级分布（堆叠占比）+ 样本量（对数轴）。"""
    if not manifest.exists():
        LOGGER.warning("F2 跳过：找不到 %s", manifest)
        return False
    import pandas as pd

    df = pd.read_csv(manifest)
    clients = [c for c in TRAIN_CLIENTS + [HELDOUT] if (df["client"] == c).any()]
    if not clients:
        LOGGER.warning("F2 跳过：manifest 里没有已知 client")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    bottom = np.zeros(len(clients))
    for g in range(5):
        frac = np.array([
            (df[(df["client"] == c) & (df["label"] == g)].shape[0]
             / max(df[df["client"] == c].shape[0], 1)) * 100
            for c in clients
        ])
        ax1.bar(clients, frac, bottom=bottom, label=GRADE_NAMES[g],
                color=GRADE_COLORS[g], edgecolor="white", linewidth=0.5)
        bottom += frac
    ax1.set_ylabel("Grade distribution (%)")
    ax1.set_title("(a) Label shift across clients")
    ax1.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=15)

    sizes = np.array([df[df["client"] == c].shape[0] for c in clients])
    bars = ax2.bar(clients, sizes,
                   color=["#4C72B0" if c != HELDOUT else "#999999" for c in clients])
    ax2.set_yscale("log")
    ax2.set_ylabel("Number of images (log)")
    ratio = sizes[:len(TRAIN_CLIENTS)].max() / max(sizes[:len(TRAIN_CLIENTS)].min(), 1)
    ax2.set_title(f"(b) Client size imbalance ({ratio:.0f}:1)")
    ax2.tick_params(axis="x", rotation=15)
    for b, s in zip(bars, sizes):
        ax2.text(b.get_x() + b.get_width() / 2, s * 1.15, f"{s:,}",
                 ha="center", fontsize=7)

    fig.suptitle("Fig. 2  Non-IID characteristics of the federated DR benchmark", y=1.04)
    save(fig, out, "F2_client_distribution")
    return True


# --------------------------------------------------------------------------- #
def fig_f3(ckpt: Optional[Path], manifest: Path, out: Path, img_size: int = 224) -> bool:
    """F3：FSR 前后的浅层幅度谱（每个 client 一行）。

    这是证明 P1（浅层风格累积）最直接的图：如果 FSR 有效，
    右列各 client 的幅度谱应该比左列更接近彼此。
    """
    if ckpt is None or not Path(ckpt).exists():
        LOGGER.warning("F3 跳过：需要 --ckpt 指向 best.pt（FSR 门控要用训练后的值）")
        return False
    if not manifest.exists():
        LOGGER.warning("F3 跳过：找不到 manifest")
        return False

    import torch

    from fedosp.data.dataset import make_client_loaders
    from fedosp.models.retfound_lora import FedOSPConfig, FedOSPNet

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = FedOSPConfig(**sd["model_cfg"]) if "model_cfg" in sd else FedOSPConfig()
    cfg.pretrained_path = None  # 权重从 checkpoint 来，不重复加载骨干
    model = FedOSPNet(cfg).eval()
    if "shared" in sd:
        model.load_shared_state_dict({k: v for k, v in sd["shared"].items()})

    n = len(TRAIN_CLIENTS)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.1 * n))
    spectra_before, spectra_after = [], []

    for i, client in enumerate(TRAIN_CLIENTS):
        try:
            loaders = make_client_loaders(manifest, client, img_size=img_size,
                                          batch_size=16, num_workers=0)
            batch = next(iter(loaders["val"] or loaders["train"]))
        except Exception as exc:
            LOGGER.warning("F3：client %s 取数据失败 %s", client, exc)
            continue
        x = batch[0][:16]
        with torch.no_grad():
            tok = model.shallow_tokens(x)
            amp_b, amp_a = model.fsr.amplitude_spectrum(tok)

        # 通道平均 + 低频居中，取 log 便于观察
        sb = np.log1p(np.fft.fftshift(amp_b.mean(dim=(0, 1)).numpy()))
        sa = np.log1p(np.fft.fftshift(amp_a.mean(dim=(0, 1)).numpy()))
        spectra_before.append(sb)
        spectra_after.append(sa)

        vmax = max(sb.max(), sa.max())
        for j, (s, ttl) in enumerate(((sb, "before FSR"), (sa, "after FSR"))):
            ax = axes[i, j]
            im = ax.imshow(s, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(ttl)
            if j == 0:
                ax.set_ylabel(client, fontsize=9)
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046)

        ax = axes[i, 2]
        c = sb.shape[0] // 2
        ax.plot(sb[c], label="before", lw=1.2)
        ax.plot(sa[c], label="after", lw=1.2)
        ax.set_xticks([])
        if i == 0:
            ax.set_title("central freq. slice"); ax.legend(fontsize=7)

    # 定量说明：client 间幅度谱的两两距离，FSR 后应显著变小
    if len(spectra_before) >= 2:
        def spread(mats):
            m = np.stack([x.ravel() for x in mats])
            m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
            d = [np.linalg.norm(m[i] - m[j])
                 for i in range(len(m)) for j in range(i + 1, len(m))]
            return float(np.mean(d))

        sb_d, sa_d = spread(spectra_before), spread(spectra_after)
        fig.suptitle(
            "Fig. 3  Shallow amplitude spectra before/after FSR\n"
            f"mean pairwise cross-client distance: {sb_d:.3f} → {sa_d:.3f} "
            f"({(sa_d - sb_d) / max(sb_d, 1e-8) * 100:+.1f}%)",
            y=1.005, fontsize=9,
        )
        LOGGER.info("F3 跨 client 幅度谱距离：FSR 前 %.4f → 后 %.4f（越小越好）", sb_d, sa_d)

    fig.tight_layout()
    save(fig, out, "F3_fsr_spectrum")
    return True


# --------------------------------------------------------------------------- #
def fig_f4(ckpt: Optional[Path], out: Path) -> bool:
    """F4：深层等级原型的二维投影，看它们是否排成有序流形。"""
    if ckpt is None or not Path(ckpt).exists():
        LOGGER.warning("F4 跳过：需要 --ckpt 指向 best.pt")
        return False
    import torch

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    deep = sd.get("deep_proto")
    shallow = sd.get("shallow_proto")
    if deep is None:
        LOGGER.warning("F4 跳过：checkpoint 里没有 deep_proto")
        return False

    panels = [("Deep grade prototypes", deep)]
    if shallow is not None:
        panels.append(("Shallow style prototypes", shallow))

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 3.4))
    axes = np.atleast_1d(axes)

    for ax, (title, proto) in zip(axes, panels):
        p = np.asarray(proto, dtype=float)
        p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
        # 只有 5 个点，PCA 比 t-SNE 稳定得多（t-SNE 在 n=5 时几乎是随机的）
        pc = p - p.mean(0)
        u, s, _ = np.linalg.svd(pc, full_matrices=False)
        xy = u[:, :2] * s[:2]
        ax.plot(xy[:, 0], xy[:, 1], "-", color="#888", lw=1, zorder=1)
        for g in range(len(xy)):
            ax.scatter(xy[g, 0], xy[g, 1], s=180, color=GRADE_COLORS[g],
                       edgecolor="k", linewidth=0.6, zorder=2)
            ax.annotate(str(g), xy[g], ha="center", va="center",
                        fontsize=9, color="white", weight="bold", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # 相邻等级距离应单调、且小于跨等级距离 —— 这是 ordinal margin 起作用的证据
    p = np.asarray(deep, dtype=float)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    dist = np.linalg.norm(p[:, None] - p[None], axis=-1)
    ax = axes[-1]
    im = ax.imshow(dist, cmap="magma")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_title("Pairwise prototype distance")
    ax.set_xlabel("grade"); ax.set_ylabel("grade")
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if dist[i, j] < dist.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    adj = float(np.mean([dist[i, i + 1] for i in range(4)]))
    far = float(np.mean([dist[i, j] for i in range(5) for j in range(5) if abs(i - j) >= 2]))
    fig.suptitle(
        "Fig. 4  Ordinal structure of learned prototypes\n"
        f"mean adjacent-grade distance {adj:.3f} < mean distant-grade distance {far:.3f}"
        f"  →  {'ordinal manifold formed' if adj < far else 'NO ordinal structure (check margin loss)'}",
        y=1.02, fontsize=9,
    )
    LOGGER.info("F4 相邻等级距离 %.4f，跨等级距离 %.4f%s",
                adj, far, "（符合有序结构）" if adj < far else "（**没有形成有序结构，检查 margin 损失**）")
    fig.tight_layout()
    save(fig, out, "F4_prototype_structure")
    return True


# --------------------------------------------------------------------------- #
def fig_f5(runs: Dict[str, List[dict]], out: Path) -> bool:
    """F5：标签效率曲线。"""
    by_method: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rs in runs.values():
        for d in rs:
            a = d.get("args", {})
            m = "FedOSP (ours)" if a.get("strategy") == "fedosp" else str(a.get("strategy", "?"))
            q = d.get("test_summary", {}).get("macro_qwk")
            if q is not None:
                by_method[m][round(float(a.get("label_budget", 1.0)), 2)].append(float(q))

    usable = {m: v for m, v in by_method.items() if len(v) >= 2}
    if not usable:
        LOGGER.warning("F5 跳过：需要至少一个方法在 2 个以上标签预算下有结果")
        return False

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, budgets in sorted(usable.items(), key=lambda kv: kv[0] != "FedOSP (ours)"):
        xs = sorted(budgets)
        ys = [np.mean(budgets[b]) * 100 for b in xs]
        es = [np.std(budgets[b]) * 100 for b in xs]
        ours = m == "FedOSP (ours)"
        ax.errorbar([x * 100 for x in xs], ys, yerr=es, marker="o" if ours else "s",
                    lw=2.2 if ours else 1.3, capsize=3, label=m,
                    color="#C44E52" if ours else None, zorder=3 if ours else 2)
    ax.set_xlabel("Label budget per client (%)")
    ax.set_ylabel("Macro QWK ×100")
    ax.set_title("Fig. 5  Label efficiency")
    ax.legend(fontsize=7)
    save(fig, out, "F5_label_efficiency")
    return True


# --------------------------------------------------------------------------- #
def fig_f6(runs: Dict[str, List[dict]], out: Path) -> bool:
    """F6：收敛曲线（左）+ 达到目标性能所需累计通信量（右）。"""
    # 只画主对比里的方法，不然消融那几十条曲线会把图糊掉
    curves = {}
    for eid, _ in COMPARE_ORDER:
        rs = runs.get(eid)
        h = rs[0].get("history") if rs else None
        if h and len(h) >= 5:
            curves[eid] = h
    if not curves:
        LOGGER.warning("F6 跳过：主对比方法里没有足够长的 history")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for eid, h in curves.items():
        ours = eid == "m_fedosp"
        label = LABEL_OF.get(eid, eid)
        ax1.plot([r["round"] for r in h], [r["macro_qwk"] * 100 for r in h],
                 lw=2.2 if ours else 1.1, label=label,
                 color="#C44E52" if ours else None, zorder=3 if ours else 2)
        ax2.plot([r.get("cum_upload_mb", 0) / 1024 for r in h],
                 [r["macro_qwk"] * 100 for r in h],
                 lw=2.2 if ours else 1.1, label=label,
                 color="#C44E52" if ours else None, zorder=3 if ours else 2)

    ax1.set_xlabel("Communication round"); ax1.set_ylabel("Macro QWK ×100")
    ax1.set_title("(a) Convergence"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Cumulative upload (GB)"); ax2.set_ylabel("Macro QWK ×100")
    ax2.set_title("(b) Performance per communication cost"); ax2.legend(fontsize=7)
    fig.suptitle("Fig. 6  Convergence and communication efficiency", y=1.02)
    save(fig, out, "F6_convergence")
    return True


# --------------------------------------------------------------------------- #
def fig_f7(runs: Dict[str, List[dict]], out: Path) -> bool:
    """F7：per-client QWK 箱线图 —— 证明 P3（小 client 被抬起）的关键图。"""
    present = [e for e, _ in COMPARE_ORDER if e in runs]
    if not present:
        LOGGER.warning("F7 跳过：runs/ 里没有 %s 中的任何一个",
                       [e for e, _ in COMPARE_ORDER])
        return False
    if len(present) == 1:
        LOGGER.warning("F7 只找到 %s 一个方法，对比图意义不大（基线跑完再画）", present[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.6),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    width = 0.8 / len(present)
    xs = np.arange(len(TRAIN_CLIENTS))

    for i, eid in enumerate(present):
        vals, errs = [], []
        for c in TRAIN_CLIENTS:
            v = [d.get("test_per_client", {}).get(c, {}).get("qwk") for d in runs[eid]]
            v = [x * 100 for x in v if x is not None]
            vals.append(np.mean(v) if v else np.nan)
            errs.append(np.std(v) if len(v) > 1 else 0.0)
        ours = eid == "m_fedosp"
        ax1.bar(xs + i * width - 0.4 + width / 2, vals, width, yerr=errs, capsize=2,
                label=LABEL_OF.get(eid, eid),
                color="#C44E52" if ours else None,
                edgecolor="black" if ours else "none", linewidth=0.8)

    ax1.set_xticks(xs)
    ax1.set_xticklabels([DISPLAY_NAME[c] for c in TRAIN_CLIENTS])
    ax1.set_ylabel("QWK ×100")
    ax1.set_title("(a) Per-client performance (error bar = std over seeds)")
    ax1.legend(fontsize=7, ncol=2)

    # 右图：worst-client 才是本文关心的公平性指标
    labels, worst, macro = [], [], []
    for eid in present:
        labels.append(LABEL_OF.get(eid, eid).replace(" (ours)", ""))
        worst.append(np.mean([d.get("test_summary", {}).get("worst_qwk", np.nan) * 100
                              for d in runs[eid]]))
        macro.append(np.mean([d.get("test_summary", {}).get("macro_qwk", np.nan) * 100
                              for d in runs[eid]]))
    x2 = np.arange(len(labels))
    ax2.bar(x2 - 0.2, worst, 0.4, label="Worst client")
    ax2.bar(x2 + 0.2, macro, 0.4, label="Macro mean")
    ax2.set_xticks(x2); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("QWK ×100")
    ax2.set_title("(b) Worst-client vs macro")
    ax2.legend(fontsize=7)

    fig.suptitle("Fig. 7  Fairness across clients: lifting the worst-performing center", y=1.03)
    save(fig, out, "F7_per_client")
    return True


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="FedOSP 的 best.pt，F3/F4 需要")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--figs", nargs="*", default=None, help="如 --figs F3 F7")
    args = ap.parse_args()

    runs = load_runs(args.runs) if args.runs.exists() else {}
    if not runs:
        LOGGER.warning("%s 下没有 result.json，只能画 F2/F3/F4", args.runs)

    # 没显式给 ckpt 就自己找一个
    ckpt = args.ckpt
    if ckpt is None:
        cands = sorted(args.runs.glob("m_fedosp*/best.pt")) if args.runs.exists() else []
        ckpt = cands[0] if cands else None
        if ckpt:
            LOGGER.info("自动选用 checkpoint %s", ckpt)

    builders = {
        "F2": lambda: fig_f2(args.manifest, args.out),
        "F3": lambda: fig_f3(ckpt, args.manifest, args.out, args.img_size),
        "F4": lambda: fig_f4(ckpt, args.out),
        "F5": lambda: fig_f5(runs, args.out),
        "F6": lambda: fig_f6(runs, args.out),
        "F7": lambda: fig_f7(runs, args.out),
    }
    wanted = args.figs or list(builders)

    done, skipped = [], []
    for key in wanted:
        if key not in builders:
            LOGGER.warning("不认识的图名 %s，可选 %s", key, list(builders))
            continue
        try:
            (done if builders[key]() else skipped).append(key)
        except Exception as exc:
            LOGGER.error("生成 %s 失败：%s", key, exc, exc_info=True)
            skipped.append(key)

    LOGGER.info("完成 %s%s", done or "无", f"；跳过 {skipped}" if skipped else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
