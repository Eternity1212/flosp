#!/usr/bin/env python3
"""把 runs/ 下所有实验结果汇总成论文主表 T1–T6。

跑完 49 个配置后，手工拼表既慢又容易错（尤其是 3 个 seed 要算 mean±std）。
这个脚本扫描全部 ``result.json``，按实验 ID 分组聚合，输出三种格式：

* ``tables/*.md``    —— 边做边看
* ``tables/*.csv``   —— 进 Excel 或再加工
* ``tables/*.tex``   —— 直接贴进论文（booktabs 风格）

生成的表：

====  =========================================================================
表     内容
====  =========================================================================
T1    数据集与 client 概览（从 manifest 直接统计，不依赖实验结果）
T2    **主结果**：8 个基线 + FedOSP 的 per-client QWK、worst、macro、外测
T3    消融：A1–A12 逐个组件的贡献
T4    标签效率：10% / 25% / 50% / 100% 标签下的 macro QWK
T5    系统开销：可训练参数、单轮上传、累计通信、墙钟时间
T6    统计检验：FedOSP vs 每个基线，样本级配对 bootstrap + Holm 校正
====  =========================================================================

用法::

    python scripts/aggregate_results.py --runs runs --out tables
    python scripts/aggregate_results.py --runs runs --tables T2 T6   # 只出指定表
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedosp.metrics import quadratic_weighted_kappa  # noqa: E402
from fedosp.stats import (  # noqa: E402
    TestResult,
    delong_test,
    holm_bonferroni,
    paired_bootstrap,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
LOGGER = logging.getLogger("aggregate")

TRAIN_CLIENTS = ["eyepacs", "aptos", "ddr", "idrid"]
HELDOUT = "messidor2"

#: 表头用的正式名称（``.capitalize()`` 会把 EyePACS 写成 Eyepacs、DDR 写成 Ddr）
DISPLAY_NAME = {
    "eyepacs": "EyePACS",
    "aptos": "APTOS",
    "ddr": "DDR",
    "idrid": "IDRiD",
    "messidor2": "Messidor-2",
}


def fmt_p(p: Optional[float]) -> str:
    """p 值的期刊惯用写法：极小值写 ``<0.001`` 而不是 ``0``。"""
    if p is None:
        return "–"
    if p < 1e-4:
        return "<0.0001"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"

#: 基线的展示名与顺序，决定主表的行顺序
METHOD_ORDER = [
    ("base_local", "Local-only (no FL)"),
    ("base_pooled", "Pooled Oracle (upper bd.)"),
    ("base_fedavg", "FedAvg"),
    ("base_fedprox", "FedProx"),
    ("base_fedbn", "FedBN"),
    ("base_scaffold", "SCAFFOLD"),
    ("base_fedper", "FedPer"),
    ("base_fedproto", "FedProto"),
    ("base_feduaa", "FedUAA-style*"),
    ("base_full", "FedAvg + full fine-tune"),
    ("base_vpt", "FedAvg + visual prompt tuning"),
    ("m_fedosp", "FedOSP (ours)"),
]

ABLATION_ORDER = [
    ("m_fedosp", "FedOSP (full)"),
    ("abl_nofsr", "  w/o FSR"),
    ("abl_noshallow", "  w/o shallow style prototype"),
    ("abl_nodeep", "  w/o deep grade prototype"),
    ("abl_noord", "  w/o EMD ordinal loss"),
    ("abl_nomargin", "  w/o ordinal margin"),
    ("abl_proto_sample", "  proto agg: sample-weighted"),
    ("abl_proto_sqrt", "  proto agg: sqrt-weighted"),
    ("abl_globalln", "  w/o local LayerNorm"),
    ("abl_equalsteps", "  w/o local-step equalization"),
    ("abl_fsr4", "  FSR after block 4"),
    ("abl_fsr12", "  FSR after block 12"),
    ("abl_rank4", "  LoRA r=4"),
    ("abl_rank16", "  LoRA r=16"),
    ("abl_rank32", "  LoRA r=32"),
    ("bb_swinv2", "  backbone: SwinV2-B"),
    ("bb_resnet50", "  backbone: ResNet-50"),
    ("bb_vit384", "  input 384x384"),
]


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #
class Run:
    """一次实验的结果，外加惰性加载的 per-sample 预测。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.dir = path.parent
        self.data = json.loads(path.read_text(encoding="utf-8"))
        args = self.data.get("args", {})
        self.exp_id = str(args.get("exp_id") or self.dir.name.rsplit("_seed", 1)[0])
        self.seed = int(args.get("seed", 0))
        self.strategy = str(args.get("strategy", "?"))
        self.label_budget = float(args.get("label_budget", 1.0))
        self.backbone = str(self.data.get("model_cfg", {}).get("backbone", "?"))
        self._preds: Optional[Dict[str, np.ndarray]] = None

    # ---- 指标读取 ---- #
    def client_metric(self, client: str, key: str = "qwk") -> Optional[float]:
        v = self.data.get("test_per_client", {}).get(client, {}).get(key)
        return float(v) if v is not None else None

    def summary(self, key: str) -> Optional[float]:
        v = self.data.get("test_summary", {}).get(key)
        return float(v) if v is not None else None

    def external(self, key: str = "qwk") -> Optional[float]:
        v = self.data.get("external", {}).get(key)
        return float(v) if v is not None else None

    def system(self, key: str) -> Optional[float]:
        v = self.data.get("system", {}).get(key)
        return float(v) if v is not None else None

    @property
    def preds(self) -> Dict[str, np.ndarray]:
        """惰性加载 ``predictions.npz``；不存在则返回空 dict。"""
        if self._preds is None:
            f = self.dir / "predictions.npz"
            self._preds = dict(np.load(f)) if f.exists() else {}
        return self._preds


def load_runs(root: Path) -> Dict[str, List[Run]]:
    """扫描 ``root`` 下所有 result.json，按 exp_id 分组。"""
    runs: Dict[str, List[Run]] = defaultdict(list)
    files = sorted(root.rglob("result.json"))
    for f in files:
        try:
            r = Run(f)
            runs[r.exp_id].append(r)
        except Exception as exc:  # 单个坏文件不该毁掉整次汇总
            LOGGER.warning("跳过 %s：%s", f, exc)
    LOGGER.info("扫到 %d 个 result.json，归成 %d 个实验配置", len(files), len(runs))
    for eid, rs in sorted(runs.items()):
        if len(rs) < 3:
            LOGGER.warning("  %s 只有 %d 个 seed（预期 3），mean±std 会不稳", eid, len(rs))
    return runs


# --------------------------------------------------------------------------- #
# 格式化
# --------------------------------------------------------------------------- #
def mean_std(values: Sequence[Optional[float]], scale: float = 100.0, nd: int = 2) -> str:
    """``mean±std`` 字符串。默认把 0–1 的指标乘 100 显示成百分点。"""
    v = [x for x in values if x is not None and not np.isnan(x)]
    if not v:
        return "–"
    m = np.mean(v) * scale
    if len(v) == 1:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}±{np.std(v) * scale:.{nd}f}"


def mean_of(values: Sequence[Optional[float]]) -> float:
    v = [x for x in values if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


class Table:
    """一张表：负责 markdown / csv / latex 三种输出。"""

    def __init__(self, key: str, title: str, header: Sequence[str], note: str = "") -> None:
        self.key = key
        self.title = title
        self.header = list(header)
        self.rows: List[List[str]] = []
        self.note = note
        #: 需要加粗的行下标（通常是我方方法）
        self.bold_rows: set = set()

    def add(self, row: Sequence, bold: bool = False) -> None:
        if bold:
            self.bold_rows.add(len(self.rows))
        self.rows.append([str(x) for x in row])

    def to_markdown(self) -> str:
        out = [f"### {self.key}　{self.title}", ""]
        out.append("| " + " | ".join(self.header) + " |")
        out.append("|" + "|".join(["---"] * len(self.header)) + "|")
        for i, r in enumerate(self.rows):
            cells = [f"**{c}**" for c in r] if i in self.bold_rows else r
            out.append("| " + " | ".join(cells) + " |")
        if self.note:
            out += ["", f"> {self.note}"]
        return "\n".join(out) + "\n"

    def to_csv(self) -> str:
        import csv
        import io

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.header)
        w.writerows(self.rows)
        return buf.getvalue()

    def to_latex(self) -> str:
        cols = "l" + "c" * (len(self.header) - 1)
        esc = lambda s: (  # noqa: E731
            s.replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_")
             .replace("*", r"$^*$").replace("–", "--")
        )
        out = [
            r"\begin{table}[t]", r"\centering",
            f"\\caption{{{self.title}}}", f"\\label{{tab:{self.key.lower()}}}",
            r"\small", f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
            " & ".join(esc(h) for h in self.header) + r" \\", r"\midrule",
        ]
        for i, r in enumerate(self.rows):
            cells = [rf"\textbf{{{esc(c)}}}" for c in r] if i in self.bold_rows else [esc(c) for c in r]
            out.append(" & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
        if self.note:
            out.append(rf"\\[2pt] \footnotesize {esc(self.note)}")
        out.append(r"\end{table}")
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# T1：数据概览
# --------------------------------------------------------------------------- #
def table_t1(manifest_path: Optional[Path]) -> Optional[Table]:
    if not manifest_path or not manifest_path.exists():
        LOGGER.info("跳过 T1：没有 manifest（--manifest 指一下就能出）")
        return None
    import pandas as pd

    df = pd.read_csv(manifest_path)
    t = Table(
        "T1", "Datasets and client partition",
        ["Client", "Role", "Images", "Train/Val/Test", "Grade 0–4 (%)", "Split rule"],
        note="Messidor-2 全程不参与训练，仅作未见中心外测。EyePACS 按患者划分（左右眼同侧）。",
    )
    rules = {
        "eyepacs": "patient-level 70/10/20",
        "aptos": "stratified 70/10/20",
        "ddr": "official + ungradable removed",
        "idrid": "official train/test",
        HELDOUT: "held-out (external only)",
    }
    for name in TRAIN_CLIENTS + [HELDOUT]:
        sub = df[df["client"] == name]
        if sub.empty:
            continue
        n = len(sub)
        splits = "/".join(str(int((sub["split"] == s).sum())) for s in ("train", "val", "test"))
        dist = np.bincount(sub["label"].astype(int), minlength=5) / n * 100
        t.add([
            name, "external" if name == HELDOUT else "train", f"{n:,}", splits,
            " / ".join(f"{d:.1f}" for d in dist), rules.get(name, "–"),
        ])
    return t


# --------------------------------------------------------------------------- #
# T2：主结果
# --------------------------------------------------------------------------- #
def table_t2(runs: Dict[str, List[Run]]) -> Table:
    t = Table(
        "T2", "Main results: per-client QWK on multi-center DR grading",
        ["Method"] + [DISPLAY_NAME[c] for c in TRAIN_CLIENTS]
        + ["Worst", "Macro", "Messidor-2 (unseen)"],
        note=("QWK x100, mean±std over 3 seeds. Macro = unweighted mean over clients "
              "(NOT sample-weighted: EyePACS would dominate). "
              "*FedUAA-style is our re-implementation under the same RETFound-LoRA backbone, "
              "not the original paper's numbers."),
    )
    for eid, label in METHOD_ORDER:
        rs = runs.get(eid)
        if not rs:
            continue
        row = [label]
        row += [mean_std([r.client_metric(c) for r in rs]) for c in TRAIN_CLIENTS]
        row += [
            mean_std([r.summary("worst_qwk") for r in rs]),
            mean_std([r.summary("macro_qwk") for r in rs]),
            mean_std([r.external("qwk") for r in rs]),
        ]
        t.add(row, bold=(eid == "m_fedosp"))
    if not t.rows:
        LOGGER.warning("T2 一行都没有：检查 result.json 里的 args.exp_id 是否与 METHOD_ORDER 对得上")
    return t


# --------------------------------------------------------------------------- #
# T3：消融
# --------------------------------------------------------------------------- #
def table_t3(runs: Dict[str, List[Run]]) -> Table:
    t = Table(
        "T3", "Ablation study",
        ["Variant", "Worst QWK", "Macro QWK", "Messidor-2", "ΔMacro"],
        note="ΔMacro 相对完整 FedOSP。负值说明去掉该组件后性能下降，即该组件有贡献。",
    )
    full = runs.get("m_fedosp")
    base_macro = mean_of([r.summary("macro_qwk") for r in full]) if full else float("nan")

    for eid, label in ABLATION_ORDER:
        rs = runs.get(eid)
        if not rs:
            continue
        macro = mean_of([r.summary("macro_qwk") for r in rs])
        delta = (macro - base_macro) * 100
        t.add([
            label,
            mean_std([r.summary("worst_qwk") for r in rs]),
            mean_std([r.summary("macro_qwk") for r in rs]),
            mean_std([r.external("qwk") for r in rs]),
            "–" if eid == "m_fedosp" else f"{delta:+.2f}",
        ], bold=(eid == "m_fedosp"))
    return t


# --------------------------------------------------------------------------- #
# T4：标签效率
# --------------------------------------------------------------------------- #
def table_t4(runs: Dict[str, List[Run]]) -> Table:
    t = Table(
        "T4", "Label efficiency: macro QWK under limited annotation",
        ["Method", "10%", "25%", "50%", "100%"],
        note="每个 client 内部按类别分层子采样，保证稀有等级不被抽空。",
    )
    budgets = [0.10, 0.25, 0.50, 1.00]
    by_method: Dict[str, Dict[float, List[Run]]] = defaultdict(lambda: defaultdict(list))
    for rs in runs.values():
        for r in rs:
            key = "FedOSP (ours)" if r.strategy == "fedosp" else r.strategy
            by_method[key][round(r.label_budget, 2)].append(r)

    for method in sorted(by_method, key=lambda m: (m != "FedOSP (ours)", m)):
        row = [method]
        found = False
        for b in budgets:
            rs = by_method[method].get(b, [])
            row.append(mean_std([r.summary("macro_qwk") for r in rs]) if rs else "–")
            found = found or bool(rs)
        if found:
            t.add(row, bold=(method == "FedOSP (ours)"))
    return t


# --------------------------------------------------------------------------- #
# T5：系统开销
# --------------------------------------------------------------------------- #
def table_t5(runs: Dict[str, List[Run]]) -> Table:
    t = Table(
        "T5", "Communication and computation cost",
        ["Method", "Trainable params", "Upload / client / round", "Rounds",
         "Total upload", "Wall clock"],
        note=("Upload 按 fp32 计。SCAFFOLD 因为要额外传 control variate，通信量约为 FedAvg 两倍。"),
    )
    for eid, label in METHOD_ORDER:
        rs = runs.get(eid)
        if not rs:
            continue
        cfg = rs[0].data.get("model_cfg", {})
        n_tr = rs[0].data.get("system", {}).get("trainable_M")
        t.add([
            label,
            f"{n_tr:.2f} M" if n_tr else ("303.9 M (full)" if cfg.get("full_finetune") else "0.70 M"),
            f"{mean_of([r.system('upload_mb_per_client_per_round') for r in rs]):.2f} MB",
            f"{mean_of([r.system('rounds_run') for r in rs]):.0f}",
            f"{mean_of([r.system('cum_upload_mb') for r in rs]) / 1024:.2f} GB",
            f"{mean_of([r.system('wall_clock_s') for r in rs]) / 3600:.2f} h",
        ], bold=(eid == "m_fedosp"))
    return t


# --------------------------------------------------------------------------- #
# T6：统计检验
# --------------------------------------------------------------------------- #
def table_t6(runs: Dict[str, List[Run]]) -> Table:
    """样本级配对检验。

    **为什么不用跨 client 的 Wilcoxon 当主检验**：只有 4 个 client，
    双侧 Wilcoxon 的 p 值下界是 0.125，再经 Holm 校正后不可能显著。
    所以这里对**每个 client 的测试样本**分别做配对 bootstrap（n 是几百到几千），
    再用 Holm 校正全部比较。跨 client 的方向一致性单独一列报。
    """
    t = Table(
        "T6", "Statistical significance: FedOSP vs baselines",
        ["Comparison", "Client", "ΔQWK", "95% CI", "p", "p (Holm)", "Sig.",
         "ΔAUROC", "p (DeLong)"],
        note=("配对 bootstrap（2000 次）比 QWK，DeLong 比 referable AUROC，"
              "两者都在同一批测试样本上配对。Holm-Bonferroni 校正整个 family。"
              "客户端级 Wilcoxon 因 n=4（p 下界 0.125）不作为主检验，见正文。"),
    )
    ours = runs.get("m_fedosp")
    if not ours or not ours[0].preds:
        LOGGER.warning("T6 跳过：缺 FedOSP 的 predictions.npz（旧版 run_fed 不会保存，需重跑）")
        return t

    ref = ours[0]  # 用 seed 0 的预测做配对检验
    results: List[Tuple[str, str, TestResult, Optional[TestResult]]] = []

    for eid, label in METHOD_ORDER:
        if eid == "m_fedosp":
            continue
        rs = runs.get(eid)
        if not rs or not rs[0].preds:
            continue
        base = rs[0]
        for client in TRAIN_CLIENTS + [HELDOUT]:
            ky, kp = f"{client}__y", f"{client}__probs"
            if ky not in ref.preds or ky not in base.preds:
                continue
            y = ref.preds[ky].astype(int)
            if len(y) != len(base.preds[ky]) or not np.array_equal(y, base.preds[ky].astype(int)):
                LOGGER.warning("%s / %s 上 %s 的测试样本顺序不一致，跳过配对检验", eid, client, client)
                continue

            pa, pb = ref.preds[kp], base.preds[kp]
            qwk_res = paired_bootstrap(
                quadratic_weighted_kappa, y, pa.argmax(1), pb.argmax(1),
                n_boot=2000, name=f"FedOSP vs {label} @{client}",
            )
            # referable DR = grade >= 2，分数取 P(grade>=2)
            try:
                auc_res = delong_test(
                    (y >= 2).astype(int), pa[:, 2:].sum(1), pb[:, 2:].sum(1),
                    name=f"AUROC @{client}",
                )
            except ValueError:
                auc_res = None
            results.append((label, client, qwk_res, auc_res))

    holm_bonferroni([r[2] for r in results])
    for label, client, q, a in results:
        t.add([
            f"vs {label}", DISPLAY_NAME.get(client, client),
            f"{q.effect * 100:+.2f}",
            f"[{q.ci_low * 100:+.2f}, {q.ci_high * 100:+.2f}]" if q.ci_low is not None else "–",
            fmt_p(q.p_value),
            fmt_p(q.p_corrected),
            "*" if q.significant else "",
            f"{a.effect * 100:+.2f}" if a else "–",
            fmt_p(a.p_value) if a else "–",
        ])
    return t


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="实验输出根目录")
    ap.add_argument("--out", type=Path, default=Path("tables"), help="表格输出目录")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"),
                    help="用于生成 T1")
    ap.add_argument("--tables", nargs="*", default=None,
                    help="只生成指定表，如 --tables T2 T6；默认全部")
    args = ap.parse_args()

    if not args.runs.exists():
        LOGGER.error("找不到 %s。先跑实验：bash scripts/run_all.sh", args.runs)
        return 1

    runs = load_runs(args.runs)
    if not runs:
        LOGGER.error("%s 下没有任何 result.json", args.runs)
        return 1

    builders = {
        "T1": lambda: table_t1(args.manifest),
        "T2": lambda: table_t2(runs),
        "T3": lambda: table_t3(runs),
        "T4": lambda: table_t4(runs),
        "T5": lambda: table_t5(runs),
        "T6": lambda: table_t6(runs),
    }
    wanted = args.tables or list(builders)

    args.out.mkdir(parents=True, exist_ok=True)
    combined = ["# FedOSP 结果汇总", "",
                f"由 `scripts/aggregate_results.py` 自动生成，来源 `{args.runs}/`。", ""]

    for key in wanted:
        if key not in builders:
            LOGGER.warning("不认识的表名 %s，可选 %s", key, list(builders))
            continue
        try:
            table = builders[key]()
        except Exception as exc:
            LOGGER.error("生成 %s 失败：%s", key, exc, exc_info=True)
            continue
        if table is None or not table.rows:
            LOGGER.info("%s 无数据，跳过", key)
            continue
        (args.out / f"{key.lower()}.md").write_text(table.to_markdown(), encoding="utf-8")
        (args.out / f"{key.lower()}.csv").write_text(table.to_csv(), encoding="utf-8")
        (args.out / f"{key.lower()}.tex").write_text(table.to_latex(), encoding="utf-8")
        combined.append(table.to_markdown())
        LOGGER.info("✓ %s（%d 行）→ %s", key, len(table.rows), args.out / f"{key.lower()}.*")

    (args.out / "all_tables.md").write_text("\n".join(combined), encoding="utf-8")
    LOGGER.info("全部表格已写入 %s/，总览见 all_tables.md", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
