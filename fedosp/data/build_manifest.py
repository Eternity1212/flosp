"""统一 manifest 构建：把五个 DR 数据集收敛成一张 CSV。

设计原则（对应 `09_方向二_落地实施方案.md` 第 3 节）：

1. **所有实验只允许从这一张 CSV 读数据**，杜绝「不同方法用了不同划分」。
2. EyePACS 必须按 patient 划分（`10_left.jpeg` / `10_right.jpeg` 是同一个人）。
3. DDR 必须丢掉 ungradable（原始标签 5），并沿用官方 train/valid/test。
4. IDRiD 必须沿用官方 413/103，val 从 413 内部抽。
5. Messidor-2 只能是 test，且标签必须来自 Google Brain 的裁定标签。

输出列：``image_id, client, split, dr_grade, patient_id, source_split, path``

用法::

    python -m fedosp.data.build_manifest \
        --data-root /path/to/raw \
        --out /path/to/manifest.csv

每个数据集都可以单独跳过（缺哪个就先不建哪个），最后用 ``--verify-only`` 对已有
manifest 重新跑一遍全部健全性检查。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("build_manifest")

MANIFEST_COLUMNS = [
    "image_id",
    "client",
    "split",
    "dr_grade",
    "patient_id",
    "source_split",
    "path",
]

#: 训练 client（4 个）+ 未见中心（1 个，只测不训）
TRAIN_CLIENTS = ("eyepacs", "aptos", "ddr", "idrid")
HELDOUT_CLIENT = "messidor2"

#: 文献记载的官方规模，用来在构建后做交叉核对（见方案 3.1 / 3.2）
EXPECTED_TOTALS: Dict[str, int] = {
    "eyepacs": 35126,
    "aptos": 3662,
    "ddr": 12522,  # 剔除 1151 张 ungradable 之后
    "idrid": 516,
    "messidor2": 1744,  # 1748 张里剔除 4 张不可分级
}

#: 文献记载的等级分布，用来核对标签是否读错列
EXPECTED_GRADE_DIST: Dict[str, List[int]] = {
    "eyepacs": [25810, 2443, 5292, 873, 708],
    "aptos": [1805, 370, 999, 193, 295],
    "ddr": [6266, 630, 4477, 236, 913],
    "idrid": [168, 25, 168, 93, 62],  # 413 训练 + 103 测试 合并
    "messidor2": [1017, 270, 347, 75, 35],
}


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def _first_existing(root: Path, candidates: List[str]) -> Optional[Path]:
    """在若干候选相对路径里返回第一个真实存在的，找不到返回 None。

    不同来源（Kaggle / GitHub / 官网）解压出来的目录名不一致，这里做一次兜底。
    """
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    # 兜底：递归找同名文件（数据集目录不大，一次性开销可接受）
    for rel in candidates:
        name = Path(rel).name
        hits = sorted(root.rglob(name))
        if hits:
            LOGGER.debug("fallback rglob 命中 %s -> %s", rel, hits[0])
            return hits[0]
    return None


def stratified_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float],
    seed: int,
    label_col: str = "dr_grade",
) -> pd.Series:
    """按标签分层的 image 级划分，返回与 df 同索引的 split 序列。"""
    rng = np.random.RandomState(seed)
    split = pd.Series(index=df.index, dtype=object)
    r_train, r_val, _ = ratios
    for grade, idx in df.groupby(label_col).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * r_train))
        n_va = int(round(n * r_val))
        # 极小类兜底：至少保证 train 里有 1 张
        n_tr = max(n_tr, 1) if n >= 1 else 0
        split.loc[idx[:n_tr]] = "train"
        split.loc[idx[n_tr : n_tr + n_va]] = "val"
        split.loc[idx[n_tr + n_va :]] = "test"
        LOGGER.debug("grade %s: n=%d -> train %d / val %d / test %d",
                     grade, n, n_tr, n_va, n - n_tr - n_va)
    return split


def stratified_group_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float],
    seed: int,
    group_col: str = "patient_id",
    label_col: str = "dr_grade",
) -> pd.Series:
    """**患者级**分层划分：同一 patient 的全部图像强制进同一个 split。

    做法：先把每个 patient 折叠成一条记录（标签取该患者的最高等级，这是眼科
    里常用的 per-patient 严重度定义），再对 patient 做分层划分，最后广播回图像。

    EyePACS 必须用这个函数，否则左右眼分到不同 split 会造成严重泄漏。
    """
    per_patient = (
        df.groupby(group_col)[label_col].max().rename("patient_grade").reset_index()
    )
    per_patient = per_patient.set_index(group_col)
    assign = stratified_split(
        per_patient.rename(columns={"patient_grade": label_col}),
        ratios,
        seed,
        label_col=label_col,
    )
    LOGGER.info(
        "患者级划分：%d 个 patient -> %s",
        len(assign),
        assign.value_counts().to_dict(),
    )
    return df[group_col].map(assign)


def _finalize(
    frame: pd.DataFrame, client: str, expect_paths: bool = True
) -> pd.DataFrame:
    """统一列顺序、类型，并做单数据集级别的基本校验。"""
    frame = frame.copy()
    frame["client"] = client
    frame["dr_grade"] = frame["dr_grade"].astype(int)
    frame["image_id"] = frame["image_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    missing = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"[{client}] manifest 缺列: {missing}")
    frame = frame[MANIFEST_COLUMNS]

    bad = frame[~frame["dr_grade"].between(0, 4)]
    if len(bad):
        raise ValueError(f"[{client}] 存在 {len(bad)} 条 grade 不在 0-4 的记录")
    if expect_paths:
        n_missing = sum(1 for p in frame["path"] if not Path(p).exists())
        if n_missing:
            LOGGER.warning("[%s] 有 %d/%d 个图像路径不存在（先建表，落盘后再核）",
                           client, n_missing, len(frame))
    LOGGER.info(
        "[%s] 共 %d 张 | split %s | grade %s",
        client,
        len(frame),
        frame["split"].value_counts().to_dict(),
        frame["dr_grade"].value_counts().sort_index().to_dict(),
    )
    return frame


# --------------------------------------------------------------------------- #
# 五个数据集各自的构建函数
# --------------------------------------------------------------------------- #
def build_eyepacs(root: Path, seed: int) -> pd.DataFrame:
    """EyePACS / Kaggle DR 2015。

    期望布局::

        <root>/eyepacs/trainLabels.csv        # 列: image, level
        <root>/eyepacs/train/<image>.jpeg

    ``patient_id`` 取文件名下划线前的数字，划分走 :func:`stratified_group_split`。
    """
    base = root / "eyepacs"
    label_csv = _first_existing(
        base, ["trainLabels.csv", "trainLabels.csv.zip", "labels/trainLabels.csv"]
    )
    if label_csv is None:
        raise FileNotFoundError(f"找不到 EyePACS 标签文件，看过 {base}")
    img_dir = _first_existing(base, ["train", "train_images", "images"]) or base / "train"

    df = pd.read_csv(label_csv)
    df = df.rename(columns={"image": "image_id", "level": "dr_grade"})
    df["patient_id"] = df["image_id"].str.split("_").str[0]
    df["source_split"] = "official_train"
    df["path"] = df["image_id"].map(lambda x: str(img_dir / f"{x}.jpeg"))
    df["split"] = stratified_group_split(df, (0.7, 0.1, 0.2), seed)
    return _finalize(df, "eyepacs")


def build_aptos(root: Path, seed: int) -> pd.DataFrame:
    """APTOS 2019。

    期望布局::

        <root>/aptos2019/train.csv            # 列: id_code, diagnosis
        <root>/aptos2019/train_images/<id_code>.png

    官方 test 的 1928 张没有公开标签，**不要用**，这里只读 train.csv。
    无 patient ID，``patient_id`` 直接等于 ``image_id``（一图一"人"）。
    """
    base = _first_existing(root, ["aptos2019", "aptos", "aptos2019-blindness-detection"])
    if base is None:
        raise FileNotFoundError(f"找不到 APTOS 目录，看过 {root}")
    label_csv = _first_existing(base, ["train.csv", "train_1.csv"])
    if label_csv is None:
        raise FileNotFoundError(f"找不到 APTOS train.csv，看过 {base}")
    img_dir = _first_existing(base, ["train_images", "train"]) or base / "train_images"

    df = pd.read_csv(label_csv)
    df = df.rename(columns={"id_code": "image_id", "diagnosis": "dr_grade"})
    df["patient_id"] = df["image_id"]
    df["source_split"] = "official_train"
    df["path"] = df["image_id"].map(lambda x: str(img_dir / f"{x}.png"))
    df["split"] = stratified_split(df, (0.7, 0.1, 0.2), seed)
    return _finalize(df, "aptos")


def build_ddr(root: Path, seed: int) -> pd.DataFrame:
    """DDR（中国 147 家医院）。

    期望布局::

        <root>/DDR-dataset/DR_grading/train.txt   # 每行 "007-0004-000.jpg 0"
        <root>/DDR-dataset/DR_grading/valid.txt
        <root>/DDR-dataset/DR_grading/test.txt
        <root>/DDR-dataset/DR_grading/{train,valid,test}/<img>

    两个必须做的处理：

    * 用**官方划分**（6835 / 2733 / 4105），不要自己重切；
    * 在每个 split 内部各自剔除 ``label == 5``（ungradable，共 1151 张）。
    """
    base = _first_existing(root, ["DDR-dataset/DR_grading", "ddr/DR_grading", "DDR/DR_grading"])
    if base is None:
        raise FileNotFoundError(f"找不到 DDR DR_grading 目录，看过 {root}")

    frames = []
    n_ungradable = 0
    for src_split, split in [("train", "train"), ("valid", "val"), ("test", "test")]:
        txt = _first_existing(base, [f"{src_split}.txt"])
        if txt is None:
            raise FileNotFoundError(f"找不到 DDR {src_split}.txt，看过 {base}")
        part = pd.read_csv(txt, sep=r"\s+", header=None, names=["image_id", "dr_grade"])
        before = len(part)
        part = part[part["dr_grade"] != 5]  # ← 剔除 ungradable
        n_ungradable += before - len(part)
        img_dir = _first_existing(base, [src_split]) or base / src_split
        part["path"] = part["image_id"].map(lambda x: str(img_dir / x))
        part["source_split"] = f"official_{src_split}"
        part["split"] = split
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df["patient_id"] = df["image_id"]  # DDR 无 patient ID
    LOGGER.info("[ddr] 已剔除 %d 张 ungradable（文献值 1151）", n_ungradable)
    if n_ungradable != 1151:
        LOGGER.warning("[ddr] ungradable 数量 %d != 文献 1151，确认标签文件版本", n_ungradable)
    return _finalize(df, "ddr")


def build_idrid(root: Path, seed: int) -> pd.DataFrame:
    """IDRiD（印度单中心）。

    期望布局（官网 zip 解压后的原始中文/空格目录名，这里用 rglob 兜底）::

        .../a. IDRiD_Disease Grading_Training Labels.csv   # 列: Image name, Retinopathy grade
        .../b. IDRiD_Disease Grading_Testing Labels.csv
        .../1. Original Images/a. Training Set/<name>.jpg
        .../1. Original Images/b. Testing Set/<name>.jpg

    用**官方 413/103 划分**，再从 413 里分层抽约 10% 当 val → 372/41/103。
    """
    base = _first_existing(root, ["idrid", "IDRiD", "B. Disease Grading"]) or root

    def _load(label_name: str, img_hint: str, src: str) -> pd.DataFrame:
        csv_path = _first_existing(base, [label_name])
        if csv_path is None:
            raise FileNotFoundError(f"找不到 IDRiD 标签 {label_name}，看过 {base}")
        part = pd.read_csv(csv_path)
        part.columns = [c.strip() for c in part.columns]
        part = part.rename(
            columns={"Image name": "image_id", "Retinopathy grade": "dr_grade"}
        )[["image_id", "dr_grade"]]
        part = part.dropna(subset=["image_id"])
        img_dirs = [p for p in base.rglob("*") if p.is_dir() and img_hint in p.name]
        img_dir = img_dirs[0] if img_dirs else base
        part["path"] = part["image_id"].map(lambda x: str(img_dir / f"{x}.jpg"))
        part["source_split"] = src
        return part

    train = _load(
        "a. IDRiD_Disease Grading_Training Labels.csv", "Training Set", "official_train"
    )
    test = _load(
        "b. IDRiD_Disease Grading_Testing Labels.csv", "Testing Set", "official_test"
    )

    # 从官方 413 训练图里分层抽 ~10% 当 val（0.9/0.1/0.0）
    train["split"] = stratified_split(train, (0.9, 0.1, 0.0), seed).replace(
        {"test": "val"}
    )
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    df["patient_id"] = df["image_id"]  # IDRiD 无 patient ID
    return _finalize(df, "idrid")


def build_messidor2(root: Path, seed: int) -> pd.DataFrame:
    """Messidor-2（未见中心，只测不训）。

    **DR 标签不在原始发布里**，必须用 Google Brain 三位视网膜专科医师的裁定标签
    （Kaggle ``google-brain/messidor2-dr-grades``，Krause et al., Ophthalmology 2018）。

    期望布局::

        <root>/messidor2/messidor_data.csv    # 列: image_id, adjudicated_dr_grade,
                                              #      adjudicated_dme, adjudicated_gradable
        <root>/messidor2/IMAGES/<image_id>

    1748 张里剔除 ``adjudicated_gradable == 0`` 的 4 张 → 1744 张，全部 split=test。
    """
    base = _first_existing(root, ["messidor2", "messidor-2", "Messidor-2"])
    if base is None:
        raise FileNotFoundError(f"找不到 Messidor-2 目录，看过 {root}")
    label_csv = _first_existing(base, ["messidor_data.csv", "messidor-2-dr-grades.csv"])
    if label_csv is None:
        raise FileNotFoundError(
            "找不到 Messidor-2 裁定标签 messidor_data.csv。"
            "必须从 Kaggle google-brain/messidor2-dr-grades 下载，"
            "原始 Messidor-2 发布里没有 DR 标签。"
        )
    img_dir = _first_existing(base, ["IMAGES", "images", "messidor-2/IMAGES"]) or base

    df = pd.read_csv(label_csv)
    df.columns = [c.strip() for c in df.columns]
    if "adjudicated_dr_grade" not in df.columns:
        raise ValueError(
            f"{label_csv} 里没有 adjudicated_dr_grade 列，说明拿到的不是裁定标签版本"
        )
    if "adjudicated_gradable" in df.columns:
        before = len(df)
        df = df[df["adjudicated_gradable"] != 0]
        LOGGER.info("[messidor2] 剔除 %d 张不可分级（文献值 4）", before - len(df))
    df = df.dropna(subset=["adjudicated_dr_grade"])
    df = df.rename(columns={"adjudicated_dr_grade": "dr_grade"})
    if "image_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "image_id"})
    df["patient_id"] = df["image_id"].astype(str).str.split("_").str[0]
    df["source_split"] = "external"
    df["split"] = "test"  # ← 全程隔离，不允许出现 train / val
    df["path"] = df["image_id"].map(lambda x: str(img_dir / str(x)))
    return _finalize(df, HELDOUT_CLIENT)


BUILDERS = {
    "eyepacs": build_eyepacs,
    "aptos": build_aptos,
    "ddr": build_ddr,
    "idrid": build_idrid,
    "messidor2": build_messidor2,
}


# --------------------------------------------------------------------------- #
# 健全性检查（对应方案第 14 节自查清单）
# --------------------------------------------------------------------------- #
def verify_manifest(df: pd.DataFrame, strict: bool = False) -> List[str]:
    """跑完整的泄漏 / 标签 / 协议检查，返回问题列表（空列表 = 全部通过）。"""
    problems: List[str] = []

    def _check(ok: bool, msg: str) -> None:
        status = "PASS" if ok else "FAIL"
        LOGGER.info("[verify] %-4s %s", status, msg)
        if not ok:
            problems.append(msg)

    _check(df["image_id"].duplicated().sum() == 0, "image_id 无重复")
    _check(bool(df["dr_grade"].between(0, 4).all()), "所有 dr_grade 落在 0-4")
    _check(
        set(df["split"].unique()) <= {"train", "val", "test"},
        "split 只有 train/val/test",
    )

    # 1) 患者级泄漏：每个 client 内部，三个 split 的 patient 集合必须两两不交
    for client, sub in df.groupby("client"):
        groups = {s: set(g["patient_id"]) for s, g in sub.groupby("split")}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            inter = groups.get(a, set()) & groups.get(b, set())
            _check(not inter, f"[{client}] {a}∩{b} 的 patient 交集为空（实际 {len(inter)}）")

    # 2) Messidor-2 必须全程隔离
    m2 = df[df["client"] == HELDOUT_CLIENT]
    if len(m2):
        _check(
            set(m2["split"].unique()) == {"test"},
            "messidor2 全部为 test（未见中心零参与训练）",
        )

    # 3) DDR 不能残留 ungradable
    ddr = df[df["client"] == "ddr"]
    if len(ddr):
        _check(int(ddr["dr_grade"].max()) <= 4, "ddr 已剔除 ungradable")

    # 4) 与文献规模交叉核对
    for client, sub in df.groupby("client"):
        exp = EXPECTED_TOTALS.get(client)
        if exp is None:
            continue
        ok = abs(len(sub) - exp) <= max(5, int(0.01 * exp))
        _check(ok, f"[{client}] 总量 {len(sub)} ≈ 文献 {exp}")
        exp_dist = EXPECTED_GRADE_DIST.get(client)
        if exp_dist:
            got = [int((sub["dr_grade"] == g).sum()) for g in range(5)]
            ok = all(
                abs(a - b) <= max(5, int(0.02 * max(b, 1)))
                for a, b in zip(got, exp_dist)
            )
            _check(ok, f"[{client}] 等级分布 {got} ≈ 文献 {exp_dist}")

    # 5) 每个训练 client 的 train/val/test 都非空
    for client in TRAIN_CLIENTS:
        sub = df[df["client"] == client]
        if not len(sub):
            continue
        for s in ("train", "val", "test"):
            _check(int((sub["split"] == s).sum()) > 0, f"[{client}] {s} 非空")

    if problems and strict:
        raise RuntimeError(f"manifest 校验未通过，共 {len(problems)} 项：{problems}")
    return problems


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """打印方案 3.3 那张规模表，便于直接回填文档。"""
    piv = (
        df.pivot_table(
            index="client", columns="split", values="image_id", aggfunc="count"
        )
        .reindex(columns=["train", "val", "test"])
        .fillna(0)
        .astype(int)
    )
    piv["total"] = piv.sum(axis=1)
    n_train = piv["train"].sum()
    piv["train_share"] = (piv["train"] / max(n_train, 1) * 100).round(1)
    if piv["train"].replace(0, np.nan).min() > 0:
        ratio = piv["train"].max() / piv["train"].replace(0, np.nan).min()
        LOGGER.info("最大/最小 client 训练规模比 = %.1fx", ratio)
    return piv


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建五个 DR 数据集的统一 manifest")
    parser.add_argument("--data-root", type=Path, required=True, help="原始数据根目录")
    parser.add_argument("--out", type=Path, default=Path("manifest.csv"))
    parser.add_argument(
        "--clients",
        nargs="+",
        default=list(BUILDERS),
        choices=list(BUILDERS),
        help="只构建其中几个（缺数据时用）",
    )
    parser.add_argument("--seed", type=int, default=0, help="划分随机种子，定稿后不要改")
    parser.add_argument("--strict", action="store_true", help="校验不通过直接报错退出")
    parser.add_argument("--verify-only", action="store_true", help="只校验 --out 已有文件")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verify_only:
        df = pd.read_csv(args.out)
        problems = verify_manifest(df, strict=args.strict)
        print(summarize(df).to_string())
        return 1 if problems else 0

    frames = []
    for client in args.clients:
        LOGGER.info("=== 构建 %s ===", client)
        try:
            frames.append(BUILDERS[client](args.data_root, args.seed))
        except FileNotFoundError as exc:
            LOGGER.error("跳过 %s：%s", client, exc)
            if args.strict:
                raise

    if not frames:
        LOGGER.error("没有成功构建任何数据集，检查 --data-root")
        return 2

    df = pd.concat(frames, ignore_index=True)
    problems = verify_manifest(df, strict=args.strict)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    LOGGER.info("已写出 %s（%d 行，md5=%s）", args.out, len(df), file_md5(args.out))
    LOGGER.info("把这个 md5 记进实验日志，所有方法必须读同一份 manifest。")
    print(summarize(df).to_string())

    if problems:
        LOGGER.warning("有 %d 项校验未通过，进入联邦实验前必须解决", len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
