"""生成**结构仿真**的迷你数据集，用来在拿到真实数据之前跑通整条管线。

动机
----
真实数据有两个门槛：体积（EyePACS 单个 35 GB）和许可（Messidor-2 / RETFound
都禁止再分发）。结果是"管线对不对"这件事，往往要等下完几十 GB 才能验证 ——
如果 ``build_manifest`` 的某个目录名猜错了，那几个小时的下载就白费。

这个脚本造出**目录布局、标签文件格式、文件名规则全部与真实数据一致**，但每个
数据集只有几十张 64×64 假图的 fixture。于是：

* ``build_manifest`` 的五个 builder 全部走真实代码路径（包括 EyePACS 的
  病人级切分、DDR 的 ungradable 剔除、IDRiD 的官方 413/103、Messidor-2 的隔离）
* ``preprocess`` 的圆形裁剪 + resize 也是真实路径（图里画了圆形视野）
* 之后可以直接跑 ``run_fed``

它**不能**验证什么：真实图像质量、真实标签分布、模型精度。
``verify_manifest`` 会在"与文献规模交叉核对"那几项上报 FAIL —— 这是**故意的**，
保证 fixture 产出的 manifest 不可能被误当成真实数据。

用法::

    python scripts/make_fixture.py --out data/fixture
    python -m fedosp.data.build_manifest --data-root data/fixture --out data/fixture/manifest.csv
    python -m fedosp.data.preprocess --manifest data/fixture/manifest.csv --out data/fixture/cache
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("make_fixture")

#: 每个 client 每个等级造几张。真实分布极不均衡（EyePACS 73.5% 是 0 级），
#: 这里保留"不均衡"这个性质但压到几十张，否则分层切分会把稀有等级抽没。
GRADE_COUNTS: Dict[str, Sequence[int]] = {
    #                 0   1   2   3   4
    "eyepacs":       (20, 4, 8, 3, 3),   # 会再 ×2（左右眼），共 76 张
    "aptos":         (12, 4, 8, 3, 4),
    "ddr":           (12, 4, 9, 3, 4),
    "idrid":         (10, 3, 10, 5, 4),
    "messidor2":     (10, 4, 5, 3, 3),
}


def _fake_fundus(rng: np.random.Generator, size: int = 64) -> Image.Image:
    """造一张有**圆形视野**的假眼底图。

    圆形是必须的：``preprocess.py`` 第一步就是"圆形视野紧裁剪"，纯噪声方图
    会让那段代码走到与真实数据不同的分支上，等于没验证。
    """
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    r = size * 0.47
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

    img = np.zeros((size, size, 3), dtype=np.float32)
    # 眼底的典型橙红色调 + 一点径向渐变，让裁剪和归一化有真实点的输入
    base = np.array([150.0, 70.0, 40.0])
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(r, 1e-6)
    for c in range(3):
        img[..., c] = base[c] * (1.0 - 0.35 * np.clip(dist, 0, 1))
    img += rng.normal(0, 8, img.shape)
    img[~inside] = 0.0                      # 视野外全黑，和真实数据一样
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _write_images(paths: Sequence[Path], rng: np.random.Generator, size: int) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        _fake_fundus(rng, size).save(p, quality=92)


def _expand(counts: Sequence[int]) -> List[int]:
    """``(20, 4, 8, 3, 3)`` -> ``[0]*20 + [1]*4 + ...``"""
    out: List[int] = []
    for grade, n in enumerate(counts):
        out += [grade] * int(n)
    return out


# --------------------------------------------------------------------------- #
def make_eyepacs(root: Path, rng: np.random.Generator, size: int) -> int:
    """布局：``eyepacs/trainLabels.csv`` + ``eyepacs/train/<id>.jpeg``

    ★ 关键：文件名必须是 ``<patient>_left`` / ``<patient>_right``，
    因为 ``build_eyepacs`` 用下划线前缀当 ``patient_id`` 做**病人级**切分。
    只造单眼图的话，这条防泄漏逻辑就完全没被测到。
    """
    base = root / "eyepacs"
    img_dir = base / "train"
    grades = _expand(GRADE_COUNTS["eyepacs"])

    rows, paths = [], []
    for i, g in enumerate(grades):
        for eye in ("left", "right"):
            image_id = f"{i + 1}_{eye}"
            rows.append((image_id, g))       # 左右眼同等级，符合真实数据的高相关
            paths.append(img_dir / f"{image_id}.jpeg")
    _write_images(paths, rng, size)

    base.mkdir(parents=True, exist_ok=True)
    with open(base / "trainLabels.csv", "w") as fh:
        fh.write("image,level\n")
        for image_id, g in rows:
            fh.write(f"{image_id},{g}\n")
    return len(rows)


def make_aptos(root: Path, rng: np.random.Generator, size: int) -> int:
    """布局：``aptos2019/train.csv``（列 id_code,diagnosis）+ ``train_images/<id>.png``"""
    base = root / "aptos2019"
    img_dir = base / "train_images"
    grades = _expand(GRADE_COUNTS["aptos"])

    ids = [f"aptos_{i:04d}" for i in range(len(grades))]
    _write_images([img_dir / f"{i}.png" for i in ids], rng, size)

    base.mkdir(parents=True, exist_ok=True)
    with open(base / "train.csv", "w") as fh:
        fh.write("id_code,diagnosis\n")
        for i, g in zip(ids, grades):
            fh.write(f"{i},{g}\n")
    return len(ids)


def make_ddr(root: Path, rng: np.random.Generator, size: int) -> int:
    """布局：``DDR-dataset/DR_grading/{train,valid,test}.txt`` + 同名图像目录。

    标签文件是空格分隔、无表头的 ``<name>.jpg <label>``。
    ★ 故意掺入 ``label == 5``（ungradable），用来验证 ``build_ddr`` 真的把它剔掉了 ——
    这是 DDR 最容易搞错的地方（5 不是第 6 个严重度等级）。
    """
    base = root / "DDR-dataset" / "DR_grading"
    grades = _expand(GRADE_COUNTS["ddr"])
    rng.shuffle(grades)

    # 按官方比例切三份（真实为 6835/2733/4105）
    n = len(grades)
    bounds = [0, int(n * 0.52), int(n * 0.73), n]
    total = 0
    for idx, split in enumerate(["train", "valid", "test"]):
        part = grades[bounds[idx]:bounds[idx + 1]]
        img_dir = base / split
        names = [f"{split}-{i:04d}.jpg" for i in range(len(part))]
        # 每个 split 掺 2 张 ungradable
        names += [f"{split}-ung-{i}.jpg" for i in range(2)]
        labels = list(part) + [5, 5]
        _write_images([img_dir / nm for nm in names], rng, size)
        base.mkdir(parents=True, exist_ok=True)
        with open(base / f"{split}.txt", "w") as fh:
            for nm, lb in zip(names, labels):
                fh.write(f"{nm} {lb}\n")
        total += len(part)                    # 只算会被保留的
    return total


def make_idrid(root: Path, rng: np.random.Generator, size: int) -> int:
    """布局：官网 zip 解压后的原始目录名（带序号和空格）。

    ``a. IDRiD_Disease Grading_Training Labels.csv``（列 ``Image name``/``Retinopathy grade``）
    ``1. Original Images/a. Training Set/<name>.jpg``

    ⚠️ **HuggingFace 上的 IDRiD 镜像往往不是这个布局**（常见是 parquet 或扁平目录）。
    如果你用的是 HF 镜像，需要先整理成这里的结构，或改 ``build_idrid`` 的候选名。
    ``_first_existing`` 有 rglob 兜底，所以只要那两个 CSV 文件名没变、
    图像目录名里含 "Training Set"/"Testing Set" 就能认出来。
    """
    base = root / "idrid"
    grades = _expand(GRADE_COUNTS["idrid"])
    n_test = max(5, len(grades) // 5)
    rng.shuffle(grades)
    train_g, test_g = grades[n_test:], grades[:n_test]

    for label_file, gs, img_sub in [
        ("a. IDRiD_Disease Grading_Training Labels.csv", train_g,
         "1. Original Images/a. Training Set"),
        ("b. IDRiD_Disease Grading_Testing Labels.csv", test_g,
         "1. Original Images/b. Testing Set"),
    ]:
        names = [f"IDRiD_{i:03d}" for i in range(len(gs))]
        if "Testing" in label_file:
            names = [f"IDRiD_T{i:03d}" for i in range(len(gs))]   # 避免与 train 重名
        _write_images([base / img_sub / f"{nm}.jpg" for nm in names], rng, size)
        base.mkdir(parents=True, exist_ok=True)
        with open(base / label_file, "w") as fh:
            fh.write("Image name,Retinopathy grade,Risk of macular edema \n")
            for nm, g in zip(names, gs):
                fh.write(f"{nm},{g},0\n")
    return len(grades)


def make_messidor2(root: Path, rng: np.random.Generator, size: int) -> int:
    """布局：``messidor2/messidor_data.csv`` + ``messidor2/IMAGES/<image_id>``

    标签列必须是 ``adjudicated_dr_grade``（Google Brain 裁定标签），
    ``build_messidor2`` 会在缺这一列时直接报错 —— 原始 Messidor-2 发布里没有 DR 标签。
    ★ 故意掺 2 张 ``adjudicated_gradable == 0``，验证剔除逻辑。
    """
    base = root / "messidor2"
    img_dir = base / "IMAGES"
    grades = _expand(GRADE_COUNTS["messidor2"])

    rows = []
    for i, g in enumerate(grades):
        rows.append((f"patient{i // 2:03d}_{i}.png", g, 1))
    for i in range(2):                       # 不可分级
        rows.append((f"patientX_ung{i}.png", 0, 0))
    _write_images([img_dir / nm for nm, _, _ in rows], rng, size)

    base.mkdir(parents=True, exist_ok=True)
    with open(base / "messidor_data.csv", "w") as fh:
        fh.write("image_id,adjudicated_dr_grade,adjudicated_dme,adjudicated_gradable\n")
        for nm, g, gradable in rows:
            fh.write(f"{nm},{g},0,{gradable}\n")
    return len(grades)


MAKERS = {
    "eyepacs": make_eyepacs,
    "aptos": make_aptos,
    "ddr": make_ddr,
    "idrid": make_idrid,
    "messidor2": make_messidor2,
}


# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/fixture"),
                    help="fixture 根目录，直接当 build_manifest 的 --data-root")
    ap.add_argument("--clients", nargs="+", default=list(MAKERS), choices=list(MAKERS))
    ap.add_argument("--img-size", type=int, default=64, help="假图边长，64 足够验证管线")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    for name in args.clients:
        n = MAKERS[name](args.out, rng, args.img_size)
        total += n
        LOGGER.info("  %-11s %3d 张（有效标签）", name, n)

    LOGGER.info("\n共 %d 张假图 → %s", total, args.out)
    LOGGER.info(
        "\n⚠ 这是**结构仿真**数据，不是真实数据。verify_manifest 会在"
        "「与文献规模交叉核对」上报 FAIL，这是故意的，"
        "保证 fixture 的 manifest 不会被误当成真实结果。\n"
    )
    LOGGER.info("下一步：")
    LOGGER.info("  python -m fedosp.data.build_manifest --data-root %s --out %s",
                args.out, args.out / "manifest.csv")
    LOGGER.info("  python -m fedosp.data.preprocess --manifest %s --out %s",
                args.out / "manifest.csv", args.out / "cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
