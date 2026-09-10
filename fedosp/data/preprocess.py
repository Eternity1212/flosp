"""预处理与缓存：圆形视野紧裁剪 → resize 短边 512 → 存 jpg。

对应 `09_方向二_落地实施方案.md` 第 3.4 节。

**一条红线**：这里只做「去黑边 + 统一尺寸」这种几何操作，
绝不做跨 client 的强风格归一化（CLAHE、直方图匹配、Ben Graham 处理等）——
那会把本课题要研究的 style shift 直接洗掉，实验就没有意义了。

用法::

    python -m fedosp.data.preprocess \
        --manifest manifest.csv --cache-dir /path/to/cache --workers 16

会在 cache-dir 下按 ``<client>/<image_id>.jpg`` 落盘，并写出一份
``manifest_cached.csv``（path 列指向缓存文件），后续训练全部读这份。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("preprocess")


def circle_crop_bbox(img: np.ndarray, thresh_ratio: float = 0.06) -> Tuple[int, int, int, int]:
    """估计眼底圆形视野的紧包围盒，返回 ``(top, bottom, left, right)``。

    做法：灰度化 → 用「全图均值 × 比例」当阈值二值化 → 取有效像素的行列范围。
    比固定阈值稳，因为不同中心的整体亮度差很多（这正是我们要保留的 style 差异）。

    找不到有效区域时返回全图，绝不抛异常打断批处理。
    """
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img
    thresh = max(gray.mean() * thresh_ratio, 7.0)
    mask = gray > thresh

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, img.shape[0], 0, img.shape[1]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    # 裁掉的面积超过 90% 通常意味着阈值判错了，回退到全图
    if (bottom - top) * (right - left) < 0.1 * img.shape[0] * img.shape[1]:
        LOGGER.debug("圆裁剪结果异常，回退全图")
        return 0, img.shape[0], 0, img.shape[1]
    return top, bottom, left, right


def process_one(
    src: str, dst: str, short_side: int = 512, quality: int = 95, overwrite: bool = False
) -> Tuple[str, bool, str]:
    """处理单张图，返回 ``(dst, ok, message)``。子进程里跑，不抛异常。"""
    from PIL import Image  # 延迟导入，避免主进程 fork 前占内存

    try:
        if not overwrite and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst, True, "skip(cached)"
        if not os.path.exists(src):
            return dst, False, "missing source"

        with Image.open(src) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)
            top, bottom, left, right = circle_crop_bbox(arr)
            im = im.crop((left, top, right, bottom))

            w, h = im.size
            scale = short_side / min(w, h)
            if scale < 1.0:  # 只缩不放，避免把 IDRiD 的高分辨率图放大反而糊
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.BICUBIC)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im.save(dst, "JPEG", quality=quality)
        return dst, True, "ok"
    except Exception as exc:  # noqa: BLE001 — 批处理不能因单张图中断
        return dst, False, f"{type(exc).__name__}: {exc}"


def run(
    manifest: Path,
    cache_dir: Path,
    short_side: int = 512,
    quality: int = 95,
    workers: int = 8,
    overwrite: bool = False,
    clients: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    if clients:
        df = df[df["client"].isin(clients)].copy()
    LOGGER.info("待处理 %d 张，输出到 %s（短边 %d, q%d）",
                len(df), cache_dir, short_side, quality)

    df["cached_path"] = [
        str(cache_dir / str(c) / f"{i}.jpg")
        for c, i in zip(df["client"], df["image_id"])
    ]

    n_ok = n_fail = n_skip = 0
    failures: List[Tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_one, s, d, short_side, quality, overwrite): d
            for s, d in zip(df["path"], df["cached_path"])
        }
        for n_done, fut in enumerate(as_completed(futures), 1):
            dst, ok, msg = fut.result()
            if not ok:
                n_fail += 1
                if len(failures) < 20:
                    failures.append((dst, msg))
            elif msg.startswith("skip"):
                n_skip += 1
            else:
                n_ok += 1
            if n_done % 2000 == 0:
                LOGGER.info("进度 %d/%d | ok=%d skip=%d fail=%d",
                            n_done, len(futures), n_ok, n_skip, n_fail)

    LOGGER.info("完成：新处理 %d，跳过 %d，失败 %d", n_ok, n_skip, n_fail)
    for dst, msg in failures:
        LOGGER.error("失败样例 %s -> %s", dst, msg)
    if n_fail:
        LOGGER.warning("有 %d 张失败，训练前请先解决，否则 DataLoader 会中途报错", n_fail)

    out = df.copy()
    out["raw_path"] = out["path"]
    out["path"] = out["cached_path"]
    out = out.drop(columns=["cached_path"])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="眼底图预处理与缓存")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="默认写到 manifest 同目录的 manifest_cached.csv")
    parser.add_argument("--short-side", type=int, default=512)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--clients", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_df = run(
        args.manifest,
        args.cache_dir,
        args.short_side,
        args.quality,
        args.workers,
        args.overwrite,
        args.clients,
    )
    out_path = args.out or args.manifest.with_name("manifest_cached.csv")
    out_df.to_csv(out_path, index=False)
    LOGGER.info("已写出 %s（%d 行）。后续训练一律读这份。", out_path, len(out_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
