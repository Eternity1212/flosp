"""Dataset / DataLoader / 增强 / 标签预算抽样。

对应方案 3.4（增强配方）、3.5（manifest 唯一入口）、8.2（标签效率曲线）。

关键约定：

* 所有实验从同一份 manifest 读数据，靠 ``client`` + ``split`` 过滤；
* 训练增强按 RETFound 原始配方，**不做跨 client 的风格归一化**；
* ``style_aug=True`` 时每个样本额外返回一个「风格扰动视图」，供 L_style / L_cons 使用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("dataset")

NUM_CLASSES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224, train: bool = True):
    """RETFound 原始配方：训练 RandomResizedCrop(scale 0.2-1.0)+翻转+轻旋转。"""
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(
            int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_style_transform(img_size: int = 224):
    """风格扰动视图：只改颜色/亮度/gamma，几何保持与主视图同分布。

    这是 FSR 门控的直接监督信号——同一张图换个「相机风格」，
    FSR 之后的特征应该一致。
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
        transforms.RandomApply(
            [transforms.Lambda(lambda im: _gamma(im, np.random.uniform(0.6, 1.6)))], p=0.8
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _gamma(im, g: float):
    from PIL import ImageEnhance  # noqa: F401  (保持依赖显式)
    import PIL.Image as Image

    arr = np.asarray(im).astype(np.float32) / 255.0
    arr = np.clip(arr ** g, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


class FundusDataset(Dataset):
    """从 manifest 的一个切片构造数据集。

    Args:
        frame: 已经按 client/split 过滤好的 DataFrame。
        img_size: 输入分辨率（224 或 384）。
        train: 是否用训练增强。
        style_aug: 是否额外返回风格扰动视图（训练 FedOSP 时为 True）。
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        img_size: int = 224,
        train: bool = True,
        style_aug: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transforms(img_size, train)
        self.style_transform = build_style_transform(img_size) if style_aug else None
        self.paths = self.frame["path"].tolist()
        self.labels = self.frame["dr_grade"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        from PIL import Image

        path = self.paths[idx]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                x = self.transform(im)
                x2 = self.style_transform(im) if self.style_transform else None
        except Exception as exc:  # noqa: BLE001
            # 单张坏图不应该拖垮整轮训练；记日志并返回零图，由上层监控 corrupt 计数
            LOGGER.error("读图失败 idx=%d path=%s: %s", idx, path, exc)
            c = 3
            size = self.transform.transforms[0].size
            size = size[0] if isinstance(size, (tuple, list)) else size
            x = torch.zeros(c, size, size)
            x2 = torch.zeros_like(x) if self.style_transform else None

        y = int(self.labels[idx])
        if x2 is None:
            return x, y
        return x, x2, y

    def class_counts(self) -> np.ndarray:
        """本地类别计数，供 CBCE 的 effective-number 权重使用（**必须用本地分布**）。"""
        return np.bincount(self.labels, minlength=NUM_CLASSES).astype(np.float64)


def apply_label_budget(
    frame: pd.DataFrame, budget: Optional[str], seed: int
) -> pd.DataFrame:
    """按标签预算做分层抽样，用于 8.2 的标签效率曲线。

    ``budget`` 支持三种写法：

    * ``None`` / ``"100%"``：全量
    * ``"400"``：每个 client 保留 400 张（分层）
    * ``"20%"``：保留 20%（分层）
    """
    if budget in (None, "", "100%", "1.0"):
        return frame
    rng = np.random.RandomState(seed)

    if str(budget).endswith("%"):
        ratio = float(str(budget)[:-1]) / 100.0
        n_target = int(round(len(frame) * ratio))
    else:
        n_target = int(budget)
    n_target = min(n_target, len(frame))

    # 按类别比例分配名额，每类至少 1 张（否则小类会被抽没）
    counts = frame["dr_grade"].value_counts().sort_index()
    quota = (counts / counts.sum() * n_target).round().astype(int).clip(lower=1)
    picked = []
    for grade, idx in frame.groupby("dr_grade").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        picked.append(idx[: min(int(quota.get(grade, 1)), len(idx))])
    out = frame.loc[np.concatenate(picked)]
    LOGGER.info(
        "标签预算 %s：%d -> %d 张，分布 %s",
        budget, len(frame), len(out),
        out["dr_grade"].value_counts().sort_index().to_dict(),
    )
    return out


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_id", "client", "split", "dr_grade", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest 缺列 {missing}：{path}")
    return df


def make_client_loaders(
    manifest: pd.DataFrame,
    client: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
    style_aug: bool = False,
    label_budget: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, DataLoader]:
    """给一个 client 造 train/val/test 三个 loader。"""
    sub = manifest[manifest["client"] == client]
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        part = sub[sub["split"] == split]
        if not len(part):
            continue
        if split == "train":
            part = apply_label_budget(part, label_budget, seed)
        ds = FundusDataset(
            part,
            img_size=img_size,
            train=(split == "train"),
            style_aug=style_aug and split == "train",
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train" and len(ds) > batch_size),
            persistent_workers=num_workers > 0,
        )
    LOGGER.info(
        "[%s] loaders: %s",
        client,
        {k: len(v.dataset) for k, v in loaders.items()},
    )
    return loaders


def local_steps(n_k: int, alpha: float = 1.275, lo: int = 20, hi: int = 200) -> int:
    """本地步数均衡（方案 4.6）：``S_k = clip(round(alpha*sqrt(n_k)), lo, hi)``。

    这是比聚合权重更隐蔽的一层不公平：按 epoch 训练时
    EyePACS 一轮走 769 步、IDRiD 只走 12 步，相差 64 倍。

    ``alpha = 200 / sqrt(24600) ≈ 1.275`` 由「让最大 client 恰好取到上界」反解得到。
    用 ``floor(x+0.5)`` 而不是内置 ``round``，避免 banker's rounding 让结果依赖平台。
    """
    steps = int(np.floor(alpha * np.sqrt(max(n_k, 1)) + 0.5))
    return int(np.clip(steps, lo, hi))


def aggregation_weights(
    n_list: Sequence[int], mode: str = "sqrt", clip_q: Tuple[float, float] = (0.1, 0.9)
) -> np.ndarray:
    """LoRA / head 的聚合权重（方案 4.5）。

    * ``sample``：FedAvg 原味，按样本数 —— EyePACS 会拿到 ~68% 权重
    * ``equal``：client 等权
    * ``sqrt``：本文默认，sqrt 缩放 + 分位裁剪，介于两者之间
    """
    n = np.asarray(n_list, dtype=np.float64)
    if mode == "sample":
        w = n
    elif mode == "equal":
        w = np.ones_like(n)
    elif mode == "sqrt":
        w = np.sqrt(n)
        lo, hi = np.quantile(w, clip_q[0]), np.quantile(w, clip_q[1])
        w = np.clip(w, lo, hi)
    else:
        raise ValueError(f"未知聚合权重模式: {mode}")
    return w / w.sum()


__all__ = [
    "NUM_CLASSES",
    "FundusDataset",
    "aggregation_weights",
    "apply_label_budget",
    "build_style_transform",
    "build_transforms",
    "load_manifest",
    "local_steps",
    "make_client_loaders",
]
