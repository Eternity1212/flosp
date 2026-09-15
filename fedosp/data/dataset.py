"""Dataset / DataLoader / 增强 / 标签预算抽样。

对应方案 3.4（增强配方）、3.5（manifest 唯一入口）、8.2（标签效率曲线）。

关键约定：

* 所有实验从同一份 manifest 读数据，靠 ``client`` + ``split`` 过滤；
* 训练增强按 RETFound 原始配方，**不做跨 client 的风格归一化**；
* ``style_aug=True`` 时每个样本额外返回一个「风格扰动视图」，供 L_style / L_cons 使用。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
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

    #: 读图失败率超过这个比例就中止训练。零图会被当成一张合法的黑底眼底照参与
    #: 反向传播，静默吞掉等于让损坏数据污染结果且事后不可见，所以必须有上限。
    MAX_FAILED_READ_RATIO = 1e-3

    def __init__(
        self,
        frame: pd.DataFrame,
        img_size: int = 224,
        train: bool = True,
        style_aug: bool = False,
        freq_bank=None,
        freq_client: str = "",
        freq_ratio: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.img_size = img_size
        self.transform = build_transforms(img_size, train)
        self.style_transform = build_style_transform(img_size) if style_aug else None
        # ---- B16 FedDG-ELCFS：第二视图改由跨 client 幅度谱插值生成 ----
        self.freq_bank = freq_bank
        self.freq_client = freq_client
        self.freq_ratio = freq_ratio
        self._seed = seed
        self._rng: Optional[np.random.Generator] = None
        if freq_bank is not None and self.style_transform is not None:
            # 两者都会去占"第二视图"这个位置，同时开必然有一个被静默忽略
            raise ValueError(
                "style_aug 与 freq_bank 不能同时启用：它们都产出第二视图，"
                "同时开会让其中一个被静默丢弃。ELCFS(B16) 用 freq_bank，"
                "FedOSP 的 L_style/L_cons 用 style_aug。"
            )
        self.paths = self.frame["path"].tolist()
        self.labels = self.frame["dr_grade"].astype(int).to_numpy()
        # DataLoader 的 worker 是独立进程，普通 int 计数器加不回主进程；用共享内存。
        self._failed = mp.Value("i", 0)
        self._max_failed = max(1, int(len(self.frame) * self.MAX_FAILED_READ_RATIO))

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def failed_reads(self) -> int:
        """本数据集累计读图失败张数，写进 ``result.json`` 的 ``data.n_failed_reads``。"""
        return int(self._failed.value)

    def verify_paths(self) -> None:
        """开跑前检查所有图片文件存在（只 stat 不解码，几万张也是秒级）。

        Raises:
            FileNotFoundError: 有缺失文件时列出前若干个。
        """
        missing = [p for p in self.paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"manifest 里有 {len(missing)}/{len(self.paths)} 张图片不存在，"
                f"例如 {missing[:5]}。先跑 fedosp.data.preprocess 生成缓存，"
                "或确认 --manifest 指向的是 manifest_cached.csv。"
            )

    def __getitem__(self, idx: int):
        from PIL import Image

        path = self.paths[idx]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                x = self.transform(im)
                if self.style_transform is not None:
                    x2 = self.style_transform(im)
                elif self.freq_bank is not None:
                    x2 = self.transform(self._freq_view(im))
                else:
                    x2 = None
        except Exception as exc:  # noqa: BLE001
            with self._failed.get_lock():
                self._failed.value += 1
                n_failed = self._failed.value
            LOGGER.error(
                "读图失败 (%d/%d 容忍上限) idx=%d path=%s: %s",
                n_failed, self._max_failed, idx, path, exc,
            )
            if n_failed > self._max_failed:
                raise RuntimeError(
                    f"读图失败累计 {n_failed} 张，超过容忍上限 {self._max_failed}"
                    f"（{self.MAX_FAILED_READ_RATIO:.2%} of {len(self.frame)}）。"
                    "中止训练：继续跑会让零图当作黑底眼底照污染结果。"
                    "请检查图片缓存是否完整（fedosp.data.preprocess）。"
                ) from exc
            c = 3
            size = self.transform.transforms[0].size
            size = size[0] if isinstance(size, (tuple, list)) else size
            x = torch.zeros(c, size, size)
            x2 = (
                torch.zeros_like(x)
                if (self.style_transform is not None or self.freq_bank is not None)
                else None
            )

        y = int(self.labels[idx])
        if x2 is None:
            return x, y
        return x, x2, y

    def _freq_view(self, im):
        """ELCFS 的频域增广视图：换成别家医院的低频幅度、保留自己的相位。"""
        from PIL import Image

        from .freq_aug import amplitude_mix

        if self._rng is None:
            # 每个 DataLoader worker 必须拿到不同的随机流，否则所有 worker 会抽到
            # 同一条幅度谱，增广多样性直接坍缩到 1/num_workers 而且看不出来。
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self._seed * 10_000 + wid + 1)

        arr = np.asarray(
            im.resize((self.img_size, self.img_size), Image.BICUBIC)
        ).astype(np.float32) / 255.0
        amp = self.freq_bank.sample(self.freq_client, self._rng)
        lam = float(self._rng.uniform(0.0, 1.0))      # 原文的 continuous frequency space
        mixed = amplitude_mix(arr, amp, lam, self.freq_ratio)
        return Image.fromarray((mixed * 255).astype(np.uint8))

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
    freq_bank=None,
    freq_ratio: float = 0.01,
    max_train: Optional[int] = None,
    train_fraction: Optional[float] = None,
    min_train: int = 500,
) -> Dict[str, DataLoader]:
    """给一个 client 造 train/val/test 三个 loader。

    Args:
        freq_bank: B16 FedDG-ELCFS 的共享幅度谱库。传了就在**训练集**上启用
            跨 client 频域增广（只在 train，理由同 style_aug）。
        train_fraction: pilot 降规模：训练集按**同一比例**抽样。这是 pilot 的推荐
            降规模方式，理由见下。
        min_train: 配合 ``train_fraction`` 的下限，小于它的 client 整体保全。
        max_train: 训练集**统一上限**。⚠ 不推荐用于 pilot（见下）。

    为什么 pilot 必须用比例而不是统一上限
    ------------------------------------
    本文 C2（精度加权聚合）的收益**完全来自客户端规模不平衡**：
    EyePACS 24600 张对 IDRiD 372 张，:math:`n_{\\mathrm{eff}}=1.75`。

    ==========================  ==============  =====================
    降规模方式                    ``n_eff``      后果
    ==========================  ==============  =====================
    全量（正式实验）               1.75           --
    统一截到 2000                 **3.34**       C2 的改进空间被消掉
    按比例 20% + 小院保全           1.90           结论仍可迁移
    ==========================  ==============  =====================

    统一截断会把这个联邦变成一个近似等规模的联邦，于是 pilot 会得出"C2 没用"
    的结论 —— 而这个结论只是降规模方式的产物，与 C2 本身无关。
    ``val`` / ``test`` **一律不降规模**，保证评估指标与正式实验可比。
    """
    sub = manifest[manifest["client"] == client]
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        part = sub[sub["split"] == split]
        if not len(part):
            continue
        if split == "train":
            part = apply_label_budget(part, label_budget, seed)
            # pilot 降规模。val/test 不动，保证评估与正式实验可比。
            target = None
            why = ""
            if train_fraction is not None and 0 < train_fraction < 1:
                target = int(round(len(part) * train_fraction))
                why = f"--train-fraction {train_fraction}"
                if len(part) <= min_train:
                    # 小院整体保全：按比例抽会把稀有等级抽没
                    # （IDRiD 的 grade 1 全院只有 20 张，20% = 4 张）
                    target = None
                    LOGGER.info(
                        "[%s] 训练集仅 %d 张 (<=%d)，pilot 降规模跳过该 client 以保住稀有等级",
                        client, len(part), min_train,
                    )
                elif target < min_train:
                    target = min_train
                    why += f"（受 --min-train-per-client {min_train} 抬升）"
            if max_train is not None and len(part) > max_train:
                target = min(target or max_train, max_train)
                why = (why + " + " if why else "") + f"--max-train-per-client {max_train}"
            if target is not None and target < len(part):
                before = len(part)
                # 复用同一个分层抽样实现，保证小类不会被抽没
                part = apply_label_budget(part, str(int(target)), seed)
                LOGGER.info(
                    "[%s] pilot 降规模：训练集 %d -> %d 张（%s）",
                    client, before, len(part), why,
                )
        ds = FundusDataset(
            part,
            img_size=img_size,
            train=(split == "train"),
            style_aug=style_aug and split == "train",
            freq_bank=freq_bank if split == "train" else None,
            freq_client=client,
            freq_ratio=freq_ratio,
            seed=seed,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            # MPS 不支持 pinned memory，传 True 只会每个 loader 刷一条警告
            pin_memory=torch.cuda.is_available(),
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
