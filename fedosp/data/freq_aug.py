"""FedDG-ELCFS（Liu et al., CVPR 2021）的连续频域增广 —— 基线 B16 的数据侧。

这个模块实现 ELCFS 的第一个组件：**跨 client 共享的幅度谱银行** + 低频段内的
连续插值增广。它同时是本文 FSR 的**直接对手**与**隐私论证的对照物**。

方法
----
一张图的傅里叶变换可以拆成幅度谱与相位谱：

.. math:: \\mathcal{F}(x) = A(x)\\,e^{\\,i P(x)}

经验事实（FDA / ELCFS 的共同前提）：**幅度谱主要编码"风格"**（相机型号、曝光、
色温、镜头晕影），**相位谱主要编码"内容"**（血管走向、出血点位置、视盘结构）。
所以把自己的幅度换成别家医院的幅度、保留自己的相位，就得到一张
"内容不变、风格变成别家"的图 —— 一个免费的跨中心数据增广。

只在**低频**一小块里换（``ratio`` 控制），因为高频幅度里含有病灶的细结构，
换掉会真的改变内容。

.. math::
    A' = (1-\\lambda M)\\odot A_{self} + \\lambda M \\odot A_{other},
    \\qquad \\lambda\\sim U(0,1)

其中 :math:`M` 是以频谱中心为心的低频掩码。:math:`\\lambda` 连续取值，
这就是原文标题里 "continuous frequency space" 的含义。

⚠ 隐私代价：这是本文必须正面论证的一条
--------------------------------------
ELCFS 要求各 client 把**原始图像的幅度谱**传到中心。幅度谱不是聚合统计量，
而是**单张图像的完整频域信息**：把它与任意一张图的相位组合就能重建出可识别的
眼底结构；即便只传低频段，也泄露了该院图像的色彩/曝光指纹。

对照之下，FSR 只在**中间特征**上按 batch 做幅度归一，**什么都不外传**。
这不是本文方法"顺便"的优点，而是 T5（系统开销）与隐私讨论里应该单列的一行：

======================  ==========================  ==============================
\\                        ELCFS                        FSR（本文）
======================  ==========================  ==============================
需要外传的东西            单张图像的幅度谱             无
通信量                   ``0.57 MB/张`` × 库大小      0
可逆性                   与任意相位组合即可重建图像     不适用
======================  ==========================  ==============================
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger("freq_aug")


def _low_freq_mask(h: int, w: int, ratio: float) -> np.ndarray:
    """以频谱中心为心的方形低频掩码（配合 ``fftshift`` 使用）。

    ``ratio`` 是半宽占边长的比例。FDA / ELCFS 常用 0.01，即 224 分辨率下
    半宽 2 像素、约 5x5 的一小块 —— 听起来很小，但低频承载了绝大部分能量，
    这一块足以改变整张图的色调与曝光。
    """
    mask = np.zeros((h, w), dtype=np.float32)
    b_h = max(1, int(np.floor(min(h, w) * ratio)))
    cy, cx = h // 2, w // 2
    mask[cy - b_h: cy + b_h + 1, cx - b_h: cx + b_h + 1] = 1.0
    return mask


def amplitude_of(img: np.ndarray) -> np.ndarray:
    """取一张 ``(H, W, 3)``、取值 ``[0,1]`` 的图的**中心化**幅度谱。

    返回 ``(H, W, 3)`` float32。已经过 ``fftshift``，所以低频在正中间 ——
    这样 :func:`_low_freq_mask` 才能直接用。
    """
    f = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    return np.abs(f).astype(np.float32)


def amplitude_mix(
    img: np.ndarray,
    amp_target: np.ndarray,
    lam: float,
    ratio: float = 0.01,
) -> np.ndarray:
    """把 ``img`` 的低频幅度朝 ``amp_target`` 插值，**相位完全保留**。

    Args:
        img: ``(H, W, 3)``，取值 ``[0, 1]``。
        amp_target: 同形状的目标幅度谱（来自 :class:`AmplitudeBank`）。
        lam: 插值强度，``0`` = 原图，``1`` = 低频幅度完全换成对方的。
        ratio: 低频掩码半宽占比。

    Returns:
        ``(H, W, 3)`` float32，已 clip 回 ``[0, 1]``。

    实现注意：逆变换后取 ``.real`` 会丢掉数值误差带来的虚部（幅度被改过之后
    结果不再严格是实信号的频谱）。这是 FDA/ELCFS 原实现的做法，不是近似错误。
    """
    if img.shape != amp_target.shape:
        raise ValueError(
            f"幅度谱形状 {amp_target.shape} 与图像 {img.shape} 不一致；"
            "AmplitudeBank 的 img_size 必须与训练分辨率相同"
        )
    h, w = img.shape[:2]
    f = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    amp, phase = np.abs(f), np.angle(f)

    m = _low_freq_mask(h, w, ratio)[:, :, None] * float(lam)
    amp_new = amp * (1.0 - m) + amp_target * m

    f_new = amp_new * np.exp(1j * phase)
    out = np.fft.ifft2(np.fft.ifftshift(f_new, axes=(0, 1)), axes=(0, 1)).real
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class AmplitudeBank:
    """跨 client 共享的幅度谱库 —— ELCFS 需要**外传**的那个东西。

    Args:
        amps: ``(N, H, W, 3)`` 幅度谱。
        owners: 长度 ``N``，每条谱来自哪个 client。

    ``owners`` 的用处是 :meth:`sample` 能排除"自己家"的谱：拿自己的幅度做增广
    等于没增广，会让这个基线被低估。
    """

    def __init__(self, amps: np.ndarray, owners: Sequence[str]) -> None:
        if len(amps) != len(owners):
            raise ValueError(f"amps({len(amps)}) 与 owners({len(owners)}) 长度不一致")
        if len(amps) == 0:
            raise ValueError("幅度谱库为空：ELCFS 没有可换的风格，退化成无增广")
        self.amps = np.asarray(amps, dtype=np.float32)
        self.owners = list(owners)
        self._by_other: Dict[str, np.ndarray] = {}
        LOGGER.info(
            "幅度谱库就绪：%d 条 @ %s | 外传体积 %.1f MB | 来源 %s",
            len(self.amps), self.amps.shape[1:3], self.payload_mb(),
            {c: self.owners.count(c) for c in sorted(set(self.owners))},
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        manifest,
        img_size: int = 224,
        per_client: int = 10,
        seed: int = 0,
        clients: Optional[Sequence[str]] = None,
    ) -> "AmplitudeBank":
        """从 manifest 的**训练**切片里各 client 抽 ``per_client`` 张图建库。

        只从 train 切片抽，理由和别处一样：val/test 的任何信息都不能进训练通路，
        哪怕只是"风格"。

        ``per_client`` 是效果与代价的折中。原文用全部图像，但幅度谱在同一台相机下
        高度相似，边际收益很快饱和；10 张已经能覆盖该院的曝光/色温范围，而外传体积
        只有 ``10 x 4 x 0.57 MB ≈ 23 MB``。这个数要写进 T5 —— 它是 ELCFS 相对
        FSR 的净额外通信开销。
        """
        from PIL import Image

        rng = np.random.default_rng(seed)
        train = manifest[manifest["split"] == "train"]
        names = list(clients) if clients is not None else sorted(train["client"].unique())

        amps: List[np.ndarray] = []
        owners: List[str] = []
        for name in names:
            paths = train[train["client"] == name]["path"].tolist()
            if not paths:
                LOGGER.warning("client %s 没有训练图，跳过建库", name)
                continue
            take = rng.choice(len(paths), size=min(per_client, len(paths)), replace=False)
            ok = 0
            for i in take:
                try:
                    with Image.open(paths[int(i)]) as im:
                        arr = np.asarray(
                            im.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
                        ).astype(np.float32) / 255.0
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("建库读图失败 %s: %s", paths[int(i)], exc)
                    continue
                amps.append(amplitude_of(arr))
                owners.append(name)
                ok += 1
            if ok == 0:
                LOGGER.warning("client %s 一张都没读成功，它的风格不会进库", name)
        return cls(np.stack(amps), owners)

    # ------------------------------------------------------------------ #
    def payload_mb(self) -> float:
        """这个库需要在网络上传输的体积（MB，float32）。"""
        return float(self.amps.nbytes) / 1024 / 1024

    def others(self, client: str) -> np.ndarray:
        """``client`` 之外的所有幅度谱的下标。结果按 client 缓存。"""
        if client not in self._by_other:
            idx = np.array(
                [i for i, o in enumerate(self.owners) if o != client], dtype=np.int64
            )
            if idx.size == 0:
                LOGGER.warning(
                    "库里只有 %s 自己的幅度谱，ELCFS 将退化成自我增广（几乎无效）。"
                    "检查建库时是否只传了一个 client。", client,
                )
                idx = np.arange(len(self.amps), dtype=np.int64)
            self._by_other[client] = idx
        return self._by_other[client]

    def sample(self, client: str, rng: np.random.Generator) -> np.ndarray:
        """随机抽一条**别家**的幅度谱。"""
        idx = self.others(client)
        return self.amps[int(rng.choice(idx))]


__all__ = ["AmplitudeBank", "amplitude_mix", "amplitude_of"]
