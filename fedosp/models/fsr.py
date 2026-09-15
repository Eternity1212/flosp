"""FSR：浅层频域风格校准（方案 4.3）。

核心假设：**幅度谱承载风格，相位谱承载语义结构。**
所以只对幅度做 instance normalization、相位原样保留，就能压掉相机/光照差异
而不破坏病灶的空间位置关系。

思路迁自 FedBCS (AAAI 2026)。原文作用在 CNN 特征图上，本文的适配工作是让它同时支持三种排布：

=================  ===========================  =========================================
排布                骨干                          FSR 怎么处理
=================  ===========================  =========================================
``tokens``          ViT / DINOv2 / RETFound      去掉 CLS，token 序列 reshape 成方形网格
``nhwc``            SwinV2                       转成 NCHW 再做 2D FFT
``nchw``            ResNet                       直接做 —— 这就是 FedBCS 原文的形式
=================  ===========================  =========================================

**已知局限**：224 输入的 ViT 只有 14x14 = 196 个频率 bin，比 CNN 特征图粗得多。
若消融 A1 显示收益 < 0.5 QWK 点，先试 384 输入（24x24 = 576 bin），见方案第 12 节 R1。
模块会在首次前向时把实际频率分辨率打进日志，方便核对。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .backbones import LAYOUT_NCHW, LAYOUT_NHWC, LAYOUT_TOKENS

LOGGER = logging.getLogger("fsr")


class FrequencyStyleRecalibration(nn.Module):
    """对浅层特征做频域幅度重标定。输入输出同形状。

    Args:
        dim: 特征通道数（ViT-L 是 1024；Swin 的浅层维度由适配层实测给出）。
        layout: ``tokens`` / ``nhwc`` / ``nchw``，见模块 docstring。
        num_prefix_tokens: 前缀 token 数（``tokens`` 排布下 CLS=1），这些 token 不参与 FFT。
        init_gate: 门控 logit 初值。0.0 → sigmoid=0.5，初始时归一化幅度与原幅度各半。
    """

    def __init__(
        self,
        dim: int,
        layout: str = LAYOUT_TOKENS,
        num_prefix_tokens: int = 1,
        init_gate: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.layout = layout
        self.num_prefix_tokens = num_prefix_tokens if layout == LAYOUT_TOKENS else 0
        self.eps = eps
        # 逐通道可学习门控 g = sigmoid(w)，全局共享、参与联邦聚合（ViT-L 只有 1024 个参数）
        self.gate_logit = nn.Parameter(torch.full((dim,), float(init_gate)))
        self._logged_resolution = False

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    # ------------------------------------------------------------------ #
    def _to_nchw(self, h: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple]:
        """统一转成 ``(B, C, H, W)``，同时返回还原需要的信息。"""
        if self.layout == LAYOUT_TOKENS:
            p = self.num_prefix_tokens
            prefix, patch = h[:, :p], h[:, p:]
            b, n, c = patch.shape
            side = int(round(n ** 0.5))
            if side * side != n:
                raise ValueError(
                    f"FSR 需要方形 patch 网格，收到 {n} 个 token（非完全平方数）。"
                    "非方形输入请改用 nhwc / nchw 排布的骨干。"
                )
            z = patch.transpose(1, 2).reshape(b, c, side, side)
            return z, prefix, (b, n, c)

        if self.layout == LAYOUT_NHWC:
            return h.permute(0, 3, 1, 2).contiguous(), None, h.shape

        return h, None, h.shape

    def _from_nchw(self, z: torch.Tensor, prefix, meta) -> torch.Tensor:
        if self.layout == LAYOUT_TOKENS:
            b, n, c = meta
            patch = z.reshape(b, c, n).transpose(1, 2)
            return torch.cat([prefix, patch], dim=1)
        if self.layout == LAYOUT_NHWC:
            return z.permute(0, 2, 3, 1).contiguous()
        return z

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z, prefix, meta = self._to_nchw(h)

        if not self._logged_resolution:
            LOGGER.info(
                "FSR 频率分辨率：%dx%d = %d 个 bin（layout=%s, dim=%d）"
                "%s",
                z.shape[-2], z.shape[-1], z.shape[-2] * z.shape[-1], self.layout, self.dim,
                "  ← 偏粗，若 A1 收益小先试 384 输入" if z.shape[-1] <= 14 else "",
            )
            self._logged_resolution = True

        # FFT 必须在 fp32 下做：bf16 自动混合精度里直接 fft 会数值不稳
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32 = z.float()
            spec = torch.fft.fft2(z32, norm="ortho")
            amp, pha = spec.abs(), torch.angle(spec)

            # 逐样本逐通道，在频率两维上做 instance norm
            mu = amp.mean(dim=(-2, -1), keepdim=True)
            sd = amp.std(dim=(-2, -1), keepdim=True)
            amp_norm = (amp - mu) / (sd + self.eps)
            # 拉回量纲，否则归一化幅度与原幅度差几个数量级，门控混合会被其中一支
            # 完全支配、梯度也容易爆。
            #
            # ⚠ 这里必须用**跨通道平均后**的统计量，不能用逐通道的 mu/sd：
            #     amp_norm * sd + mu == (amp-mu)/sd*sd + mu == amp
            # 用逐通道 sd/mu 会把归一化精确抵消回去，FSR 退化成恒等映射（曾经的 bug，
            # 见 test_fsr_gate_interpolates_between_identity_and_normalization）。
            # 相机/光照差异正是体现在**逐通道**的幅度增益上，所以跨通道拉平才是要的效果；
            # 保留逐样本维度是为了不引入 batch 依赖（推理时 batch=1 行为一致）。
            scale = sd.detach().mean(dim=1, keepdim=True)
            shift = mu.detach().mean(dim=1, keepdim=True)
            amp_norm = amp_norm * scale + shift

            g = self.gate.float().view(1, -1, 1, 1)
            amp_mix = g * amp_norm + (1.0 - g) * amp

            # ⚠ 幅度必须非负：归一化后 amp < mu 的 bin 会变成负数，而 torch.polar 对
            # 负幅度的处理等价于把相位旋转 π —— 那就直接违背了「相位原样保留」这个
            # FSR 的立论基础（曾经的 bug，见 test_fsr_preserves_phase_spectrum）。
            amp_mix = amp_mix.clamp_min(0.0)

            # 相位原样保留 —— 这是 FSR 不破坏病灶空间语义的关键
            z_new = torch.fft.ifft2(torch.polar(amp_mix, pha), norm="ortho").real

        return self._from_nchw(z_new.to(h.dtype), prefix, meta)

    @torch.no_grad()
    def amplitude_spectrum(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(FSR 前的幅度谱, FSR 后的幅度谱)``，两者都是 ``(B, C, H, W)``。

        只给论文图 F3 用：四个 client 各画一行，直观展示风格被压平。
        """
        z, _, _ = self._to_nchw(h)
        amp_before = torch.fft.fft2(z.float(), norm="ortho").abs()
        z_after, _, _ = self._to_nchw(self.forward(h))
        amp_after = torch.fft.fft2(z_after.float(), norm="ortho").abs()
        return amp_before, amp_after

    def extra_repr(self) -> str:
        return f"dim={self.dim}, layout={self.layout}, prefix={self.num_prefix_tokens}"


class IdentityFSR(nn.Module):
    """A1 消融用：结构占位但不做任何频域操作，保证流程与参数名完全一致。"""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        super().__init__()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h


def build_fsr(enabled: bool, dim: int, **kwargs) -> nn.Module:
    return FrequencyStyleRecalibration(dim, **kwargs) if enabled else IdentityFSR()
