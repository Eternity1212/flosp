"""双层原型库：浅层风格原型 + 深层等级原型（方案 4.4）。

两层原型的分工：

* ``shallow``：FSR 之后 patch token 的池化特征的类均值 —— 用来**跨 client 对齐风格**
* ``deep``：CLS 特征的类均值 —— 用来**承载 0→4 的有序几何**

原型都用 EMA 累积（动量 0.9），而不是每步重算。原因：IDRiD 一个 batch 可能只有
1~2 张 grade 1，直接用 batch 均值当原型会剧烈抖动。

原型以 ``buffer`` 形式存在，不是 ``Parameter`` —— 它们由统计量更新、由服务器聚合，
不参与本地梯度下降。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("prototypes")


class PrototypeBank(nn.Module):
    """维护 ``(num_classes, dim)`` 的一组原型，支持 EMA 本地更新与服务器覆写。

    Args:
        num_classes: DR 等级数，5。
        dim: 特征维度。
        momentum: EMA 动量，越大越平滑。
        normalize: 是否在更新与比较前做 L2 归一化（ordinal margin 依赖归一化后的尺度）。
    """

    def __init__(
        self,
        num_classes: int = 5,
        dim: int = 1024,
        momentum: float = 0.9,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.momentum = momentum
        self.normalize = normalize
        self.register_buffer("proto", torch.zeros(num_classes, dim))
        # seen[c] = 该 client 是否见过类别 c；上传给服务器决定「哪些 client 参与该类平均」
        self.register_buffer("seen", torch.zeros(num_classes, dtype=torch.bool))
        # ---- 精度加权聚合所需的两个统计量（各 C 个标量，C=5 时共 10 个 float = 40 B）----
        # var_within[c]：类内**每维**特征方差 s^2_c 的 EMA
        self.register_buffer("var_within", torch.zeros(num_classes))
        # var_unit[c]：原型估计量的方差除以 s^2，即 Var(proto[c]) / s^2 —— 等价于 1/n_eff。
        # 不能直接用累计样本数当 n：EMA(动量 m) 的有效窗口只有约 1/(1-m) 个 batch，
        # 用总样本数会把原型的精度高估一个数量级。这里按 EMA 递推精确地累计：
        #     首次赋值：      V = 1 / b_t
        #     之后每步：      V = m^2 * V + (1-m)^2 / b_t
        # （b_t 是本 batch 中该类的样本数）
        self.register_buffer("var_unit", torch.zeros(num_classes))
        self.register_buffer("count", torch.zeros(num_classes))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update(self, feat: torch.Tensor, labels: torch.Tensor) -> None:
        """用一个 batch 的特征做 EMA 更新。``feat: (B, D)``, ``labels: (B,)``。"""
        f = F.normalize(feat.detach().float(), dim=-1) if self.normalize else feat.detach().float()
        m = self.momentum
        for c in labels.unique():
            c_int = int(c)
            mask = labels == c
            b = int(mask.sum())
            if b == 0:
                continue
            fc = f[mask]
            batch_mean = fc.mean(dim=0)
            # 类内每维方差（b=1 时无法估计，沿用历史值）
            batch_var = float(fc.var(dim=0, unbiased=True).mean()) if b > 1 else None

            if bool(self.seen[c_int]):
                self.proto[c_int].mul_(m).add_(batch_mean, alpha=1 - m)
                self.var_unit[c_int] = m * m * self.var_unit[c_int] + (1 - m) ** 2 / b
                if batch_var is not None:
                    self.var_within[c_int] = m * self.var_within[c_int] + (1 - m) * batch_var
            else:
                # 首次见到该类：直接赋值，避免从 0 向量慢慢爬
                self.proto[c_int].copy_(batch_mean)
                self.var_unit[c_int] = 1.0 / b
                self.var_within[c_int] = batch_var if batch_var is not None else 0.0
                self.seen[c_int] = True
            self.count[c_int] += b
        if self.normalize:
            self.proto.copy_(F.normalize(self.proto, dim=-1))

    @torch.no_grad()
    def load_global(self, global_proto: torch.Tensor) -> None:
        """服务器下发的全局原型覆写本地（只覆写服务器实际有值的类）。"""
        valid = global_proto.abs().sum(dim=-1) > 0
        self.proto[valid] = global_proto[valid].to(self.proto.device, self.proto.dtype)

    def export(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """导出给服务器：``(proto, seen)``。seen 决定 client 等权平均的分母。"""
        return self.proto.detach().cpu().clone(), self.seen.detach().cpu().clone()

    @torch.no_grad()
    def sampling_variance(self) -> torch.Tensor:
        """原型估计量的**每维**抽样方差 ``v_c = s_c^2 / n_eff,c``，形状 ``(C,)``。

        这是精度加权聚合（``mode='precision'``）的输入，也是随机效应模型里的 ``v_k``。
        没见过的类别返回 ``inf``，这样精度加权会自动给它零权重。

        额外上传成本：C 个 float32 = 20 B（C=5），相对 2.31 MB 参数完全可忽略。
        """
        v = self.var_within * self.var_unit
        v = torch.where(self.seen, v, torch.full_like(v, float("inf")))
        # 方差估计为 0（例如某类只在 b=1 的 batch 里出现过）会让权重变成 inf。
        # 退化成该类的一个极小正数，语义是「精度很高但不是无限高」。
        floor = 1e-8
        return torch.clamp(v, min=floor).detach().cpu().clone()

    def distances(self, feat: torch.Tensor) -> torch.Tensor:
        """样本到各原型的平方欧氏距离，``(B, C)``。

        用 :func:`fedosp.losses.squared_distances` 而不是 ``torch.cdist``：
        后者的反向在 MPS 上未实现，会让 Apple Silicon 上的原型策略直接崩。
        """
        from ..losses import squared_distances  # 延迟导入，避免模块级循环依赖

        f = F.normalize(feat, dim=-1) if self.normalize else feat
        return squared_distances(f, self.proto)

    def is_ready(self) -> bool:
        return bool(self.seen.any())

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, dim={self.dim}, "
            f"momentum={self.momentum}, normalize={self.normalize}"
        )


PROTO_AGG_MODES = ("client_equal", "sqrt", "sample", "precision")


@torch.no_grad()
def aggregate_prototypes(
    protos: Dict[str, torch.Tensor],
    seens: Dict[str, torch.Tensor],
    sizes: Optional[Dict[str, int]] = None,
    mode: str = "client_equal",
    sampling_vars: Optional[Dict[str, torch.Tensor]] = None,
    tau2: Optional[float] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """服务器端原型聚合（方案 4.5，对应消融 A6）。

    Args:
        protos: ``client -> (C, D)``
        seens: ``client -> (C,) bool``，标记该 client 是否有该类样本
        sizes: ``client -> n_k``，``sample`` / ``sqrt`` 模式需要
        mode: 见下表
        sampling_vars: ``client -> (C,)`` 每类的原型抽样方差，``precision`` 模式需要。
            由 :meth:`PrototypeBank.sampling_variance` 提供。
        tau2: 显式指定 between-client 方差（A6 的 tau^2 扫描用）。``None`` 则逐类用
            DerSimonian–Laird 估计。
        diagnostics: 传入一个 dict 则**原地填入** ``proto_weights`` 与 ``tau2_per_class``，
            供 ``result.json`` 记录。不传则不计算。

    Returns:
        ``(C, D)`` 全局原型。没有任何 client 拥有的类别保持为 0 向量。

    四种模式
    --------
    ==============  ===========================================  ===================
    mode            权重                                          对应
    ==============  ===========================================  ===================
    ``sample``      :math:`w_k \\propto n_k`                      FedProto (B8)
    ``sqrt``        :math:`w_k \\propto \\sqrt{n_k}`               折中
    ``client_equal`` :math:`w_k = 1/K`                            方案 v1.0
    ``precision``   :math:`w_k \\propto 1/(\\tau^2 + v_k)`         **本文 C2**
    ==============  ===========================================  ===================

    ``precision`` 是随机效应模型下 MSE 最优的权重，且 ``sample`` 与 ``client_equal``
    分别是它在 :math:`\\tau^2=0` 与 :math:`\\tau^2 \\to \\infty` 时的极限特例
    （推导见 :mod:`fedosp.fed.diagnostics`）。

    只对「真的有该类样本」的 client 求平均，避免没见过 grade 1 的 client 拿 0 向量
    拉低均值。
    """
    from ..fed.diagnostics import precision_weights

    keys = list(protos.keys())
    if not keys:
        raise ValueError("aggregate_prototypes 收到空的 protos")
    if mode not in PROTO_AGG_MODES:
        raise ValueError(f"未知原型聚合模式: {mode}，可选 {PROTO_AGG_MODES}")
    if mode == "precision" and sampling_vars is None:
        raise ValueError(
            "mode='precision' 需要提供 sampling_vars"
            "（由 PrototypeBank.sampling_variance() 得到）"
        )

    c, d = protos[keys[0]].shape
    out = torch.zeros(c, d, dtype=torch.float32)
    diag_w: Dict[int, list] = {}
    diag_tau2: Dict[int, float] = {}

    for cls in range(c):
        owners = [k for k in keys if bool(seens[k][cls])]
        if not owners:
            continue
        stacked = torch.stack([protos[k][cls].float() for k in owners])

        if mode == "client_equal":
            w = torch.ones(len(owners))
        elif mode == "sample":
            if sizes is None:
                raise ValueError("mode='sample' 需要提供 sizes")
            w = torch.tensor([float(sizes[k]) for k in owners])
        elif mode == "sqrt":
            if sizes is None:
                raise ValueError("mode='sqrt' 需要提供 sizes")
            w = torch.tensor([float(sizes[k]) for k in owners]).sqrt()
        else:  # precision
            v = np.array([float(sampling_vars[k][cls]) for k in owners], dtype=np.float64)
            w_np, tau2_c = precision_weights(stacked.numpy(), v, tau2=tau2)
            w = torch.from_numpy(w_np).float()
            diag_tau2[cls] = tau2_c

        w = w / w.sum()
        out[cls] = (w[:, None] * stacked).sum(dim=0)
        diag_w[cls] = [float(x) for x in w]

    out = F.normalize(out, dim=-1)
    out[out.isnan()] = 0.0
    n_valid = int((out.abs().sum(dim=-1) > 0).sum())
    LOGGER.debug("原型聚合 mode=%s，有效类别 %d/%d", mode, n_valid, c)

    if diagnostics is not None:
        diagnostics["proto_weights"] = diag_w
        diagnostics["proto_agg_mode"] = mode
        if diag_tau2:
            diagnostics["tau2_per_class"] = diag_tau2
    return out
