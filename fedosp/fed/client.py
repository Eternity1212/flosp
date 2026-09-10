"""本地 client：训练一轮固定步数、更新原型、导出上传包、做评估。

对应方案 4.6（本地步数均衡）、4.7（本地 LayerNorm）、第 5 节（损失）。

一个 client 一轮的流程::

    load_global(下发的 LoRA/head/gate + 全局原型)   # LayerNorm 保持本地不动
    for step in 1..S_k:                             # S_k = clip(1.275*sqrt(n_k), 20, 200)
        前向 -> 算 6 项损失 -> 反传
        用 batch 特征 EMA 更新本地原型
    导出 shared + 双原型 + seen 掩码 + 平均不确定性
"""

from __future__ import annotations

import logging
import math
import time
from itertools import cycle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..losses import FedOSPLoss, LossWeights, edl_mse_loss, edl_uncertainty, fedprox_term
from ..metrics import evaluate_predictions
from ..models.retfound_lora import FedOSPNet
from .strategies import ClientUpdate, ServerState

LOGGER = logging.getLogger("client")


class LocalClient:
    """封装一个中心的模型副本、数据与优化器。

    Args:
        name: client 名（eyepacs / aptos / ddr / idrid）。
        model: 该 client 的模型副本。**LayerNorm 是它私有的，不要跨 client 复用同一对象。**
        loaders: ``{'train': ..., 'val': ..., 'test': ...}``。
        loss_weights: 各项 λ。
        lr / weight_decay: AdamW 超参。
        device: 训练设备。
    """

    def __init__(
        self,
        name: str,
        model: FedOSPNet,
        loaders: Dict[str, DataLoader],
        loss_weights: Optional[LossWeights] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        device: str = "cpu",
        grad_clip: float = 1.0,
        amp: bool = False,
    ) -> None:
        self.name = name
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.grad_clip = grad_clip
        self.amp = amp and device.startswith("cuda")

        train_ds = loaders["train"].dataset
        self.n_train = len(train_ds)
        counts = train_ds.class_counts()
        # 类别权重必须按**本地**分布算，这是联邦设定下的正确做法
        self.criterion = FedOSPLoss(counts, loss_weights or LossWeights()).to(device)
        LOGGER.info("[%s] n_train=%d 本地分布=%s", name, self.n_train, counts.astype(int).tolist())

        self.lr = lr  # SCAFFOLD 的 Option-II 公式需要用到 eta_l
        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self._iter = None
        # SCAFFOLD 的本地 control variate c_i，跨轮持久保留
        self.control: Dict[str, torch.Tensor] = {}
        self._round_start: Dict[str, torch.Tensor] = {}
        # 只对会被聚合的参数做 control variate：本地 LayerNorm 从不聚合，无 drift 可言
        self._scaffold_keys = set(self.model.shared_state_dict().keys())

    # ------------------------------------------------------------------ #
    def _next_batch(self):
        """无限取 batch：本地步数是按 sqrt 规则定的，不是按 epoch。"""
        if self._iter is None:
            self._iter = iter(self.loaders["train"])
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loaders["train"])
            return next(self._iter)

    def load_from_server(self, state: ServerState) -> None:
        """下发全局参数。注意 LayerNorm 不在 ``state.shared`` 里，本地值原样保留。"""
        if state.shared:
            self.model.load_shared_state_dict(
                {k: v.to(self.device) for k, v in state.shared.items()}
            )
        if state.deep_proto is not None:
            self.model.deep_proto.load_global(state.deep_proto)
        if state.shallow_proto is not None:
            self.model.shallow_proto.load_global(state.shallow_proto)
        self._global_snapshot = (
            {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if state.shared else None
        )
        # SCAFFOLD Option-II 要用「本轮起点 x」和「本地训练终点 y_i」之差算新的 c_i
        if state.control is not None:
            self._round_start = {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()
                if k in self._scaffold_keys
            }
            self._c_global = state.control

    # ------------------------------------------------------------------ #
    def local_train(self, steps: int, config: Optional[Dict[str, Any]] = None) -> ClientUpdate:
        """跑 ``steps`` 步本地训练，返回上传包。"""
        cfg = config or {}
        self.model.train()
        t0 = time.time()

        use_proto = bool(cfg.get("use_proto_loss", False))
        use_margin = bool(cfg.get("use_ordinal_margin", False))
        evidential = bool(cfg.get("evidential", False))
        mu = float(cfg.get("fedprox_mu", 0.0))

        base_margin = self.criterion.w.ordinal_margin
        if not use_margin:
            self.criterion.w.ordinal_margin = 0.0

        agg: Dict[str, float] = {}
        n_batches = 0
        uncertainties = []

        for _ in range(steps):
            batch = self._next_batch()
            if len(batch) == 3:
                x, x_aug, y = batch
                x_aug = x_aug.to(self.device, non_blocking=True)
            else:
                x, y = batch
                x_aug = None
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                out = self.model(x)
                out_aug = self.model(x_aug) if x_aug is not None else None

                if evidential:
                    # B9 FedUAA-style：本地换成 evidential 头
                    loss = edl_mse_loss(
                        out.logits, y, anneal=float(cfg.get("edl_anneal", 1.0))
                    )
                    parts = {"edl": float(loss.detach())}
                    uncertainties.append(float(edl_uncertainty(out.logits).mean().detach()))
                else:
                    deep_p = self.model.deep_proto.proto if use_proto else None
                    shallow_p = (
                        self.model.shallow_proto.proto
                        if use_proto and self.model.cfg.use_shallow_proto
                        else None
                    )
                    loss, parts = self.criterion(
                        out, y, deep_protos=deep_p, shallow_protos=shallow_p, out_aug=out_aug
                    )

                if mu > 0 and getattr(self, "_global_snapshot", None):
                    loss = loss + fedprox_term(self.model, self._global_snapshot, mu)

            if not torch.isfinite(loss):
                LOGGER.error("[%s] 第 %d 步 loss 非有限，跳过该步", self.name, n_batches)
                continue

            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.grad_clip)
                # 梯度修正必须在 step 之前、且在 unscale 之后
                if cfg.get("scaffold"):
                    self._scaffold_correct_grad()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.grad_clip)
                if cfg.get("scaffold"):
                    self._scaffold_correct_grad()
                self.optimizer.step()

            self.model.update_prototypes(out, y)

            n_batches += 1
            agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v

        self.criterion.w.ordinal_margin = base_margin
        metrics = {k: v / max(n_batches, 1) for k, v in agg.items()}
        metrics["steps"] = n_batches
        metrics["sec"] = time.time() - t0

        shallow_p, shallow_seen = self.model.shallow_proto.export()
        deep_p, deep_seen = self.model.deep_proto.export()

        control_delta = (
            self._scaffold_update_control(n_batches) if cfg.get("scaffold") else None
        )

        return ClientUpdate(
            client=self.name,
            n=self.n_train,
            shared=self.model.shared_state_dict(),
            metrics=metrics,
            shallow_proto=shallow_p,
            shallow_seen=shallow_seen,
            deep_proto=deep_p,
            deep_seen=deep_seen,
            control_delta=control_delta,
            uncertainty=float(np.mean(uncertainties)) if uncertainties else None,
        )

    # -------------------------- SCAFFOLD -------------------------- #
    def _scaffold_correct_grad(self) -> None:
        """按原论文修正梯度：``g_i <- g_i - c_i + c``。

        论文的本地更新是 ``y_i <- y_i - eta_l * (g_i(y_i) - c_i + c)``。
        因为优化器统一做 ``p -= eta * p.grad``，所以等价于给 ``p.grad`` 加上 ``c - c_i``。

        **这一步必须在 optimizer.step() 之前**。之前的实现是在 step 之后记了个梯度滑动
        平均，从头到尾没有修正过任何一次更新，等于根本没开 SCAFFOLD。
        """
        c_global = getattr(self, "_c_global", None) or {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad or p.grad is None or name not in self._scaffold_keys:
                    continue
                cg = c_global.get(name)
                ci = self.control.get(name)
                if cg is None and ci is None:
                    continue  # 第一轮 c 与 c_i 都是 0，无需修正
                delta = torch.zeros_like(p.grad)
                if cg is not None:
                    delta += cg.to(p.device, p.dtype)
                if ci is not None:
                    delta -= ci.to(p.device, p.dtype)
                p.grad.add_(delta)

    def _scaffold_update_control(self, steps: int) -> Dict[str, torch.Tensor]:
        """原论文 Option-II：``c_i^+ = (x - y_i) / (K * eta_l)``，返回要上传的 ``dc_i``。

        ``x`` 是本轮下发的全局参数，``y_i`` 是本地训练 K 步后的参数。
        Option-II 不需要额外过一遍完整数据集（那是 Option-I），代价是估计噪声更大 ——
        这是 SCAFFOLD 论文自己推荐的实用做法。
        """
        if not self._round_start or steps <= 0:
            return {}
        deltas: Dict[str, torch.Tensor] = {}
        current = self.model.state_dict()
        denom = max(steps * self.lr, 1e-12)
        with torch.no_grad():
            for name, x in self._round_start.items():
                y = current.get(name)
                if y is None:
                    continue
                new_ci = ((x.float() - y.detach().float()) / denom).cpu()
                old_ci = self.control.get(name)
                deltas[name] = new_ci - old_ci if old_ci is not None else new_ci.clone()
                self.control[name] = new_ci
        return deltas

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, split: str = "val", loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """在指定 split 上评估，返回 :func:`fedosp.metrics.evaluate_predictions` 的结果。"""
        loader = loader or self.loaders.get(split)
        if loader is None:
            return {}
        self.model.eval()
        probs, labels = [], []
        for batch in loader:
            x, y = (batch[0], batch[-1])
            x = x.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                logits = self.model(x).logits
            probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
            labels.append(y.numpy())
        if not probs:
            return {}
        res = evaluate_predictions(np.concatenate(labels), np.concatenate(probs))
        LOGGER.debug("[%s/%s] %s", self.name, split,
                     {k: round(v, 4) for k, v in res.items()})
        return res

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """返回 ``(labels, probs)``，供 Messidor-2 外测与统计检验使用。"""
        self.model.eval()
        probs, labels = [], []
        for batch in loader:
            x, y = (batch[0], batch[-1])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                logits = self.model(x.to(self.device)).logits
            probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
            labels.append(y.numpy())
        return np.concatenate(labels), np.concatenate(probs)
