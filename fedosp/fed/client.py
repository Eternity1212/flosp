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
from contextlib import contextmanager
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
        #: 已完成的本地训练轮数。MOON 用它判断"有没有上一轮的本地模型"。
        self.rounds_done = 0
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

    def load_from_server(self, state: ServerState, config: Optional[Dict[str, Any]] = None) -> None:
        """下发全局参数。注意 LayerNorm 不在 ``state.shared`` 里，本地值原样保留。

        Args:
            config: 本轮的策略配置。FedALA 的全部机制都在**下发这一步**，
                所以它必须能在这里看到配置 —— 这也是这个参数存在的唯一理由。
        """
        cfg = config or {}
        if state.shared:
            # MOON 需要在本地参数被覆盖**之前**记下上一轮的本地模型
            if cfg.get("moon"):
                self._moon_prev = self._snapshot_shared() if self.rounds_done else None
                self._moon_global = {
                    k: v.detach().clone() for k, v in state.shared.items()
                }
            if cfg.get("ala"):
                # FedALA：元素级插值代替直接覆盖
                self._ala_download(state.shared, cfg)
            else:
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
        #: 下发的全局 shared 参数原件。Ditto 的 prox 项要朝它正则化，
        #: 不能用 ``_global_snapshot``：FedALA 下发后模型里装的是插值结果而非 w^t。
        self._server_shared = (
            {k: v.detach().clone() for k, v in state.shared.items()}
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
        moon_mu = float(cfg.get("moon_mu", 0.0)) if cfg.get("moon") else 0.0
        moon_tau = float(cfg.get("moon_tau", 0.5))

        base_margin = self.criterion.w.ordinal_margin
        if not use_margin:
            self.criterion.w.ordinal_margin = 0.0

        # q-FedAvg：必须在**本地训练开始之前**评 F_k(w^t)
        loss_at_global = (
            self.loss_at_current_params() if cfg.get("report_loss_at_global") else None
        )

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

            # MOON 的两次冻结前向必须在建图之前做完（见 _moon_reference_feats）
            moon_refs = self._moon_reference_feats(x) if moon_mu > 0 else None

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

                if moon_mu > 0:
                    l_con = self._moon_loss(out.deep_feat, moon_refs, moon_tau)
                    parts["moon_con"] = l_con.detach()
                    loss = loss + moon_mu * l_con

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
                # 必须转成 float：parts 里是 tensor（有的还带计算图），
                # 累加后 metrics 就成了 tensor，既与声明的 Dict[str, float] 不符，
                # 也会让 result.json 的序列化直接抛异常。
                agg[k] = agg.get(k, 0.0) + (
                    float(v.detach()) if torch.is_tensor(v) else float(v)
                )

        self.criterion.w.ordinal_margin = base_margin
        metrics = {k: v / max(n_batches, 1) for k, v in agg.items()}
        metrics["steps"] = n_batches
        metrics["sec"] = time.time() - t0

        # ---- Ditto：全局模型 w 训完之后，再训个人模型 v（评估会用 v）----
        ditto_steps = 0
        if cfg.get("ditto"):
            if self._server_shared is None:
                LOGGER.warning("[%s] Ditto 第一轮还没有全局参数，本轮跳过个人模型", self.name)
            else:
                ditto_steps = self._ditto_train(
                    n_batches, self._server_shared, float(cfg.get("ditto_lambda", 0.1))
                )
            metrics["ditto_steps"] = ditto_steps
        if cfg.get("ala"):
            metrics["ala_mean_w"] = float(getattr(self, "_ala_mean_w", float("nan")))
        if loss_at_global is not None:
            metrics["loss_at_global"] = loss_at_global

        shallow_p, shallow_seen = self.model.shallow_proto.export()
        deep_p, deep_seen = self.model.deep_proto.export()

        control_delta = (
            self._scaffold_update_control(n_batches) if cfg.get("scaffold") else None
        )
        self.rounds_done += 1

        return ClientUpdate(
            client=self.name,
            n=self.n_train,
            shared=self.model.shared_state_dict(),
            metrics=metrics,
            shallow_proto=shallow_p,
            shallow_seen=shallow_seen,
            deep_proto=deep_p,
            deep_seen=deep_seen,
            shallow_var=self.model.shallow_proto.sampling_variance(),
            deep_var=self.model.deep_proto.sampling_variance(),
            control_delta=control_delta,
            uncertainty=float(np.mean(uncertainties)) if uncertainties else None,
            loss_at_global=loss_at_global,
            # MOON 的 3 次前向、Ditto 的第二阶段都是真实计算代价，T5 按这个数报
            compute_steps=n_batches + ditto_steps
            + (2 * n_batches if moon_mu > 0 else 0),
        )

    # ---------------------------- MOON (B12) ---------------------------- #
    def _snapshot_shared(self) -> Dict[str, torch.Tensor]:
        """复制一份 shared 参数（LoRA+head+gate，约 0.7M 个数，~2.8 MB）。"""
        return {k: v.detach().clone() for k, v in self.model.shared_state_dict().items()}

    def _feat_under(self, params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """把 ``params`` 临时换进模型、做一次**无梯度**前向、取深层特征，然后换回来。

        MOON / Ditto 都需要"在另一组参数下前向"。直接 ``deepcopy(model)`` 要多占
        两份 ViT-L（约 2.4 GB），而**骨干是冻结的、三份模型只有 LoRA/head/gate 不同**，
        所以只换这 0.7M 个参数就完全等价，显存开销是 O(1)。

        ``try/finally`` 是必须的：中途抛异常而没换回参数，会让模型静默地带着
        别人的 LoRA 继续训练，且不会有任何报错。
        """
        backup = self._snapshot_shared()
        try:
            self.model.load_shared_state_dict(
                {k: v.to(self.device) for k, v in params.items()}
            )
            with torch.no_grad():
                return self.model(x).deep_feat.detach()
        finally:
            self.model.load_shared_state_dict(backup)

    def _moon_reference_feats(self, x: torch.Tensor):
        """取全局模型与上一轮本地模型在 ``x`` 上的深层特征。返回 ``None`` 表示本轮不算对比项。

        ⚠ **必须在可训练前向之前调用**。这里会用 ``load_shared_state_dict`` 原地覆写
        参数，而原地写会递增参数张量的 version counter。如果此时可训练前向的计算图
        已经建好，backward 会直接抛

            "one of the variables needed for gradient computation has been
             modified by an inplace operation"

        把参数换入换出全部挪到建图之前，版本号在 forward/backward 之间就是稳定的。
        （这个坑踩过一次：把对比项写在损失里、顺手调用换参数，MOON 直接跑不起来。）

        第一轮没有"上一轮的本地模型"，返回 ``None``：若拿全局模型充当 prev，
        则 ``s_p ≡ s_g``、损失恒为 ``log 2`` 且梯度恒为 0，与返回 None 等价但白费两次前向。
        """
        prev = getattr(self, "_moon_prev", None)
        glob = getattr(self, "_moon_global", None)
        if prev is None or glob is None:
            return None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            z_g = self._feat_under(glob, x)
            z_p = self._feat_under(prev, x)
        return F.normalize(z_g.float(), dim=-1), F.normalize(z_p.float(), dim=-1)

    @staticmethod
    def _moon_loss(feat: torch.Tensor, refs, tau: float) -> torch.Tensor:
        """MOON 的模型级对比损失。

        .. math:: \\ell = -\\log\\frac{e^{s_g/T}}{e^{s_g/T}+e^{s_p/T}}
                        = \\mathrm{softplus}\\big((s_p-s_g)/T\\big)

        右边那个恒等变形不是为了好看，是为了**数值稳定**：按左式直接写，
        ``exp(s/T)`` 在 ``T=0.1`` 时会在 fp16 下溢出。``softplus`` 全程有界。
        """
        if refs is None:
            return feat.new_zeros(())
        z_g, z_p = refs
        z = F.normalize(feat.float(), dim=-1)
        s_g = (z * z_g.to(z.device)).sum(-1)
        s_p = (z * z_p.to(z.device)).sum(-1)
        return F.softplus((s_p - s_g) / max(tau, 1e-6)).mean()

    # ---------------------------- Ditto (B15) ---------------------------- #
    def _ditto_train(self, steps: int, w_global: Dict[str, torch.Tensor], lam: float) -> int:
        """训练个人模型 ``v``：``min F_k(v) + (lambda/2)||v - w^t||^2``。

        ``w^t`` 是**本轮下发的全局参数**（不是本地训练后的 ``w_k``）—— 原论文
        Algorithm 1 就是朝下发的全局模型正则化。

        做法：把 ``v`` 换进模型、用**独立的优化器** ``_ditto_opt`` 训练、再换回来。
        独立优化器是必须的：AdamW 的动量按参数对象身份存储，而这里 ``w`` 和 ``v``
        复用同一批参数对象，共用优化器会把两个模型的动量搅在一起。
        """
        if not hasattr(self, "_ditto_v") or self._ditto_v is None:
            # 第一轮：个人模型从下发的全局模型出发
            self._ditto_v = {k: v.detach().clone() for k, v in w_global.items()}
        if not hasattr(self, "_ditto_opt"):
            self._ditto_opt = torch.optim.AdamW(
                self.model.trainable_parameters(),
                lr=self.lr, weight_decay=0.0,   # prox 项已经在做正则，别再叠 wd
            )

        w_keep = self._snapshot_shared()
        done = 0
        try:
            self.model.load_shared_state_dict(
                {k: v.to(self.device) for k, v in self._ditto_v.items()}
            )
            self.model.train()
            for _ in range(steps):
                batch = self._next_batch()
                x, y = batch[0].to(self.device), batch[-1].to(self.device)
                self._ditto_opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    out = self.model(x)
                    loss, _ = self.criterion(out, y)
                    loss = loss + fedprox_term(self.model, w_global, lam)
                if not torch.isfinite(loss):
                    LOGGER.error("[%s] Ditto 个人模型第 %d 步 loss 非有限，跳过", self.name, done)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), self.grad_clip
                )
                self._ditto_opt.step()
                done += 1
            # 把训练后的 v 存回来
            self._ditto_v = self._snapshot_shared()
        finally:
            self.model.load_shared_state_dict(w_keep)
        LOGGER.debug("[%s] Ditto 个人模型训练 %d 步（lambda=%.3g）", self.name, done, lam)
        return done

    # ---------------------------- FedALA (B13) ---------------------------- #
    def _ala_download(self, w_global: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> None:
        """FedALA 的自适应下发：``w <- w_local + W ⊙ (w_global - w_local)``。

        ``W`` 是逐元素的、在本地数据上学出来的、**跨轮持久**的插值系数。

        梯度的算法：因为 :math:`\\hat w = w_l + W\\odot(w_g-w_l)`，链式法则给出

        .. math:: \\frac{\\partial L}{\\partial W}
                  = \\frac{\\partial L}{\\partial \\hat w}\\odot (w_g - w_l)

        所以不需要把 ``W`` 做成真的 ``nn.Parameter`` 去建图，直接拿参数的梯度
        乘上 ``(w_g - w_l)`` 就是 ``W`` 的梯度 —— 这也是原论文实现的做法。

        分层策略（原论文的 ``p``）：只有**最高的 ``ala_last_n`` 层**做插值，
        更低的层直接整体覆盖（等价于 ``W=1``）。理由是低层学通用特征、
        本地化没有收益，而且全模型学 ``W`` 会显著变慢。
        """
        w_local = self._snapshot_shared()
        if not hasattr(self, "_ala_W"):
            self._ala_W = {}

        keys = sorted(w_global.keys())
        # "高层" = 名字里 block 序号最大的那几层 + head。用 shared_state_dict 的
        # key 顺序做近似分层：head 永远算高层。
        n_last = int(cfg.get("ala_last_n", 4))
        block_ids = sorted({self._block_id(k) for k in keys if self._block_id(k) >= 0})
        hi_blocks = set(block_ids[-n_last:]) if block_ids else set()
        ala_keys = [
            k for k in keys
            if k.startswith("head.") or self._block_id(k) in hi_blocks
        ]
        low_keys = [k for k in keys if k not in set(ala_keys)]

        # 低层：直接覆盖
        self.model.load_shared_state_dict(
            {k: w_global[k].to(self.device) for k in low_keys}
        )
        if not ala_keys:
            return

        # 高层：先用上一轮的 W 初始化，再在本地数据上更新 W
        diff = {k: (w_global[k].to(self.device).float() - w_local[k].to(self.device).float())
                for k in ala_keys}
        for k in ala_keys:
            if k not in self._ala_W:
                self._ala_W[k] = torch.ones_like(diff[k])

        def apply_W() -> None:
            self.model.load_shared_state_dict(
                {k: (w_local[k].to(self.device).float() + self._ala_W[k] * diff[k])
                 for k in ala_keys}
            )

        apply_W()
        iters = int(cfg.get("ala_iters", 5))
        # 第一轮 W 从全 1 出发、离最优最远，原论文也是第一轮多训几轮直到收敛
        if int(cfg.get("ala_round", 0)) == 0:
            iters = max(iters, 20)
        lr_w = float(cfg.get("ala_lr", 0.1))

        name_of = self._shared_param_objects()
        self.model.train()
        for it in range(iters):
            batch = self._next_batch()
            x, y = batch[0].to(self.device), batch[-1].to(self.device)
            self.model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                loss, _ = self.criterion(self.model(x), y)
            if not torch.isfinite(loss):
                LOGGER.error("[%s] ALA 第 %d 次迭代 loss 非有限，提前停止", self.name, it)
                break
            loss.backward()
            with torch.no_grad():
                for k in ala_keys:
                    p = name_of.get(k)
                    if p is None or p.grad is None:
                        continue
                    # dL/dW = dL/dw_hat * (w_g - w_l)
                    self._ala_W[k] -= lr_w * p.grad.float() * diff[k]
                    self._ala_W[k].clamp_(0.0, 1.0)
            apply_W()
        self.model.zero_grad(set_to_none=True)
        mean_w = float(np.mean([float(v.mean()) for v in self._ala_W.values()]))
        LOGGER.debug(
            "[%s] ALA 下发完成：%d 个 key 参与插值（共 %d），W 均值 %.3f",
            self.name, len(ala_keys), len(keys), mean_w,
        )
        self._ala_mean_w = mean_w

    @staticmethod
    def _block_id(key: str) -> int:
        """从参数名里抠出 transformer block 序号，抠不到返回 -1。"""
        import re
        m = re.search(r"\.blocks\.(\d+)\.", key)
        return int(m.group(1)) if m else -1

    def _shared_param_objects(self) -> Dict[str, torch.nn.Parameter]:
        """``shared_state_dict`` 的 key -> 对应的 ``nn.Parameter`` 对象。

        ALA 需要按 shared key 去取 ``p.grad``，而 ``shared_state_dict`` 的 key
        与 ``named_parameters`` 的 key 可能不完全一致，这里建一次映射并缓存。
        """
        if getattr(self, "_shared_objs", None) is None:
            shared_keys = set(self.model.shared_state_dict().keys())
            self._shared_objs = {
                n: p for n, p in self.model.named_parameters() if n in shared_keys
            }
            missing = shared_keys - set(self._shared_objs)
            if missing:
                LOGGER.warning(
                    "[%s] 有 %d 个 shared key 找不到对应参数对象（如 %s），"
                    "FedALA 不会对它们插值",
                    self.name, len(missing), sorted(missing)[:3],
                )
        return self._shared_objs

    # -------------------------- q-FedAvg (B14) -------------------------- #
    @torch.no_grad()
    def loss_at_current_params(self, max_batches: int = 20) -> float:
        """在**当前**参数（= 刚下发的 ``w^t``）处评估本地训练目标 ``F_k(w^t)``。

        q-FedAvg 的权重是 ``F_k(w^t)^q``，必须是**训练前**在全局参数处的损失。
        用训练过程的平均损失代替是错的：那个值已经被本地更新推低了，而且各 client
        被推低的程度不同，公平性权重会整体失真。

        ``max_batches`` 是精度与代价的折中：完整过一遍 EyePACS 要 24.6k 张图，
        而 q-FedAvg 只需要一个标量。20 个 batch（~640 张）的标准误差约为
        单 batch 的 1/4.5，对 ``F^q`` 已经足够稳。这是相对原文的一处近似，需写明。
        """
        self.model.eval()
        total, n = 0.0, 0
        for _ in range(max_batches):
            batch = self._next_batch()
            x, y = batch[0].to(self.device), batch[-1].to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                loss, _ = self.criterion(self.model(x), y)
            if torch.isfinite(loss):
                total += float(loss)
                n += 1
        self.model.train()
        return total / max(n, 1)

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
    @contextmanager
    def personal_model(self):
        """进入这个上下文时，模型里装的是 Ditto 的个人模型 ``v``；退出时换回 ``w``。

        Ditto 的全部价值都体现在"**用 v 评估**"上。如果评估仍然用 w，Ditto 就精确
        退化成 FedAvg，而主表会把这显示成"Ditto 在本任务上无效"—— 一个看不出来的假结论。
        把切换做成上下文管理器，是为了让"忘记换回来"不可能发生。

        非 Ditto 策略下 ``_ditto_v`` 不存在，此时什么都不做（no-op）。
        """
        v = getattr(self, "_ditto_v", None)
        if v is None:
            yield False
            return
        backup = self._snapshot_shared()
        try:
            self.model.load_shared_state_dict(
                {k: t.to(self.device) for k, t in v.items()}
            )
            yield True
        finally:
            self.model.load_shared_state_dict(backup)

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
            # 必须走 class_probs：序数头输出的是 K-1 个阈值 logit，直接 softmax
            # 会算出一个含义错误的 4 维向量且**不报错**
            probs.append(self.model.class_probs(logits).cpu().numpy())
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
            probs.append(self.model.class_probs(logits).cpu().numpy())
            labels.append(y.numpy())
        return np.concatenate(labels), np.concatenate(probs)
