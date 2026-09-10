"""联邦聚合策略：8 个方法在同一套接口下实现（方案第 7 节 B3–B9 + FedOSP）。

设计要点：**所有策略共享同一个骨干、同一份 manifest、同一套本地步数规则**，
唯一的差别就在「上传什么、怎么聚合、本地多一项什么损失」。
只有这样，主表里的差异才能归因到聚合机制本身而不是实现细节。

======================  ===============================================================
策略                     与 FedAvg 的差别
======================  ===============================================================
``fedavg``              基准：按样本数加权平均 LoRA + head
``fedprox``             本地多一个近端项 ``mu/2 ||w - w_g||^2``
``fedbn``               LayerNorm 留本地（模型层面已支持，这里只是关掉原型/FSR）
``scaffold``            上传 control variate 修正 client drift
``fedper``              分类头留本地，不参与聚合
``fedproto``            额外聚合原型，**按样本数加权**（本文 client 等权的直接对手）
``feduaa``              本地 evidential 头，按 ``1-u`` 不确定性加权聚合
``fedosp``              FSR + 双层原型 + **client 等权原型聚合** + sqrt 参数权重
======================  ===============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import aggregation_weights
from ..models.prototypes import aggregate_prototypes

LOGGER = logging.getLogger("strategies")


@dataclass
class ClientUpdate:
    """一个 client 一轮训练之后回传给服务器的全部内容。"""

    client: str
    n: int                                            # 本地训练样本数
    shared: Dict[str, torch.Tensor]                   # LoRA + head + FSR gate
    metrics: Dict[str, float] = field(default_factory=dict)
    shallow_proto: Optional[torch.Tensor] = None
    shallow_seen: Optional[torch.Tensor] = None
    deep_proto: Optional[torch.Tensor] = None
    deep_seen: Optional[torch.Tensor] = None
    control_delta: Optional[Dict[str, torch.Tensor]] = None   # SCAFFOLD
    uncertainty: Optional[float] = None                       # FedUAA-style

    def num_params(self) -> int:
        n = sum(v.numel() for v in self.shared.values())
        for t in (self.shallow_proto, self.deep_proto):
            if t is not None:
                n += t.numel()
        if self.control_delta:
            n += sum(v.numel() for v in self.control_delta.values())
        return n

    def upload_mb(self) -> float:
        return self.num_params() * 4 / 1024 / 1024


@dataclass
class ServerState:
    """服务器下发给 client 的内容。"""

    shared: Dict[str, torch.Tensor] = field(default_factory=dict)
    shallow_proto: Optional[torch.Tensor] = None
    deep_proto: Optional[torch.Tensor] = None
    control: Optional[Dict[str, torch.Tensor]] = None


def weighted_average(
    states: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]
) -> Dict[str, torch.Tensor]:
    """按权重平均若干 state_dict。缺 key 的 client 会被跳过并重新归一化权重。"""
    out: Dict[str, torch.Tensor] = {}
    keys = set().union(*[set(s.keys()) for s in states]) if states else set()
    for k in keys:
        vals, ws = [], []
        for s, w in zip(states, weights):
            if k in s:
                vals.append(s[k].float())
                ws.append(float(w))
        if not vals:
            continue
        w_arr = np.asarray(ws)
        w_arr = w_arr / w_arr.sum()
        acc = torch.zeros_like(vals[0])
        for v, w in zip(vals, w_arr):
            acc += v * float(w)
        out[k] = acc
    return out


# --------------------------------------------------------------------------- #
class FedStrategy:
    """策略基类。子类只需要覆写自己真正不同的那一两个钩子。"""

    name = "fedavg"
    #: LoRA/head 的聚合权重模式：sample / sqrt / equal
    param_weight_mode = "sample"
    #: 是否聚合原型
    uses_prototypes = False
    #: 原型聚合模式（A6 消融的核心开关）
    proto_agg_mode = "client_equal"
    #: 哪些 shared key 前缀不参与聚合（FedPer 用来把 head 留本地）
    local_only_prefixes: Sequence[str] = ()

    def __init__(self, **kwargs) -> None:
        self.cfg = kwargs
        self.round = 0
        self.history: List[Dict[str, float]] = []

    # -------------------- 下发给 client 的额外配置 -------------------- #
    def client_config(self, round_idx: int) -> Dict[str, object]:
        """本地训练需要知道的策略相关参数（如 FedProx 的 mu）。"""
        return {}

    # ---------------------------- 参数权重 ---------------------------- #
    def compute_weights(self, updates: Sequence[ClientUpdate]) -> np.ndarray:
        return aggregation_weights([u.n for u in updates], self.param_weight_mode)

    # ------------------------------ 聚合 ------------------------------ #
    def aggregate(self, updates: Sequence[ClientUpdate], state: ServerState) -> ServerState:
        if not updates:
            raise ValueError("本轮没有任何 client 上传，检查参与率设置")
        self.round += 1
        weights = self.compute_weights(updates)

        shared_list = [
            {k: v for k, v in u.shared.items()
             if not any(k.startswith(p) for p in self.local_only_prefixes)}
            for u in updates
        ]
        new_shared = weighted_average(shared_list, weights)

        new_state = ServerState(shared=new_shared, control=state.control)

        if self.uses_prototypes:
            sizes = {u.client: u.n for u in updates}
            if updates[0].deep_proto is not None:
                new_state.deep_proto = aggregate_prototypes(
                    {u.client: u.deep_proto for u in updates},
                    {u.client: u.deep_seen for u in updates},
                    sizes,
                    self.proto_agg_mode,
                )
            if updates[0].shallow_proto is not None:
                new_state.shallow_proto = aggregate_prototypes(
                    {u.client: u.shallow_proto for u in updates},
                    {u.client: u.shallow_seen for u in updates},
                    sizes,
                    self.proto_agg_mode,
                )

        self._log_round(updates, weights)
        return new_state

    def _log_round(self, updates: Sequence[ClientUpdate], weights: np.ndarray) -> None:
        mb = sum(u.upload_mb() for u in updates)
        LOGGER.info(
            "[%s] round %d | 权重 %s | 本轮上传 %.2f MB | loss %s",
            self.name,
            self.round,
            {u.client: round(float(w), 3) for u, w in zip(updates, weights)},
            mb,
            {u.client: round(u.metrics.get("loss", float("nan")), 3) for u in updates},
        )
        self.history.append({"round": self.round, "upload_mb": mb})


class FedAvg(FedStrategy):
    name = "fedavg"
    param_weight_mode = "sample"


class FedProx(FedStrategy):
    """B4：本地目标加近端项，抑制 client drift。"""

    name = "fedprox"
    param_weight_mode = "sample"

    def client_config(self, round_idx: int) -> Dict[str, object]:
        return {"fedprox_mu": float(self.cfg.get("mu", 0.01))}


class FedBN(FedStrategy):
    """B5：LayerNorm affine 留本地。

    模型侧只要 ``FedOSPConfig.personal_layernorm=True``，
    ``shared_state_dict()`` 就已经不含 LN，这里不需要额外处理。
    """

    name = "fedbn"
    param_weight_mode = "sample"


class FedPer(FedStrategy):
    """B7：分类头留本地（representation 共享、head 个性化）。"""

    name = "fedper"
    param_weight_mode = "sample"
    local_only_prefixes = ("head.",)


class Scaffold(FedStrategy):
    """B6：control variates 修正 client drift。

    服务器维护全局 control ``c``，每轮按 ``c <- c + (1/N) * sum_k dc_k`` 更新。
    通信量因此翻倍，这一点要如实报进系统开销表 T5。
    """

    name = "scaffold"
    param_weight_mode = "sample"
    #: 标记：run_fed 会据此把 ServerState.control 初始化成 ``{}`` 而不是 ``None``。
    #: 不这么做的话 client 那侧 ``state.control is not None`` 永远为假，
    #: 就不会记录本轮起点，control variate 永远是空的 —— SCAFFOLD 静默退化成 FedAvg。
    needs_control = True

    def aggregate(self, updates: Sequence[ClientUpdate], state: ServerState) -> ServerState:
        new_state = super().aggregate(updates, state)
        deltas = [u.control_delta for u in updates if u.control_delta]
        control = dict(state.control or {})
        if deltas:
            # 原论文：c <- c + (|S|/N) * mean(dc_i)。本文全 client 参与，|S|=N，系数为 1。
            for k in deltas[0]:
                acc = torch.zeros_like(deltas[0][k].float())
                for d in deltas:
                    acc += d[k].float()
                acc /= len(deltas)
                control[k] = control.get(k, torch.zeros_like(acc)) + acc
        new_state.control = control
        return new_state

    def client_config(self, round_idx: int) -> Dict[str, object]:
        return {"scaffold": True}


class FedProto(FedStrategy):
    """B8：原型联邦，**按样本数加权** —— 本文 client 等权聚合的直接对手。"""

    name = "fedproto"
    param_weight_mode = "sample"
    uses_prototypes = True
    proto_agg_mode = "sample"

    def client_config(self, round_idx: int) -> Dict[str, object]:
        return {"use_proto_loss": True, "use_ordinal_margin": False}


class FedUAAStyle(FedStrategy):
    """B9：按论文描述在统一骨干下的 FedUAA 复现版（MICCAI 2023）。

    本地用 evidential 头，回传平均不确定性 ``u``；服务器按 ``(1-u) * n_k`` 加权。

    **不声称等同原文**：原文骨干、训练细节与本文不同，这里为了公平比较统一换成
    RETFound-LoRA。论文里必须写明这一点（方案第 12 节 R4）。
    """

    name = "feduaa"

    def client_config(self, round_idx: int) -> Dict[str, object]:
        # KL 项按原文做法线性退火
        anneal = min(1.0, round_idx / max(1, int(self.cfg.get("anneal_rounds", 10))))
        return {"evidential": True, "edl_anneal": anneal}

    def compute_weights(self, updates: Sequence[ClientUpdate]) -> np.ndarray:
        u = np.array([
            up.uncertainty if up.uncertainty is not None else 0.5 for up in updates
        ])
        conf = np.clip(1.0 - u, 1e-3, None)
        w = conf * np.array([up.n for up in updates], dtype=np.float64)
        return w / w.sum()


class FedOSP(FedStrategy):
    """本文方法：FSR + 双层原型 + **client 等权原型聚合** + sqrt 参数权重。"""

    name = "fedosp"
    param_weight_mode = "sqrt"       # A6 可切 sample / equal
    uses_prototypes = True
    proto_agg_mode = "client_equal"  # A6 可切 sqrt / sample

    def __init__(self, proto_agg_mode: str = "client_equal",
                 param_weight_mode: str = "sqrt", **kwargs) -> None:
        super().__init__(**kwargs)
        self.proto_agg_mode = proto_agg_mode
        self.param_weight_mode = param_weight_mode

    def client_config(self, round_idx: int) -> Dict[str, object]:
        return {"use_proto_loss": True, "use_ordinal_margin": True, "style_aug": True}


STRATEGIES = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedbn": FedBN,
    "fedper": FedPer,
    "scaffold": Scaffold,
    "fedproto": FedProto,
    "feduaa": FedUAAStyle,
    "fedosp": FedOSP,
}


def build_strategy(name: str, **kwargs) -> FedStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"未知策略 {name}，可选：{sorted(STRATEGIES)}")
    LOGGER.info("使用策略 %s（%s）", name, kwargs or "默认超参")
    return STRATEGIES[name](**kwargs)
