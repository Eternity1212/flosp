"""联邦聚合策略：13 个方法在同一套接口下实现（方案第 7 节 B3–B16 + FedOSP）。

设计要点：**所有策略共享同一个骨干、同一份 manifest、同一套本地步数规则**，
唯一的差别就在「上传什么、怎么聚合、本地多一项什么损失」。
只有这样，主表里的差异才能归因到聚合机制本身而不是实现细节。

======================  ====  ==========  ==========================================
策略                     年份  改哪一环     与 FedAvg 的差别
======================  ====  ==========  ==========================================
``fedavg``              2017  --          基准：按样本数加权平均 LoRA + head
``fedprox``             2020  本地损失     多一个近端项 ``mu/2 ||w - w_g||^2``
``fedbn``               2021  参数划分     LayerNorm 留本地
``scaffold``            2020  本地梯度     control variate 修正 client drift
``fedper``              2019  参数划分     分类头留本地，不参与聚合
``fedproto``            2022  聚合         额外聚合原型，**按样本数加权**
``feduaa``              2023  聚合权重     按 ``1-u`` 不确定性加权
``moon``                2021  本地损失     模型级对比：拉向全局、推离上一轮自己
``fedala``              2023  **下发**     元素级插值 ``w_l + W ⊙ (w_g - w_l)``
``qfedavg``             2020  聚合公式     非加权平均；按 ``F_k^q`` 偏向高损失 client
``ditto``               2021  个性化       双模型，**评估用个人模型 v**
``feddg``               2021  数据         共享幅度谱 + 频域增广（FSR 的直接对手）
``fedosp``              本文  聚合+表征    FSR + 双层原型 + **精度加权原型聚合**
======================  ====  ==========  ==========================================

两个容易踩的坑，都已经在类型系统里封死：

1. **辅助正则的公平性**。``L_style``/``L_cons`` 曾经只对 FedOSP 开启，增益无法归因。
   现在由 ``--aux-reg`` 全局控制，注入点在 final 方法 :meth:`FedStrategy.client_config`
   里，子类改不到（见 :meth:`FedStrategy._strategy_config`）。
2. **个性化方法的评估对象**。FedALA / Ditto 把个性化存在本地，如果评估前照常
   ``load_from_server``，个性化会被冲掉、方法精确退化成 FedAvg —— 而主表只会显示成
   "这个方法没效果"。由 :attr:`FedStrategy.eval_mode` 显式声明，见那里的说明。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import aggregation_weights
from ..models.prototypes import aggregate_prototypes
from .diagnostics import aggregation_diagnostics

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
    #: q-FedAvg 需要的 ``F_k(w^t)``：**在下发的全局参数处**评估的本地损失，
    #: 不是本轮训练过程的平均损失（那个已经被本地更新污染了）。
    loss_at_global: Optional[float] = None
    #: 本轮本地训练的实际步数（Ditto 会翻倍，系统开销表 T5 要如实报）
    compute_steps: Optional[int] = None
    # 精度加权聚合（C2）所需：每类原型的抽样方差 (C,)。各 5 个标量，通信开销可忽略。
    shallow_var: Optional[torch.Tensor] = None
    deep_var: Optional[torch.Tensor] = None

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
    #: **评估用哪个模型**。这是个性化联邦方法最容易被实现错的地方。
    #:
    #: ==============  ==========================================================
    #: 取值             语义
    #: ==============  ==========================================================
    #: ``"global"``    评估前重新下发全局参数（FedAvg 族的标准做法）
    #: ``"local"``     **不重新下发**，就评估本地训练完的那个模型（FedALA）
    #: ``"personal"``  切到显式维护的个人模型（Ditto 的 ``v``）
    #: ==============  ==========================================================
    #:
    #: 为什么必须显式区分：``run_fed`` 的评估循环默认会先 ``load_from_server``，
    #: 对 FedALA / Ditto 这类把个性化存在本地的方法，这一步会把个性化结果整个冲掉，
    #: 于是它们精确退化成 FedAvg —— 而主表上只会显示成"这个方法在本任务上没效果"，
    #: 一个从结果里完全看不出来的假结论。
    #:
    #: FedPer / FedBN 不需要特殊处理：它们的个性化部分本来就不在 ``state.shared`` 里，
    #: ``"global"`` 模式下 ``load_shared_state_dict`` 自然不会碰到。
    eval_mode = "global"

    def __init__(self, aux_reg: bool = False, **kwargs) -> None:
        self.cfg = kwargs
        self.round = 0
        self.history: List[Dict[str, float]] = []
        #: L_style / L_cons 辅助正则开关。由 ``--aux-reg`` 全局统一控制，
        #: **所有策略共享同一个值**，保证公平比较（设计文档 §3.3）。
        self.aux_reg = aux_reg

    # -------------------- 下发给 client 的额外配置 -------------------- #
    def client_config(self, round_idx: int) -> Dict[str, object]:
        """下发给 client 的完整配置。**子类不要覆写这个方法**，改覆写 :meth:`_strategy_config`。

        ``style_aug``（即 L_style / L_cons 是否启用）必须对所有策略取同一个值，
        否则就回到「只有 FedOSP 吃到辅助正则」的不公平比较（设计文档 §3.3）。
        把它放在这个 final 方法里，子类就不可能漏掉。
        """
        cfg: Dict[str, object] = {"style_aug": self.aux_reg}
        cfg.update(self._strategy_config(round_idx))
        return cfg

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        """各策略自己特有的参数（如 FedProx 的 mu）。子类覆写这个。"""
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

        proto_diag: Dict[str, object] = {}
        if self.uses_prototypes:
            sizes = {u.client: u.n for u in updates}
            if updates[0].deep_proto is not None:
                new_state.deep_proto = aggregate_prototypes(
                    {u.client: u.deep_proto for u in updates},
                    {u.client: u.deep_seen for u in updates},
                    sizes,
                    self.proto_agg_mode,
                    sampling_vars={u.client: u.deep_var for u in updates}
                    if updates[0].deep_var is not None
                    else None,
                    tau2=self.cfg.get("tau2_override"),
                    diagnostics=proto_diag,
                )
            if updates[0].shallow_proto is not None:
                new_state.shallow_proto = aggregate_prototypes(
                    {u.client: u.shallow_proto for u in updates},
                    {u.client: u.shallow_seen for u in updates},
                    sizes,
                    self.proto_agg_mode,
                    sampling_vars={u.client: u.shallow_var for u in updates}
                    if updates[0].shallow_var is not None
                    else None,
                    tau2=self.cfg.get("tau2_override"),
                )

        self._log_round(updates, weights, proto_diag)
        return new_state

    def _log_round(
        self,
        updates: Sequence[ClientUpdate],
        weights: np.ndarray,
        proto_diag: Optional[Dict[str, object]] = None,
    ) -> None:
        mb = sum(u.upload_mb() for u in updates)
        LOGGER.info(
            "[%s] round %d | 权重 %s | 本轮上传 %.2f MB | loss %s",
            self.name,
            self.round,
            {u.client: round(float(w), 3) for u, w in zip(updates, weights)},
            mb,
            {u.client: round(u.metrics.get("loss", float("nan")), 3) for u in updates},
        )

        # ---- C2 诊断：这一轮聚合实际"用上"了几个 client ----
        proto_diag = proto_diag or {}
        diag = aggregation_diagnostics(
            client_names=[u.client for u in updates],
            param_weights=weights,
            local_steps=[u.metrics.get("steps", 0.0) for u in updates],
            proto_weights=proto_diag.get("proto_weights"),      # type: ignore[arg-type]
            tau2_per_class=proto_diag.get("tau2_per_class"),    # type: ignore[arg-type]
        )
        LOGGER.info(
            "[%s] round %d | n_eff(参数) %.2f / %d | n_eff(复合) %.2f%s",
            self.name,
            self.round,
            diag["n_eff_param"],
            diag["n_clients"],
            diag["n_eff_compound"],
            f" | tau2 {diag['tau2_mean']:.2e}" if "tau2_mean" in diag else "",
        )
        self.history.append({"round": self.round, "upload_mb": mb, **diag})


class FedAvg(FedStrategy):
    name = "fedavg"
    param_weight_mode = "sample"


class FedProx(FedStrategy):
    """B4：本地目标加近端项，抑制 client drift。"""

    name = "fedprox"
    param_weight_mode = "sample"

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
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

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        return {"scaffold": True}


class FedProto(FedStrategy):
    """B8：原型联邦，**按样本数加权** —— 本文 client 等权聚合的直接对手。"""

    name = "fedproto"
    param_weight_mode = "sample"
    uses_prototypes = True
    proto_agg_mode = "sample"

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        return {"use_proto_loss": True, "use_ordinal_margin": False}


class FedUAAStyle(FedStrategy):
    """B9：按论文描述在统一骨干下的 FedUAA 复现版（MICCAI 2023）。

    本地用 evidential 头，回传平均不确定性 ``u``；服务器按 ``(1-u) * n_k`` 加权。

    **不声称等同原文**：原文骨干、训练细节与本文不同，这里为了公平比较统一换成
    RETFound-LoRA。论文里必须写明这一点（方案第 12 节 R4）。
    """

    name = "feduaa"

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
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


# --------------------------------------------------------------------------- #
# B12–B16：文献综述后补上的 4 个关键基线
#
# 补这几个的理由（设计文档 §3.3）：原来的 8 个基线里，**没有一个是 2021 年之后的
# 通用 FL 强基线**。审稿人会直接问"为什么不比 MOON / FedALA"。更要紧的是，
# FedOSP 的两个贡献分别属于"表征对齐"和"聚合权重"两个族，必须各自有同族的强对手：
#
#   * C1（序数原型几何）  同族对手 → MOON（对比式表征对齐）、FedProto
#   * C2（精度加权聚合）  同族对手 → q-FedAvg（按损失加权）、FedALA（元素级插值）
#   * FSR（风格归一）     同族对手 → FedDG-ELCFS（频域增广，且**需要共享幅度谱**）
#   * 个性化这条线                → Ditto、FedPer
# --------------------------------------------------------------------------- #
class MOON(FedStrategy):
    """B12：MOON（Li, He & Song, CVPR 2021）—— 模型级对比学习。

    本地目标加一项对比损失：把当前表征**拉向全局模型的表征**、**推离上一轮自己的表征**：

    .. math::
        \\ell_{con} = -\\log\\frac{\\exp(\\mathrm{sim}(z, z_{glob})/T)}
        {\\exp(\\mathrm{sim}(z,z_{glob})/T)+\\exp(\\mathrm{sim}(z,z_{prev})/T)}

    服务器侧与 FedAvg **完全相同**，差别全在本地（见 ``LocalClient._moon_contrastive``）。

    为什么它是 C1 最重要的对手：MOON 也在做"表征对齐"，而且是最强的通用做法。
    如果 FedOSP 的序数原型几何只是"某种表征对齐"在起作用，那 MOON 应该能拿到同样的
    增益；只有当 FedOSP 在 **QWK / 远端误判** 上明显赢过 MOON 而 accuracy 相当时，
    才能说增益来自**序数结构**而不是泛泛的对齐。这正是 C1 的归因实验。

    代价要如实报：每步需要 3 次前向（当前 / 全局 / 上一轮），本地计算约 1.7x。
    """

    name = "moon"
    param_weight_mode = "sample"

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        return {
            "moon": True,
            "moon_mu": float(self.cfg.get("moon_mu", 1.0)),
            "moon_tau": float(self.cfg.get("moon_tau", 0.5)),
        }


class Ditto(FedStrategy):
    """B15：Ditto（Li et al., ICML 2021）—— 个性化与全局的双层目标。

    每个 client 同时维护两个模型：

    * ``w``：正常参与联邦聚合的全局模型
    * ``v``：**个人模型**，目标是 :math:`F_k(v) + \\frac{\\lambda}{2}\\lVert v-w^t\\rVert^2`

    **评估用 ``v``**，这是 Ditto 的全部要点，也是最容易实现错的地方：如果评估仍然用
    ``w``，Ditto 就完全退化成 FedAvg，而主表上会显示成"Ditto 没效果"这种假结论。

    服务器侧就是 FedAvg（只聚合 ``w``）。代价是本地步数翻倍，T5 里按 2x 报。
    """

    name = "ditto"
    param_weight_mode = "sample"
    eval_mode = "personal"        # 用 v 评估，不是 w

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        return {"ditto": True, "ditto_lambda": float(self.cfg.get("ditto_lambda", 0.1))}


class FedALA(FedStrategy):
    """B13：FedALA（Zhang et al., AAAI 2023）—— 自适应本地聚合。

    **它改的是"下发"而不是"聚合"**：client 收到全局参数后不直接覆盖本地参数，而是做
    一次学出来的**元素级**插值

    .. math:: \\hat w \\leftarrow w_{local} + W \\odot (w_{global} - w_{local})

    其中 :math:`W\\in[0,1]` 逐元素、在本地数据上用梯度下降学出来、**跨轮持久保留**。

    服务器侧与 FedAvg 相同。为什么它是 C2 的关键对手：FedALA 与 C2 都在回答
    "全局信息该以多大强度进入本地"，但走的是两条不同的路 ——

    ======================  ====================  ==========================
    \\                       FedALA                 FedOSP C2
    ======================  ====================  ==========================
    调节位置                 下发侧（每个 client）   聚合侧（服务器）
    粒度                    参数元素级              类别级
    依据                    本地数据上的梯度        随机效应方差分解（闭式解）
    是否有最优性保证          无（启发式）            有（MSE 最优）
    ======================  ====================  ==========================

    这个对比本身就是论文里一段很好的 discussion：FedALA 表达力更强但没有最优性保证，
    C2 粒度更粗但每一步都有理论依据。两者**不冲突**，可以叠加（见 B13+C2 的组合行）。
    """

    name = "fedala"
    param_weight_mode = "sample"
    #: FedALA 的个性化模型就是本地训练完的那个（从插值起点训出来的），
    #: 评估前再下发一次全局参数会把它冲掉
    eval_mode = "local"

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        return {
            "ala": True,
            "ala_lr": float(self.cfg.get("ala_lr", 0.1)),
            "ala_iters": int(self.cfg.get("ala_iters", 5)),
            # 只对"高层"做插值，低层直接覆盖（原论文的 p）
            "ala_last_n": int(self.cfg.get("ala_last_n", 4)),
            "ala_round": round_idx,
        }


class QFedAvg(FedStrategy):
    """B14：q-FedAvg（Li et al., ICLR 2020）—— 公平性导向的联邦优化。

    它**不是加权平均**，所以整个 :meth:`aggregate` 都要覆写。原论文 Algorithm 2：

    .. math::
        \\Delta_k = F_k(w^t)^q\\,\\Delta w_k,\\qquad
        h_k = q F_k(w^t)^{q-1}\\lVert\\Delta w_k\\rVert^2 + L\\,F_k(w^t)^q

    .. math:: w^{t+1} = w^t - \\frac{\\sum_k \\Delta_k}{\\sum_k h_k}

    其中 :math:`\\Delta w_k = L(w^t - w_k^{t+1})`，:math:`L` 是 Lipschitz 常数估计
    （实践中取 ``1/eta_l``）。``q=0`` 时精确退化为 FedAvg；``q`` 越大越偏向**损失高的
    client**，即牺牲平均性能换取最差 client 的性能。

    为什么必须有它：本文的 ``n_eff`` 叙事（"EyePACS 淹没了 IDRiD"）本质上是一个
    **公平性**论述，而 q-FedAvg 是公平性联邦最标准的做法。不比它，"我们让小 client
    受益"这句话就没有对照。注意两者机制完全不同：q-FedAvg 按**损失**加权（谁学得差谁
    权重大），C2 按**方差**加权（谁估计得准谁权重大），后者与损失高低无关。

    ⚠ 两条已知性质，都不是 bug，但都很容易被误读：

    1. **q>0 时平均准确率通常下降**。这是公平性定义本身的代价，报结果时必须同时给
       mean 与 worst-client 两列，只看平均会得出"q-FedAvg 没用"的错误结论。
    2. **有效步长对 q 不单调**。分母里 :math:`qF^{q-1}\\lVert\\Delta w\\rVert^2`
       含 :math:`L^2` 因子，比分子的 :math:`F^q` 增长更快，于是 q 越大步子越小
       （在本文配置下 q=1 的步长是 3.0e-4，q=5 只剩 1.3e-4）。
       随 q 单调上升的是**权重份额** :math:`F_k^q/\\sum_j F_j^q`，不是步长。
       调大 q 却"看起来没反应"时，先怀疑这一条再怀疑实现。
    """

    name = "qfedavg"

    def __init__(self, q: float = 1.0, lipschitz: Optional[float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.q = float(q)
        #: L 的估计。None 时在 aggregate 里用 1/lr 兜底（需要 client 上报 lr）
        self.lipschitz = lipschitz

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        # 让 client 在下发参数处多评一次损失（F_k(w^t)），这是 q-FedAvg 的必需输入
        return {"report_loss_at_global": True}

    def aggregate(self, updates: Sequence[ClientUpdate], state: ServerState) -> ServerState:
        if not updates:
            raise ValueError("本轮没有任何 client 上传，检查参与率设置")
        if not state.shared:
            # 第一轮服务器还没有 w^t（没得可减），退化成一次普通 FedAvg 做初始化
            LOGGER.info("[qfedavg] 第一轮无全局参数，本轮按 FedAvg 初始化")
            return super().aggregate(updates, state)

        self.round += 1
        lip = self.lipschitz or 1.0 / max(float(self.cfg.get("lr", 1e-4)), 1e-12)

        losses = []
        for u in updates:
            if u.loss_at_global is None:
                raise RuntimeError(
                    f"q-FedAvg 需要 client 上报 F_k(w^t)，但 {u.client} 没有传 "
                    "loss_at_global。检查 client_config 里的 report_loss_at_global 是否生效。"
                )
            # F_k 必须为正：q 次幂与 q-1 次幂都要求这一点
            losses.append(max(float(u.loss_at_global), 1e-10))

        keys = [k for k in state.shared if not any(
            k.startswith(p) for p in self.local_only_prefixes)]
        num: Dict[str, torch.Tensor] = {k: torch.zeros_like(state.shared[k].float()) for k in keys}
        denom = 0.0

        for u, fk in zip(updates, losses):
            # dw = L * (w^t - w_k)，只在双方都有的 key 上算
            dw = {
                k: lip * (state.shared[k].float() - u.shared[k].float())
                for k in keys if k in u.shared
            }
            sq = sum(float(v.pow(2).sum()) for v in dw.values())
            fq = fk ** self.q
            for k, v in dw.items():
                num[k] += fq * v
            denom += self.q * (fk ** (self.q - 1.0)) * sq + lip * fq

        if denom <= 0:
            raise RuntimeError(f"q-FedAvg 的 h 之和为 {denom}，无法更新；检查 q 与 lr")

        new_shared = {k: state.shared[k].float() - num[k] / denom for k in keys}
        LOGGER.info(
            "[qfedavg] round %d | q=%.1f L=%.1f | F_k=%s | sum(h)=%.3e",
            self.round, self.q, lip,
            {u.client: round(f, 4) for u, f in zip(updates, losses)}, denom,
        )

        # q-FedAvg 没有显式权重，用 F_k^q 的归一化值作为"等效权重"记进 n_eff 诊断，
        # 这样公平性叙事里 q-FedAvg 与 C2 能放在同一张图上比
        eff_w = np.array([f ** self.q for f in losses], dtype=np.float64)
        self._log_round(updates, eff_w / eff_w.sum(), {})
        return ServerState(shared=new_shared, control=state.control)


class FedDGELCFS(FedStrategy):
    """B16：FedDG-ELCFS（Liu et al., CVPR 2021）的分类版复现 —— **FSR 的直接对手**。

    原文做两件事：(1) 各 client 把自己图像的**幅度谱**上传到一个共享 bank，本地用
    别家的幅度谱做频域插值增广（continuous frequency space）；(2) 在"原图 / 增广图"
    上做 episodic meta-learning，并加边界与平滑两项**分割专用**损失。

    **本实现的范围与偏离（论文必须写明，方案第 12 节 R4）**：

    * ✅ 实现了 (1) 的全部：共享幅度谱 bank + 低频掩码内的连续插值增广
    * ✅ 实现了 (2) 的分类版对应物：原图/增广图的一致性（meta-train/meta-test 的
      分类类比），因为 episodic 二阶梯度在 ViT-L 上的显存代价不可接受
    * ❌ 未实现边界损失与平滑损失 —— 它们定义在分割掩码上，DR 分级里没有对应物

    为什么必须有它：它是 FSR 的同族对手（都在频域做风格归一），而且对比会暴露一个
    **对本文有利的结构性差异** —— ELCFS 必须把**原始图像的幅度谱**传出本地，
    而幅度谱足以恢复出可识别的图像结构；FSR 只在**中间特征**上做归一，什么都不外传。
    这不是我们方法"顺便"的优点，而是应该在隐私一节里正面论证的一条。
    T5 里要把 ELCFS 的 bank 通信量单列一行。
    """

    name = "feddg"
    param_weight_mode = "sample"
    #: 标记：run_fed 会据此构建共享幅度谱 bank 并挂到各 client 的 dataset 上
    needs_amplitude_bank = True

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        return {
            "feddg": True,
            # 原文的一致性项权重
            "feddg_lambda": float(self.cfg.get("feddg_lambda", 1.0)),
        }


class FedOSP(FedStrategy):
    """本文方法：FSR + 双层原型 + **精度加权原型聚合(C2)** + sqrt 参数权重。"""

    name = "fedosp"
    param_weight_mode = "sqrt"       # A6 可切 sample / equal
    uses_prototypes = True
    proto_agg_mode = "precision"     # A6 可切 client_equal / sqrt / sample

    def __init__(self, proto_agg_mode: str = "precision",
                 param_weight_mode: str = "sqrt", **kwargs) -> None:
        # aux_reg 不在这里单独处理：它由基类从 --aux-reg 全局接收，对所有策略取同一个值。
        # 曾经的写法是在这里硬编码 style_aug=True，导致只有 FedOSP 吃到 L_style/L_cons
        # 而 baseline 吃不到，增益无法归因（设计文档 §3.3）。
        super().__init__(**kwargs)
        self.proto_agg_mode = proto_agg_mode
        self.param_weight_mode = param_weight_mode

    def _strategy_config(self, round_idx: int) -> Dict[str, object]:
        # style_aug 由基类的 client_config 统一注入，这里不要重复声明
        return {"use_proto_loss": True, "use_ordinal_margin": True}


STRATEGIES = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedbn": FedBN,
    "fedper": FedPer,
    "scaffold": Scaffold,
    "fedproto": FedProto,
    "feduaa": FedUAAStyle,
    # B12–B16：文献综述后补上的 2021+ 强基线
    "moon": MOON,
    "fedala": FedALA,
    "qfedavg": QFedAvg,
    "ditto": Ditto,
    "feddg": FedDGELCFS,
    "fedosp": FedOSP,
}


def build_strategy(name: str, **kwargs) -> FedStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"未知策略 {name}，可选：{sorted(STRATEGIES)}")
    LOGGER.info("使用策略 %s（%s）", name, kwargs or "默认超参")
    return STRATEGIES[name](**kwargs)
