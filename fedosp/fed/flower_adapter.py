"""可选的 Flower 适配层：把 :mod:`fedosp.fed.strategies` 接到 ``flwr`` 上。

**默认不需要它。** ``fedosp.run_fed`` 自带的模拟器对 4 个 client 已经够用，
而且完全可控（本地步数、原型通道、LayerNorm 留本地都需要非标准的通信内容，
用内置模拟器更容易讲清楚、也更容易在论文里复现）。

什么时候用这里：
* 审稿人要求「用标准联邦框架实现」
* 后续要把 client 拆到多台机器上真跑

``pip install flwr>=1.7`` 之后::

    from fedosp.fed.flower_adapter import run_flower_simulation
    run_flower_simulation(clients, strategy, num_rounds=100, steps={...})

实现方式：把 :class:`ClientUpdate` 的所有张量拍平成 ``List[np.ndarray]`` 走 Flower 的
``Parameters``，键顺序由服务器统一维护，保证两端一致。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .strategies import ClientUpdate, FedStrategy, ServerState

LOGGER = logging.getLogger("flower_adapter")


def _require_flwr():
    try:
        import flwr  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要 Flower：pip install 'flwr[simulation]>=1.7'。"
            "如果只是想跑实验，用内置模拟器即可：python -m fedosp.run_fed"
        ) from exc
    return flwr


class TensorCodec:
    """在 ``Dict[str, Tensor]`` 与 ``List[np.ndarray]`` 之间来回转，键顺序固定。"""

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = list(keys)

    def encode(self, state: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        return [state[k].detach().cpu().numpy() for k in self.keys]

    def decode(self, arrays: Sequence[np.ndarray]) -> Dict[str, torch.Tensor]:
        if len(arrays) != len(self.keys):
            raise ValueError(
                f"参数数量不匹配：收到 {len(arrays)}，期望 {len(self.keys)}。"
                "通常是某个 client 的模型结构与服务器不一致。"
            )
        return {k: torch.from_numpy(np.asarray(a)) for k, a in zip(self.keys, arrays)}


def make_flower_client(local_client, steps: int, codec: TensorCodec):
    """把 :class:`fedosp.fed.client.LocalClient` 包成 ``flwr.client.NumPyClient``。"""
    flwr = _require_flwr()

    class _Client(flwr.client.NumPyClient):
        def get_parameters(self, config):  # noqa: D102
            return codec.encode(local_client.model.shared_state_dict())

        def fit(self, parameters, config):  # noqa: D102
            state = ServerState(shared=codec.decode(parameters))
            # 原型走 config 通道传字节串（Flower 的 Parameters 只放共享权重）
            if "deep_proto" in config:
                state.deep_proto = torch.from_numpy(
                    np.frombuffer(config["deep_proto"], dtype=np.float32).reshape(
                        config["proto_shape"]
                    ).copy()
                )
            local_client.load_from_server(state)
            update = local_client.local_train(steps, dict(config))
            metrics = {k: float(v) for k, v in update.metrics.items()}
            metrics["deep_proto"] = update.deep_proto.numpy().astype(np.float32).tobytes()
            metrics["deep_seen"] = update.deep_seen.numpy().tobytes()
            return codec.encode(update.shared), update.n, metrics

        def evaluate(self, parameters, config):  # noqa: D102
            local_client.load_from_server(ServerState(shared=codec.decode(parameters)))
            m = local_client.evaluate("val")
            return float(1.0 - m.get("qwk", 0.0)), int(m.get("n", 0)), m

    return _Client().to_client()


def run_flower_simulation(
    clients: Sequence[Any],
    strategy: FedStrategy,
    steps: Dict[str, int],
    num_rounds: int = 100,
    client_resources: Optional[Dict[str, float]] = None,
):
    """用 ``flwr.simulation.start_simulation`` 跑同一套策略。

    Args:
        clients: :class:`LocalClient` 列表。
        strategy: 本项目的策略对象，聚合逻辑会被 Flower 的 Strategy 代理调用。
        steps: ``client 名 -> 本地步数``（sqrt 规则的结果）。
    """
    flwr = _require_flwr()
    keys = list(clients[0].model.shared_state_dict().keys())
    codec = TensorCodec(keys)
    index = {str(i): c for i, c in enumerate(clients)}

    class _Strategy(flwr.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):  # noqa: D102
            if failures:
                LOGGER.warning("round %d 有 %d 个 client 失败", server_round, len(failures))
            updates: List[ClientUpdate] = []
            for client_proxy, fit_res in results:
                arrays = flwr.common.parameters_to_ndarrays(fit_res.parameters)
                m = fit_res.metrics
                updates.append(
                    ClientUpdate(
                        client=client_proxy.cid,
                        n=fit_res.num_examples,
                        shared=codec.decode(arrays),
                        metrics={k: v for k, v in m.items() if isinstance(v, (int, float))},
                        deep_proto=torch.from_numpy(
                            np.frombuffer(m["deep_proto"], dtype=np.float32)
                            .reshape(-1, len(keys) and clients[0].model.embed_dim)
                            .copy()
                        ) if "deep_proto" in m else None,
                        deep_seen=torch.from_numpy(
                            np.frombuffer(m["deep_seen"], dtype=bool).copy()
                        ) if "deep_seen" in m else None,
                    )
                )
            new_state = strategy.aggregate(updates, ServerState())
            return flwr.common.ndarrays_to_parameters(codec.encode(new_state.shared)), {}

        def configure_fit(self, server_round, parameters, client_manager):  # noqa: D102
            cfg = strategy.client_config(server_round)
            ins = flwr.common.FitIns(parameters, cfg)
            sampled = client_manager.sample(
                num_clients=len(index), min_num_clients=len(index)
            )
            return [(c, ins) for c in sampled]

    def client_fn(cid: str):
        c = index[cid]
        return make_flower_client(c, steps[c.name], codec)

    LOGGER.info("启动 Flower 模拟：%d client × %d 轮", len(clients), num_rounds)
    return flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=_Strategy(
            initial_parameters=flwr.common.ndarrays_to_parameters(
                codec.encode(clients[0].model.shared_state_dict())
            )
        ),
        client_resources=client_resources or {"num_cpus": 2, "num_gpus": 0.25},
    )
