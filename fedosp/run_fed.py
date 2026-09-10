"""联邦模拟入口：一条命令跑完一个实验配置（方案第 6 节协议）。

用法::

    python -m fedosp.run_fed --config configs/default.yaml \
        --strategy fedosp --seed 0 --out runs/M0_s0

    # A6 消融：原型改成按样本数加权
    python -m fedosp.run_fed --strategy fedosp --proto-agg sample --out runs/A6a_s0

    # 没有数据时先跑通流程（合成假数据 + 微型 ViT）
    python -m fedosp.run_fed --dry-run

每轮做四件事：下发全局参数 → 各 client 按 ``S_k`` 步本地训练 → 聚合 → 在各 client 的
val 上评估并按 **macro-over-client QWK** 做早停。

**早停绝对不能看按样本加权的平均**，否则等于只看 EyePACS（方案第 12 节 R2）。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .data.dataset import load_manifest, local_steps, make_client_loaders
from .fed.client import LocalClient
from .fed.strategies import ServerState, build_strategy
from .losses import LossWeights
from .metrics import aggregate_over_clients, evaluate_predictions
from .models.retfound_lora import FedOSPConfig, build_model

LOGGER = logging.getLogger("run_fed")

TRAIN_CLIENTS = ["eyepacs", "aptos", "ddr", "idrid"]
HELDOUT_CLIENT = "messidor2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
def average_layernorm(clients: List[LocalClient]) -> Dict[str, torch.Tensor]:
    """把各 client 的本地 LayerNorm 参数取算术平均。

    未见中心（Messidor-2）没有自己的个性化参数，按方案 4.7 用全局平均。
    **这条规则必须写进论文**，否则审稿人会问「未见中心哪来的个性化参数」。
    """
    dicts = [c.model.personal_state_dict() for c in clients]
    out: Dict[str, torch.Tensor] = {}
    for k in dicts[0]:
        out[k] = torch.stack([d[k].float() for d in dicts]).mean(dim=0)
    return out


def evaluate_external(
    clients: List[LocalClient], state: ServerState, loader,
    raw_out: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """在未见中心上评估：全局 shared 参数 + 平均 LayerNorm。

    Args:
        raw_out: 传入一个 dict 就会把 per-sample 预测写进去（供统计检验）。
    """
    if loader is None:
        return {}
    probe = copy.deepcopy(clients[0])
    probe.name = HELDOUT_CLIENT
    probe.load_from_server(state)
    probe.model.load_state_dict(
        {k: v.to(probe.device) for k, v in average_layernorm(clients).items()}, strict=False
    )
    labels, probs = probe.predict(loader)
    if raw_out is not None:
        raw_out[f"{HELDOUT_CLIENT}__y"] = labels
        raw_out[f"{HELDOUT_CLIENT}__probs"] = probs
    res = evaluate_predictions(labels, probs)
    LOGGER.info("[external:%s] %s", HELDOUT_CLIENT, {k: round(v, 4) for k, v in res.items()})
    return res


# --------------------------------------------------------------------------- #
def build_clients(args, manifest, model_cfg: FedOSPConfig, loss_w: LossWeights) -> List[LocalClient]:
    clients: List[LocalClient] = []
    style_aug = args.strategy == "fedosp"
    for name in args.clients:
        loaders = make_client_loaders(
            manifest,
            name,
            img_size=model_cfg.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            style_aug=style_aug,
            label_budget=args.label_budget,
            seed=args.seed,
        )
        if "train" not in loaders:
            LOGGER.warning("client %s 没有训练数据，跳过", name)
            continue
        # 每个 client 一份独立模型副本 —— LayerNorm 是私有的，绝不能共享同一对象
        model = build_model(copy.deepcopy(model_cfg))
        clients.append(
            LocalClient(
                name,
                model,
                loaders,
                loss_weights=loss_w,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device,
                amp=args.amp,
            )
        )
    if not clients:
        raise RuntimeError("没有构建出任何 client，检查 manifest 与 --clients")
    return clients


def run(args) -> Dict:
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = (
        make_fake_manifest(out_dir / "fake") if args.dry_run else load_manifest(Path(args.manifest))
    )

    model_cfg = FedOSPConfig(
        backbone=args.backbone,
        img_size=args.img_size,
        fsr_after_block=args.fsr_block,
        use_fsr=not args.no_fsr,
        use_shallow_proto=not args.no_shallow_proto,
        use_deep_proto=not args.no_deep_proto,
        lora_rank=args.lora_rank,
        lora_last_n_blocks=args.lora_blocks,
        personal_layernorm=not args.no_personal_ln,
        pretrained_path=args.pretrained,
        full_finetune=getattr(args, "full_finetune", False),
        num_prompts=getattr(args, "num_prompts", 0),
    )
    loss_w = LossWeights(
        ord=args.lambda_ord,
        proto_grade=args.lambda_proto_grade,
        proto_style=args.lambda_proto_style,
        style=args.lambda_style,
        cons=args.lambda_cons,
        ordinal_margin=args.ordinal_margin,
    )

    clients = build_clients(args, manifest, model_cfg, loss_w)
    strategy = build_strategy(
        args.strategy,
        mu=args.fedprox_mu,
        proto_agg_mode=args.proto_agg,
        param_weight_mode=args.param_weight,
    ) if args.strategy == "fedosp" else build_strategy(args.strategy, mu=args.fedprox_mu)

    # 本地步数按 sqrt 规则分配（方案 4.6）
    steps = {c.name: local_steps(c.n_train, lo=args.min_steps, hi=args.max_steps) for c in clients}
    if args.steps_rule == "equal":
        steps = {k: args.max_steps // 2 for k in steps}
    elif args.steps_rule == "epoch":
        steps = {c.name: max(1, len(c.loaders["train"])) for c in clients}
    LOGGER.info("本地步数分配（%s）：%s | 每轮图像数 %d",
                args.steps_rule, steps, sum(steps.values()) * args.batch_size)

    external_loader = None
    if not args.dry_run:
        ext = make_client_loaders(
            manifest, HELDOUT_CLIENT, img_size=model_cfg.img_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        external_loader = ext.get("test")

    state = ServerState(shared=clients[0].model.shared_state_dict())
    if getattr(strategy, "needs_control", False):
        # SCAFFOLD：control 必须是 dict（哪怕是空的），否则 client 侧不会记录本轮起点
        state.control = {}
    best = {"round": -1, "macro_qwk": -np.inf}
    history: List[Dict] = []
    cum_mb = 0.0
    t_start = time.time()

    for rnd in range(1, args.rounds + 1):
        cfg = strategy.client_config(rnd)
        selected = select_clients(clients, args.participation, rnd, args.seed)

        updates = []
        for c in selected:
            c.load_from_server(state)
            updates.append(c.local_train(steps[c.name], cfg))
        state = strategy.aggregate(updates, state)
        cum_mb += sum(u.upload_mb() for u in updates)

        # ---- 验证：必须用 macro-over-client，不能用样本加权 ----
        per_client = {}
        for c in clients:
            c.load_from_server(state)
            m = c.evaluate("val")
            if m:
                per_client[c.name] = m
        summary = aggregate_over_clients(per_client, "qwk")
        row = {
            "round": rnd, "cum_upload_mb": cum_mb,
            "elapsed_s": time.time() - t_start,
            **summary,
            **{f"val_qwk_{k}": v["qwk"] for k, v in per_client.items()},
        }
        history.append(row)
        LOGGER.info(
            "round %3d | macro %.4f worst %.4f | per-client %s | 累计上传 %.1f MB",
            rnd, summary["macro_qwk"], summary["worst_qwk"],
            {k: round(v["qwk"], 3) for k, v in per_client.items()}, cum_mb,
        )

        if summary["macro_qwk"] > best["macro_qwk"]:
            best = {"round": rnd, **summary}
            torch.save(
                {"shared": state.shared,
                 "deep_proto": state.deep_proto,
                 "shallow_proto": state.shallow_proto,
                 "personal": {c.name: c.model.personal_state_dict() for c in clients}},
                out_dir / "best.pt",
            )
        elif rnd - best["round"] >= args.patience:
            LOGGER.info("早停：macro QWK 已 %d 轮未提升（最好在第 %d 轮）",
                        args.patience, best["round"])
            break

    # ---------------------------- 最终评估 ---------------------------- #
    # 同时把 per-sample 预测存成 npz。
    # 这是统计检验的**前提**：只有 4 个 client，client 级 Wilcoxon 的 p 值下界是 0.125，
    # 永远达不到 0.05。主检验必须放在样本级（DeLong / 配对 bootstrap），
    # 那就需要逐样本的 y_true 与 probs，而不只是汇总指标。
    test_per_client = {}
    raw_preds: Dict[str, np.ndarray] = {}
    for c in clients:
        c.load_from_server(state)
        m = c.evaluate("test")
        if m:
            test_per_client[c.name] = m
        loader = c.loaders.get("test")
        if loader is not None:
            y, p = c.predict(loader)
            raw_preds[f"{c.name}__y"] = y
            raw_preds[f"{c.name}__probs"] = p

    external = evaluate_external(clients, state, external_loader, raw_out=raw_preds)

    if raw_preds:
        np.savez_compressed(out_dir / "predictions.npz", **raw_preds)
        LOGGER.info("per-sample 预测已存入 %s（统计检验用）", out_dir / "predictions.npz")

    result = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_cfg": asdict(model_cfg),
        "loss_weights": asdict(loss_w),
        "local_steps": steps,
        "best_val": best,
        "test_per_client": test_per_client,
        "test_summary": aggregate_over_clients(test_per_client, "qwk"),
        "external": external,
        "system": {
            "cum_upload_mb": cum_mb,
            "upload_mb_per_client_per_round": clients[0].model.communication_mb(),
            "rounds_run": len(history),
            "wall_clock_s": time.time() - t_start,
        },
        "history": history,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    LOGGER.info("测试集：%s", {k: round(v["qwk"], 4) for k, v in test_per_client.items()})
    LOGGER.info("汇总：%s", {k: round(v, 4) for k, v in result["test_summary"].items()})
    LOGGER.info("结果已写入 %s", out_dir / "result.json")
    return result


def select_clients(clients, participation: float, rnd: int, seed: int):
    """主实验 participation=1.0（只有 4 个 client，采样会带来巨大方差）。

    D1/D2 的 dropout 压力测试才会把它调低。
    """
    if participation >= 1.0:
        return clients
    k = max(1, int(round(len(clients) * participation)))
    rng = random.Random(seed * 10000 + rnd)
    picked = rng.sample(clients, k)
    LOGGER.info("round %d 参与 client：%s", rnd, [c.name for c in picked])
    return picked


# --------------------------------------------------------------------------- #
def make_fake_manifest(cache_dir: Path, per_client=(400, 120, 200, 40), size: int = 64):
    """``--dry-run`` 用：生成带 client 风格差异的合成眼底图，验证全流程能跑通。

    不同 client 用不同的色偏和亮度，模拟真实的 style shift；
    等级信号编码成中心亮斑的数量，保证任务本身可学。
    """
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    rows = []
    tints = {"eyepacs": (1.0, 0.7, 0.5), "aptos": (0.8, 0.9, 1.0),
             "ddr": (0.9, 1.0, 0.8), "idrid": (1.0, 1.0, 1.0)}
    for client, n in zip(TRAIN_CLIENTS, per_client):
        d = cache_dir / client
        d.mkdir(exist_ok=True)
        for i in range(n):
            grade = int(rng.choice(5, p=[0.5, 0.15, 0.2, 0.08, 0.07]))
            img = rng.randint(20, 60, (size, size, 3)).astype(np.float32)
            for _ in range(grade * 4):
                cy, cx = rng.randint(8, size - 8, 2)
                img[cy - 3:cy + 3, cx - 3:cx + 3] += 120
            img *= np.array(tints[client])[None, None, :]
            path = d / f"{client}_{i}.jpg"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
            split = "train" if i < n * 0.7 else ("val" if i < n * 0.85 else "test")
            rows.append(dict(image_id=f"{client}_{i}", client=client, split=split,
                             dr_grade=grade, patient_id=f"{client}_{i}",
                             source_split="fake", path=str(path)))
    import pandas as pd

    LOGGER.warning("dry-run：使用 %d 张合成图，结果没有任何科学意义，只验证流程", len(rows))
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FedOSP 联邦模拟")
    # 数据
    p.add_argument("--manifest", type=str, default="manifest_cached.csv")
    p.add_argument("--clients", nargs="+", default=TRAIN_CLIENTS)
    p.add_argument("--label-budget", type=str, default=None,
                   help="标签效率实验：'400' 或 '20%%'；默认全量")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    # 模型
    p.add_argument("--backbone", type=str, default="vit_large_patch16_224")
    p.add_argument("--pretrained", type=str, default=None, help="RETFound 权重路径")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-blocks", type=int, default=12)
    p.add_argument("--fsr-block", type=int, default=6)
    # B10 / B11 两个 PEFT 对照基线
    p.add_argument("--full-finetune", action="store_true",
                   help="B10：解冻整个骨干做全量微调（上传量回到 ~1.2GB）")
    p.add_argument("--num-prompts", type=int, default=0,
                   help="B11：visual prompt tuning 的 prompt token 数，0 = 关闭")
    # 消融开关
    p.add_argument("--no-fsr", action="store_true", help="A1")
    p.add_argument("--no-shallow-proto", action="store_true", help="A2")
    p.add_argument("--no-deep-proto", action="store_true", help="A3")
    p.add_argument("--no-personal-ln", action="store_true", help="A7")
    p.add_argument("--ordinal-margin", type=float, default=0.5, help="A4：置 0 退化成普通原型")
    p.add_argument("--lambda-ord", type=float, default=0.5, help="A5：置 0 退回普通 CE")
    p.add_argument("--lambda-proto-grade", type=float, default=0.1)
    p.add_argument("--lambda-proto-style", type=float, default=0.1)
    p.add_argument("--lambda-style", type=float, default=0.05)
    p.add_argument("--lambda-cons", type=float, default=0.1)
    # 联邦
    p.add_argument("--strategy", type=str, default="fedosp")
    p.add_argument("--proto-agg", type=str, default="client_equal",
                   choices=["client_equal", "sqrt", "sample"], help="A6 核心消融")
    p.add_argument("--param-weight", type=str, default="sqrt",
                   choices=["sqrt", "sample", "equal"])
    p.add_argument("--steps-rule", type=str, default="sqrt",
                   choices=["sqrt", "equal", "epoch"], help="A10")
    p.add_argument("--min-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--participation", type=float, default=1.0, help="D1/D2 调这个")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    # 优化
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # 其它
    p.add_argument("--config", type=str, default=None,
                   help="YAML 配置文件；命令行显式给的参数优先级更高")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/debug")
    p.add_argument("--exp-id", type=str, default=None,
                   help="实验 ID，会写进 result.json 供 aggregate_results.py 分组")
    p.add_argument("--dry-run", action="store_true", help="用合成数据+微型ViT跑通全流程")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def apply_config_file(parser: argparse.ArgumentParser, args, argv: Optional[List[str]]):
    """把 YAML 里的值填进去，但**不覆盖命令行显式传入的参数**。"""
    if not args.config:
        return args
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("读取 --config 需要 pyyaml：pip install pyyaml") from exc

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    given = set()
    for token in (argv if argv is not None else sys.argv[1:]):
        if token.startswith("--"):
            given.add(token.lstrip("-").split("=")[0].replace("-", "_"))
    for key, value in cfg.items():
        key = key.replace("-", "_")
        if not hasattr(args, key):
            LOGGER.warning("配置文件里的 %r 不是已知参数，已忽略", key)
            continue
        if key in given:
            continue  # 命令行优先
        setattr(args, key, value)
    LOGGER.info("已加载配置 %s", args.config)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = apply_config_file(parser, args, argv)
    if args.dry_run:
        # 用微型 ViT + 极小配置，几十秒内跑完；显式传的 --rounds 仍然生效
        args.backbone = "debug_vit"
        args.img_size = 64
        args.batch_size = 8
        args.num_workers = 0
        args.max_steps = 12
        args.min_steps = 4
        args.lora_blocks = 4
        if args.rounds == parser.get_default("rounds"):
            args.rounds = 6
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
