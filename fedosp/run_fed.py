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
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .data.dataset import load_manifest, local_steps, make_client_loaders
from .fed.client import LocalClient
from .fed.strategies import STRATEGIES, ServerState, build_strategy
from .losses import ORDINAL_HEAD_LOSSES, ORDINAL_LOSSES, LossWeights
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


def eval_client(
    client: LocalClient, state: ServerState, eval_mode: str,
    split: str = "val", loader=None, want_raw: bool = False,
):
    """按策略的 ``eval_mode`` 决定"评估哪个模型"，返回 ``(metrics, (y, probs) | None)``。

    这个函数是个性化联邦方法能否被正确评估的唯一关口，三种模式见
    :attr:`FedStrategy.eval_mode`。**不要在别处直接写 ``load_from_server`` + ``evaluate``**，
    那样会让 FedALA / Ditto 静默退化成 FedAvg。
    """
    if eval_mode == "global":
        client.load_from_server(state)          # 不传 cfg：评估时不做 ALA 插值/MOON 快照
    elif eval_mode not in ("local", "personal"):
        raise ValueError(f"未知 eval_mode={eval_mode!r}")

    ctx = client.personal_model() if eval_mode == "personal" else nullcontext(None)
    with ctx as switched:
        if eval_mode == "personal" and switched is False:
            LOGGER.warning(
                "[%s] eval_mode=personal 但个人模型还不存在（第一轮？），本次用全局模型评估",
                client.name,
            )
        metrics = client.evaluate(split, loader=loader)
        raw = None
        if want_raw:
            # 只在最终测试时才多过一遍拿 per-sample 预测；
            # 每轮验证都做会把评估开销翻倍
            lo = loader if loader is not None else client.loaders.get(split)
            if lo is not None:
                raw = client.predict(lo)
    return metrics, raw


#: 方案定义的正式实验配置。任何一项不满足，这次 run 就只能算 pilot。
MAIN_TIER_REQUIREMENTS = {
    "backbone": "vit_large_patch16_224",
    "img_size": 224,
    "min_rounds": 100,
}


def result_tier(args, model_cfg: FedOSPConfig, weight_source: str) -> Dict[str, object]:
    """判定这次 run 算 ``main`` 还是 ``pilot``，并列出不达标的具体原因。

    为什么需要这个字段：降规模 run（小骨干、少轮数、抽样训练集、非 RETFound 权重）
    产出的 ``result.json`` 在**格式上与正式结果完全相同**。``stage`` 那道闸门只拦
    "随机骨干 + main"这一种情况，拦不住"ViT-Small 跑 30 轮"这种。等到写论文时翻出
    几十个 run 目录，靠文件名和记忆去分辨哪份能引用，是必然出错的。

    所以这里把方案对"正式实验"的定义写成可执行的判定，并**把不达标的原因逐条列出**
    —— 不只是给一个布尔值，而是让人一眼看到差在哪。

    Returns:
        ``{"tier": "main"|"pilot", "tier_violations": [...]}``
    """
    v: List[str] = []
    if not str(weight_source).startswith("retfound"):
        v.append(f"骨干权重来源是 {weight_source!r}，非 RETFound")
    if getattr(args, "stage", "smoke") != "main":
        v.append(f"stage={getattr(args, 'stage', 'smoke')!r}，非 main")
    if model_cfg.backbone != MAIN_TIER_REQUIREMENTS["backbone"]:
        v.append(
            f"骨干是 {model_cfg.backbone}，方案规定 "
            f"{MAIN_TIER_REQUIREMENTS['backbone']}"
        )
    if model_cfg.img_size != MAIN_TIER_REQUIREMENTS["img_size"]:
        v.append(f"分辨率 {model_cfg.img_size}，方案规定 {MAIN_TIER_REQUIREMENTS['img_size']}")
    if int(args.rounds) < MAIN_TIER_REQUIREMENTS["min_rounds"]:
        v.append(f"只跑了 {args.rounds} 轮，方案规定 ≥{MAIN_TIER_REQUIREMENTS['min_rounds']}")
    if getattr(args, "max_train_per_client", None):
        v.append(f"训练集被降规模到每 client ≤{args.max_train_per_client} 张")
    if getattr(args, "train_fraction", None):
        v.append(f"训练集被按比例降规模到 {args.train_fraction:.0%}")
    if getattr(args, "dry_run", False):
        v.append("dry_run：用的是合成假数据")
    return {"tier": "pilot" if v else "main", "tier_violations": v}


def default_device() -> str:
    """自动探测可用设备：cuda > mps > cpu。

    加 mps 分支的理由是实测差距很大：本机（M 系列，12 核 / 36 GB）上
    ViT-Large LoRA 训练 cpu 只有 3.5 图/s，mps 有 11.4 图/s，**3.3 倍**。
    原来的默认值只认 cuda，在 Mac 上会静默退到 cpu，白白慢 3 倍。
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
def build_clients(
    args, manifest, model_cfg: FedOSPConfig, loss_w: LossWeights, freq_bank=None,
) -> List[LocalClient]:
    clients: List[LocalClient] = []
    # 曾经这里是 `style_aug = args.strategy == "fedosp"`，等于只给 FedOSP 开辅助正则，
    # baseline 拿不到 —— 这正是设计文档 §3.3 要消除的不公平比较。现在由 --aux-reg
    # 统一控制，对所有策略取同一个值。
    style_aug = bool(args.aux_reg)
    if freq_bank is not None and style_aug:
        # B16 的第二视图来自幅度谱插值，L_style/L_cons 的第二视图来自颜色抖动，
        # 两者抢同一个位置。与其静默丢一个，不如直接拦住。
        raise ValueError(
            "--strategy feddg 与 --aux-reg 不能同时用：两者都要占用第二视图。"
            "跑 ELCFS 基线时请关掉 --aux-reg。"
        )
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
            freq_bank=freq_bank,
            freq_ratio=getattr(args, "feddg_ratio", 0.01),
            max_train=getattr(args, "max_train_per_client", None),
            train_fraction=getattr(args, "train_fraction", None),
            min_train=getattr(args, "min_train_per_client", 500),
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
        imagenet_pretrained=getattr(args, "imagenet_pretrained", False),
        stage=getattr(args, "stage", "smoke"),
        full_finetune=getattr(args, "full_finetune", False),
        num_prompts=getattr(args, "num_prompts", 0),
        # 阈值式序数范式必须同时换输出头，否则损失会拿到 K 列而不是 K-1 列。
        # 这个联动放在这里做，而不是让使用者记住要同时传两个参数 ——
        # 忘了传头的话不会报错，只会安静地算出一个错的东西。
        ordinal_head=(
            args.ord_type if args.ord_type in ORDINAL_HEAD_LOSSES else "none"
        ),
    )
    loss_w = LossWeights(
        ord=args.lambda_ord,
        proto_grade=args.lambda_proto_grade,
        proto_style=args.lambda_proto_style,
        style=args.lambda_style,
        cons=args.lambda_cons,
        ordinal_margin=args.ordinal_margin,
        ord_type=args.ord_type,
    )

    # B16 ELCFS：策略需要共享幅度谱库时先建库。这一步是**真实的跨中心数据外传**，
    # 体积记进 result.json 供 T5 引用。
    freq_bank = None
    strategy_cls = STRATEGIES[args.strategy]
    if getattr(strategy_cls, "needs_amplitude_bank", False):
        from .data.freq_aug import AmplitudeBank
        freq_bank = AmplitudeBank.build(
            manifest,
            img_size=model_cfg.img_size,
            per_client=args.feddg_per_client,
            seed=args.seed,
            clients=args.clients,
        )

    clients = build_clients(args, manifest, model_cfg, loss_w, freq_bank=freq_bank)
    # aux_reg 传给所有策略（不只 fedosp），保证 L_style/L_cons 的开关对各方法一视同仁
    strategy_kwargs = {
        "mu": args.fedprox_mu,
        "aux_reg": bool(args.aux_reg),
        "tau2_override": args.tau2_override,
        # 下面几个只有对应策略会读，其余策略从 self.cfg 里忽略掉
        "moon_mu": args.moon_mu,
        "moon_tau": args.moon_tau,
        "ditto_lambda": args.ditto_lambda,
        "ala_lr": args.ala_lr,
        "ala_iters": args.ala_iters,
        "ala_last_n": args.ala_last_n,
        "feddg_ratio": args.feddg_ratio,
        "lr": args.lr,               # q-FedAvg 的 Lipschitz 估计 L = 1/lr
    }
    if args.strategy == "qfedavg":
        strategy_kwargs["q"] = args.q
    if args.strategy == "fedosp":
        strategy_kwargs.update(
            proto_agg_mode=args.proto_agg, param_weight_mode=args.param_weight
        )
    strategy = build_strategy(args.strategy, **strategy_kwargs)

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
            # cfg 必须传进去：FedALA 的全部机制都在下发这一步，
            # MOON 也要在参数被覆盖前记下上一轮的本地模型
            c.load_from_server(state, cfg)
            updates.append(c.local_train(steps[c.name], cfg))
        state = strategy.aggregate(updates, state)
        cum_mb += sum(u.upload_mb() for u in updates)

        # ---- 验证：必须用 macro-over-client，不能用样本加权 ----
        per_client = {}
        for c in clients:
            m, _ = eval_client(c, state, strategy.eval_mode, "val")
            if m:
                per_client[c.name] = m
        summary = aggregate_over_clients(per_client, "qwk")
        row = {
            "round": rnd, "cum_upload_mb": cum_mb,
            "elapsed_s": time.time() - t_start,
            **summary,
            **{f"val_qwk_{k}": v["qwk"] for k, v in per_client.items()},
        }
        # 把服务器端的聚合诊断（C2 的 n_eff / tau^2）并进同一行，否则它们只活在
        # 日志里、进不了 result.json，事后无法分析。strategy.history 与本地 history
        # 是逐轮一一对应的。
        if strategy.history:
            row.update({k: v for k, v in strategy.history[-1].items()
                        if k not in ("round",)})
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
        m, raw = eval_client(c, state, strategy.eval_mode, "test", want_raw=True)
        if m:
            test_per_client[c.name] = m
        if raw is not None:
            raw_preds[f"{c.name}__y"], raw_preds[f"{c.name}__probs"] = raw

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
            # B16 ELCFS 需要把原始图像的幅度谱传出本地，这是 FSR 没有的额外通信
            # 与隐私开销。T5 单列一行，隐私讨论直接引用这个数。
            "amplitude_bank_mb": freq_bank.payload_mb() if freq_bank else 0.0,
            "amplitude_bank_n": len(freq_bank.amps) if freq_bank else 0,
            # 本地计算量：MOON 每步 3 次前向、Ditto 本地步数翻倍，都在这里体现
            "local_steps_total": sum(steps.values()),
            "compute_steps_total": sum(
                u.compute_steps or u.metrics.get("steps", 0) for u in updates
            ),
        },
        # ---- 事后审计字段：判断这次 run 到底算不算正式结果 ----
        "provenance": {
            # 骨干权重来源：retfound:<path> / imagenet / random / random:debug_vit
            # 只有 retfound:* 才是方案里定义的正式实验
            "weight_source": clients[0].model.weight_source,
            "stage": getattr(args, "stage", "smoke"),
            # 辅助正则是否开启；开启时对所有策略同等生效
            "aux_reg": bool(args.aux_reg),
            # 序数范式与输出头：B17 交叉组靠这两项区分，且必须一致
            "ord_type": args.ord_type,
            "ordinal_head": model_cfg.ordinal_head,
            # 评估用的是全局模型还是个性化模型。FedALA/Ditto 若误记成 global，
            # 说明个性化被评估流程冲掉了，结果无效
            "eval_mode": strategy.eval_mode,
            # ---- 结果等级：只有全部条件满足才算 main，否则一律 pilot ----
            # 这个字段的存在理由与 stage 硬闸门相同：降规模 run（小骨干 / 少轮数 /
            # 抽样训练集）同样产出格式完全正常的 result.json，半年后翻出来根本
            # 分不清哪份是正式结果。把判定写成代码而不是靠记忆。
            **result_tier(args, model_cfg, clients[0].model.weight_source),
            # 读图失败张数。非 0 意味着有零图混进了训练，结果需打折看
            "n_failed_reads": sum(
                ds.failed_reads
                for c in clients
                for ds in (
                    getattr(ld, "dataset", None) for ld in c.loaders.values() if ld is not None
                )
                if hasattr(ds, "failed_reads")
            ),
        },
        "history": history,
    }
    prov = result["provenance"]
    if prov["tier"] == "pilot":
        LOGGER.warning(
            "⚠ 本次 run 等级 = PILOT，不能作为正式实验结果引用。不达标项：\n    - %s",
            "\n    - ".join(prov["tier_violations"]),
        )
    else:
        LOGGER.info("✓ 本次 run 等级 = MAIN，满足方案对正式实验的全部要求")
    if prov["n_failed_reads"]:
        LOGGER.warning("⚠ 本次 run 有 %d 张图读取失败（已用零图替代）", prov["n_failed_reads"])
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
    p.add_argument("--train-fraction", type=float, default=None,
                   help="pilot 降规模：各 client 训练集**按同一比例**抽样（如 0.2）。"
                        "★ 必须用比例而不是统一上限：C2 的收益完全来自客户端规模不平衡，"
                        "统一截到每院 2000 张会把 n_eff 从 1.75 推到 3.34，"
                        "等于人为消掉 C2 的改进空间，pilot 就问不出它有没有用了。"
                        "按比例抽样保留 n_eff≈1.90，结论仍可迁移")
    p.add_argument("--min-train-per-client", type=int, default=500,
                   help="配合 --train-fraction 的下限：≤这个数的 client 整体保全。"
                        "默认 500 是为了让 IDRiD（372 张）完全不被削 —— 按比例抽会把它的"
                        "稀有等级抽没（grade 1 全院仅 20 张，20%% 只剩 4 张），"
                        "而它只占全部机时的 6%%，削它几乎没有收益")
    p.add_argument("--max-train-per-client", type=int, default=None,
                   help="每个 client 的训练集**统一上限**。⚠ 一般不要用它做 pilot 降规模，"
                        "理由见 --train-fraction。保留它是为了个别需要等规模联邦的诊断实验")
    p.add_argument("--label-budget", type=str, default=None,
                   help="标签效率实验：'400' 或 '20%%'；默认全量")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    # 模型
    p.add_argument("--backbone", type=str, default="vit_large_patch16_224")
    p.add_argument("--pretrained", type=str, default=None, help="RETFound 权重路径")
    p.add_argument("--imagenet-pretrained", action="store_true",
                   help="降级路径（方案 R-B）：RETFound 权重未获批时用 timm 的 ImageNet "
                        "预训练。论文里必须声明，且不能再声称『眼科基础模型』")
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
    p.add_argument("--aux-reg", action="store_true",
                   help="开启 L_style / L_cons 两项辅助正则。默认关闭：它们不对应 "
                        "P1/P2/P3，且只给 FedOSP 开会造成不公平比较（设计文档 §3.3）。"
                        "开启时对**所有策略**同等生效，用于做带/不带对照")
    # 联邦
    p.add_argument("--strategy", type=str, default="fedosp", choices=sorted(STRATEGIES))
    # ---- B12–B16 新基线的超参 ----
    p.add_argument("--moon-mu", type=float, default=1.0,
                   help="B12 MOON 对比损失权重（原文 DR 类任务常用 1.0）")
    p.add_argument("--moon-tau", type=float, default=0.5, help="B12 MOON 温度")
    p.add_argument("--ditto-lambda", type=float, default=0.1,
                   help="B15 Ditto 个人模型朝全局模型的 prox 强度")
    p.add_argument("--ala-lr", type=float, default=0.1, help="B13 FedALA 学插值权重 W 的步长")
    p.add_argument("--ala-iters", type=int, default=5,
                   help="B13 每轮更新 W 的迭代数（第 1 轮会自动放大到 >=20）")
    p.add_argument("--ala-last-n", type=int, default=4,
                   help="B13 只对最高的这几个 block 做元素级插值，更低层直接覆盖")
    p.add_argument("--q", type=float, default=1.0,
                   help="B14 q-FedAvg 的公平性指数。0 = 精确退化为 FedAvg，越大越偏向损失高的 client")
    p.add_argument("--feddg-per-client", type=int, default=10,
                   help="B16 ELCFS 每个 client 贡献进共享幅度谱库的图片数。"
                        "注意这些谱要**真的传出本地**，体积会记进 result.json 的 amplitude_bank_mb")
    p.add_argument("--feddg-ratio", type=float, default=0.01,
                   help="B16 ELCFS 低频掩码半宽占比（原文量级 0.01）")
    # ---- B17 交叉组：序数范式 ----
    p.add_argument("--ord-type", type=str, default="emd", choices=list(ORDINAL_LOSSES),
                   help="序数范式（B17 交叉组 / A5 消融）。emd = 本文默认；"
                        "binomial 与 ordinal_encoding 是 Corbetta MIDL'25 用的两种；"
                        "coral 在此之上用共享权重保证秩单调性。"
                        "coral / ordinal_encoding 会自动把输出头换成 K-1 维")
    p.add_argument("--proto-agg", type=str, default="precision",
                   choices=["client_equal", "sqrt", "sample", "precision"],
                   help="A6 核心消融。precision = 随机效应最优权重（C2，本文默认）；"
                        "sample / client_equal 分别是它 tau^2=0 与 tau^2->inf 的极限特例")
    p.add_argument("--tau2-override", type=float, default=None,
                   help="固定 between-client 方差 tau^2 而不用 DerSimonian-Laird 估计，"
                        "用于 A6 的 tau^2 扫描（验证退化行为）")
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
    p.add_argument("--device", type=str, default=default_device(),
                   help="cuda / mps / cpu。默认自动探测；Apple Silicon 上 mps 比 cpu 快约 3.3x")
    # 其它
    p.add_argument("--config", type=str, default=None,
                   help="YAML 配置文件；命令行显式给的参数优先级更高")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/debug")
    p.add_argument("--exp-id", type=str, default=None,
                   help="实验 ID，会写进 result.json 供 aggregate_results.py 分组")
    p.add_argument("--stage", type=str, default="smoke", choices=["smoke", "main"],
                   help="main = 正式实验：随机初始化骨干会直接报错，必须传 --pretrained "
                        "或显式 --imagenet-pretrained。smoke = 调试，仅告警")
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
