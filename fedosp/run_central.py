"""集中式训练：B1 Local-only / B2 Centralized oracle，以及**第 2 周的健全性检查**。

这个脚本的首要用途不是拿名次，而是回答一个问题：
**我的数据管线和模型加载是不是对的？**

判据是方案 9.3 的文献锚点：

* RETFound 原文微调 AUROC：APTOS 0.943、IDRiD 0.822、MESSIDOR-2 0.884
* Messidor-2 集中式 quadratic kappa 参考值 ≈ 0.78

**对不上就不要往联邦实验走**，先查划分、标签、预处理。

用法::

    # B1 Local-only：每个 client 各训一个模型
    python -m fedosp.run_central --mode local --pretrained RETFound_mae.pth --out runs/B1_s0

    # B2 Centralized oracle：四集汇总
    python -m fedosp.run_central --mode pooled --pretrained RETFound_mae.pth --out runs/B2_s0

    # 健全性检查（只在 APTOS 上训，跟 RETFound 原文对数）
    python -m fedosp.run_central --mode local --clients aptos --check-anchors
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data.dataset import FundusDataset, apply_label_budget, load_manifest, make_client_loaders
from .losses import FedOSPLoss, LossWeights
from .metrics import (
    aggregate_over_clients,
    check_against_anchors,
    evaluate_predictions,
)
from .models.retfound_lora import FedOSPConfig, build_model
from .run_fed import HELDOUT_CLIENT, TRAIN_CLIENTS, make_fake_manifest, set_seed

LOGGER = logging.getLogger("run_central")


def train_one(
    frame: pd.DataFrame,
    tag: str,
    args,
    model_cfg: FedOSPConfig,
) -> Dict:
    """在给定的 manifest 切片上做一次集中式训练，返回各 split 的指标。"""
    from torch.utils.data import DataLoader

    def _loader(split: str, shuffle: bool) -> Optional[DataLoader]:
        part = frame[frame["split"] == split]
        if not len(part):
            return None
        if split == "train":
            part = apply_label_budget(part, args.label_budget, args.seed)
        ds = FundusDataset(part, img_size=model_cfg.img_size, train=(split == "train"))
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True,
            drop_last=shuffle and len(ds) > args.batch_size,
        )

    train_loader = _loader("train", True)
    val_loader = _loader("val", False)
    test_loader = _loader("test", False)
    if train_loader is None:
        raise RuntimeError(f"[{tag}] 没有训练数据")

    model = build_model(copy.deepcopy(model_cfg)).to(args.device)
    counts = train_loader.dataset.class_counts()
    LOGGER.info("[%s] n_train=%d 分布=%s", tag, len(train_loader.dataset), counts.astype(int).tolist())

    # 集中式基线不用原型/风格项，就是 CBCE + ordinal EMD
    criterion = FedOSPLoss(
        counts, LossWeights(ord=args.lambda_ord, proto_grade=0, proto_style=0, style=0, cons=0)
    ).to(args.device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    amp = args.amp and str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best = {"epoch": -1, "qwk": -np.inf}
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_b = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x)
                loss, _ = criterion(out, y)
            if not torch.isfinite(loss):
                LOGGER.error("[%s] epoch %d 出现非有限 loss，跳过该 batch", tag, epoch)
                continue
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step()
            running += float(loss.detach())
            n_b += 1
        sched.step()

        val = predict_and_eval(model, val_loader, args) if val_loader else {}
        LOGGER.info("[%s] epoch %2d | loss %.4f | val qwk %.4f auroc %.4f",
                    tag, epoch, running / max(n_b, 1),
                    val.get("qwk", float("nan")), val.get("referable_auroc", float("nan")))
        if val.get("qwk", -np.inf) > best["qwk"]:
            best = {"epoch": epoch, **val}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best["epoch"] >= args.patience:
            LOGGER.info("[%s] 早停于 epoch %d（最好 %d）", tag, epoch, best["epoch"])
            break

    if best_state:
        model.load_state_dict(best_state)
    return {
        "tag": tag,
        "best_val": best,
        "test": predict_and_eval(model, test_loader, args) if test_loader else {},
        "wall_clock_s": time.time() - t0,
        "_model": model,
    }


@torch.no_grad()
def predict_and_eval(model, loader, args) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    probs, labels = [], []
    amp = args.amp and str(args.device).startswith("cuda")
    for batch in loader:
        x, y = batch[0], batch[-1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x.to(args.device)).logits
        probs.append(model.class_probs(logits).cpu().numpy())
        labels.append(y.numpy())
    return evaluate_predictions(np.concatenate(labels), np.concatenate(probs))


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
        use_fsr=False,            # 集中式基线不带 FSR / 原型，保持"纯骨干 + LoRA"
        use_shallow_proto=False,
        use_deep_proto=False,
        lora_rank=args.lora_rank,
        lora_last_n_blocks=args.lora_blocks,
        personal_layernorm=False,
        pretrained_path=args.pretrained,
    )

    results: Dict[str, Dict] = {}
    models: Dict[str, object] = {}

    if args.mode == "pooled":
        pooled = manifest[manifest["client"].isin(args.clients)]
        r = train_one(pooled, "pooled", args, model_cfg)
        models["pooled"] = r.pop("_model")
        results["pooled"] = r
        # 汇总模型还要在每个 client 的 test 上单独报，才能跟联邦主表对齐
        per_client = {}
        for c in args.clients:
            loaders = make_client_loaders(
                manifest, c, img_size=model_cfg.img_size,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            if "test" in loaders:
                per_client[c] = predict_and_eval(models["pooled"], loaders["test"], args)
        results["per_client_test"] = per_client
        results["summary"] = aggregate_over_clients(per_client, "qwk")
    else:
        per_client = {}
        for c in args.clients:
            sub = manifest[manifest["client"] == c]
            if not len(sub):
                LOGGER.warning("client %s 无数据，跳过", c)
                continue
            r = train_one(sub, c, args, model_cfg)
            models[c] = r.pop("_model")
            results[c] = r
            per_client[c] = r["test"]
        results["per_client_test"] = per_client
        results["summary"] = aggregate_over_clients(per_client, "qwk")

    # -------- 未见中心外测 --------
    if not args.dry_run:
        ext_loaders = make_client_loaders(
            manifest, HELDOUT_CLIENT, img_size=model_cfg.img_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        ext_loader = ext_loaders.get("test")
        if ext_loader is not None:
            results["external"] = {
                name: predict_and_eval(m, ext_loader, args) for name, m in models.items()
            }

    # -------- 与文献锚点对数（第 2 周的关键动作）--------
    if args.check_anchors:
        anchor_input = dict(results.get("per_client_test", {}))
        for name, ext in (results.get("external") or {}).items():
            anchor_input.setdefault(HELDOUT_CLIENT, ext)
        kw = {}
        if args.anchor_tol is not None:
            kw["tol"] = args.anchor_tol
        if args.anchor_z is not None:
            kw["z"] = args.anchor_z
        results["anchor_check"] = check_against_anchors(
            anchor_input, "retfound_finetune_auroc", "referable_auroc", **kw
        )
        off = [k for k, v in results["anchor_check"].items() if not v.startswith("OK")]
        if off:
            LOGGER.warning(
                "以下数据集偏离文献锚点：%s —— 先查划分/标签/预处理，不要进联邦实验", off
            )
            # ★ 方向本身是重要线索，不要只看"过没过"
            for k in off:
                v = results["anchor_check"][k]
                if "delta=+" in v:
                    LOGGER.warning(
                        "  [%s] 实测**高于**文献。用 0.23%% 参数的 LoRA 超过全量微调的 "
                        "ViT-L 并不合理，优先查两件事：① 指标定义是否一致"
                        "（我们报的是 referable(≥2) 二分类 AUROC，文献若报 5 类 macro "
                        "one-vs-rest 会系统性偏低）；② 划分是否泄漏"
                        "（APTOS 无 patient_id，一图一人的假设若不成立，同一病人双眼会跨 split）",
                        k,
                    )
                else:
                    LOGGER.warning(
                        "  [%s] 实测**低于**文献。先看偏差的 SE 倍数：小于 2 个 SE 基本是"
                        "小测试集的正常波动（IDRiD n=103 时 SE≈0.040）；"
                        "确实超出才查 epoch 数是否够、LoRA 容量是否不足", k,
                    )
        else:
            LOGGER.info("全部对上文献锚点，数据管线可以放行 ✅")

    results.pop("_model", None)
    (out_dir / "result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    LOGGER.info("结果已写入 %s", out_dir / "result.json")
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="集中式基线 / 健全性检查")
    p.add_argument("--manifest", type=str, default="manifest_cached.csv")
    p.add_argument("--mode", choices=["local", "pooled"], default="local",
                   help="local = B1 各训各的；pooled = B2 汇总 oracle")
    p.add_argument("--clients", nargs="+", default=TRAIN_CLIENTS)
    p.add_argument("--label-budget", type=str, default=None)
    p.add_argument("--backbone", type=str, default="vit_large_patch16_224")
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-blocks", type=int, default=12)
    p.add_argument("--lambda-ord", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/central")
    p.add_argument("--exp-id", type=str, default=None,
                   help="实验 ID，会写进 result.json 供 aggregate_results.py 分组")
    p.add_argument("--check-anchors", action="store_true",
                   help="与方案 9.3 的文献锚点对数")
    p.add_argument("--anchor-tol", type=float, default=None,
                   help="固定容差（绝对点数）。**默认不传** —— 默认走随测试集规模缩放的 "
                        "2.5×标准误 判据。固定容差在这里是错的：各 client 测试集规模差 68 倍"
                        "（IDRiD 103 张 vs EyePACS 7000 张），±0.02 在 IDRiD 上只有 0.48 个 SE，"
                        "实现完全正确也约 63% 概率判 FAIL")
    p.add_argument("--anchor-z", type=float, default=None,
                   help="容差取几个标准误，默认 2.5（双侧约 p=0.012）")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-12s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.dry_run:
        args.backbone = "debug_vit"
        args.img_size = 64
        args.batch_size = 8
        args.num_workers = 0
        args.lora_blocks = 4
        if args.epochs == parser.get_default("epochs"):
            args.epochs = 6
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
