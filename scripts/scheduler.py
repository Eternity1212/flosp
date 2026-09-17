#!/usr/bin/env python3
"""多 GPU 任务池调度器：把实验矩阵铺到所有可用 GPU 上跑完。

**为什么需要它**：49 个配置 × 3 seeds ≈ 143 GPU 小时。串行跑要 6 天，
4 卡并行 1.5 天。但手工 ``CUDA_VISIBLE_DEVICES=0 python ... &`` 有三个问题：
断了不知道断在哪、重跑会把已完成的再跑一遍、一个配置崩了后面全不跑。

这个调度器解决的正是这三件事：

* **断点续跑**：已有 ``result.json`` 的配置默认跳过（``--force`` 可覆盖）
* **失败重试**：非零退出自动重试 ``--retries`` 次，仍失败则记进 ``failed.txt`` 并继续
* **GPU 亲和**：每张卡一个 worker 线程，卡空出来立刻领下一个任务
* **单卡多任务**：``--jobs-per-gpu 2`` 让 80GB 卡同时跑两个实验

用法（一般由 ``run_all.sh`` 调用，也可单独用）::

    python scripts/scheduler.py --stage all --gpus 0,1,2,3
    python scripts/scheduler.py --stage main --gpus 0 --dry-run     # 只打印命令
    python scripts/scheduler.py --stage ablation --gpus 0,1 --retries 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("scheduler")

#: 阶段 → 该阶段包含的实验 ID 前缀。顺序即执行顺序。
#: 前缀必须**互不为前缀**，否则 ``--stage all`` 会把同一个配置跑两遍。
#: （早前用 "a" 表示消融、"a12" 表示骨干，结果 a12 被两个阶段同时匹配到。）
STAGES = {
    # 两道前置闸门，都是"不过就停"：
    #   s_*  文献锚点核对（数据管线是否正确）
    #   b0_* B0 闸门（n_eff 口径是否正确 + 3 个 seed 的统计功效够不够）
    # 放同一阶段是因为它们必须一起在正式矩阵之前跑完。
    "sanity": ["s_", "b0_"],
    "main": ["base_", "m_", "ord_", "comb_"],  # 基线 + 主方法 + B17 交叉 + 组合
    "ablation": ["abl_"],        # 组件消融
    "label": ["lab_"],           # 标签效率曲线
    "robust": ["rob_"],          # 参与率 / dropout 压力测试
    "backbone": ["bb_"],         # 骨干替换
    # 本机 pilot（configs/pilot_local.csv）。产出一律是 tier="pilot"，不可引用。
    "pilot": ["p_"],
}


@dataclass
class Job:
    """一个待跑的实验。"""

    exp_id: str
    seed: int
    cmd: List[str]
    out_dir: Path
    stage: str = ""
    attempts: int = 0
    status: str = "pending"
    gpu: Optional[str] = None
    seconds: float = 0.0

    @property
    def tag(self) -> str:
        return f"{self.exp_id}_seed{self.seed}"

    @property
    def done(self) -> bool:
        return (self.out_dir / "result.json").exists()


class Scheduler:
    """GPU 任务池。每张卡一个 worker，从共享队列里取任务。"""

    def __init__(
        self,
        gpus: Sequence[str],
        jobs_per_gpu: int = 1,
        retries: int = 1,
        log_dir: Path = ROOT / "logs",
        dry_run: bool = False,
        timeout_h: float = 12.0,
    ) -> None:
        self.slots = [g for g in gpus for _ in range(jobs_per_gpu)]
        self.retries = retries
        self.log_dir = log_dir
        self.dry_run = dry_run
        self.timeout = timeout_h * 3600
        self.q: "queue.Queue[Job]" = queue.Queue()
        self.lock = threading.Lock()
        self.finished: List[Job] = []
        self.failed: List[Job] = []
        self.t0 = time.time()
        self._total = 0

    def submit(self, jobs: Sequence[Job]) -> None:
        for j in jobs:
            self.q.put(j)
        self._total += len(jobs)

    # ------------------------------------------------------------------ #
    def _run_one(self, job: Job, gpu: str) -> bool:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"{job.tag}.log"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # 每个进程限一个 OMP 线程，否则多任务并行时 CPU 会被抢爆
        env.setdefault("OMP_NUM_THREADS", "4")
        env["PYTHONUNBUFFERED"] = "1"

        job.gpu = gpu
        t0 = time.time()
        if self.dry_run:
            LOGGER.info("[dry] GPU%s %s: %s", gpu, job.tag, " ".join(shlex.quote(c) for c in job.cmd))
            job.status = "dry"
            return True

        try:
            with open(log_file, "w", encoding="utf-8") as fh:
                fh.write(f"# {' '.join(shlex.quote(c) for c in job.cmd)}\n")
                fh.write(f"# CUDA_VISIBLE_DEVICES={gpu}\n\n")
                fh.flush()
                proc = subprocess.run(
                    job.cmd, cwd=ROOT, env=env, stdout=fh,
                    stderr=subprocess.STDOUT, timeout=self.timeout,
                )
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            LOGGER.error("[GPU%s] %s 超过 %.1f 小时未结束，已终止", gpu, job.tag, self.timeout / 3600)
            ok = False
        except Exception as exc:
            LOGGER.error("[GPU%s] %s 启动失败：%s", gpu, job.tag, exc)
            ok = False

        job.seconds = time.time() - t0
        # 进程退出码为 0 还不够，必须真的写出了 result.json
        if ok and not job.done:
            LOGGER.error("[GPU%s] %s 退出码 0 但没有 result.json，视为失败（看 %s）",
                         gpu, job.tag, log_file)
            ok = False
        return ok

    def _worker(self, gpu: str) -> None:
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                return

            job.attempts += 1
            ok = self._run_one(job, gpu)

            with self.lock:
                if ok:
                    job.status = "done"
                    self.finished.append(job)
                    n = len(self.finished) + len(self.failed)
                    eta = ""
                    if n and not self.dry_run:
                        rate = (time.time() - self.t0) / n
                        left = self._total - n
                        eta = f"，预计剩余 {rate * left / 3600:.1f} h"
                    LOGGER.info("✓ [GPU%s] %s（%.1f min）  进度 %d/%d%s",
                                gpu, job.tag, job.seconds / 60, n, self._total, eta)
                elif job.attempts <= self.retries:
                    LOGGER.warning("↻ [GPU%s] %s 失败，重试第 %d 次",
                                   gpu, job.tag, job.attempts)
                    self.q.put(job)
                else:
                    job.status = "failed"
                    self.failed.append(job)
                    LOGGER.error("✗ [GPU%s] %s 重试 %d 次仍失败，跳过",
                                 gpu, job.tag, self.retries)
            self.q.task_done()

    def run(self) -> int:
        if self.q.empty():
            LOGGER.info("没有待跑任务")
            return 0
        LOGGER.info("开始调度 %d 个任务，%d 个并发槽位（GPU: %s）",
                    self._total, len(self.slots), ",".join(sorted(set(self.slots))))
        threads = [threading.Thread(target=self._worker, args=(g,), daemon=True)
                   for g in self.slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        dt = (time.time() - self.t0) / 3600
        LOGGER.info("=" * 68)
        LOGGER.info("完成 %d / 失败 %d，总耗时 %.2f h", len(self.finished), len(self.failed), dt)
        if self.failed:
            f = self.log_dir / "failed.txt"
            f.write_text("\n".join(
                f"{j.tag}\t{' '.join(shlex.quote(c) for c in j.cmd)}" for j in self.failed
            ), encoding="utf-8")
            LOGGER.error("失败清单已写入 %s，逐个日志在 %s/<tag>.log", f, self.log_dir)
            LOGGER.error("修好之后重跑同一条命令即可，已完成的会自动跳过")
        return 1 if self.failed else 0


# --------------------------------------------------------------------------- #
def build_jobs(
    matrix: Path,
    stage: str,
    runs_dir: Path,
    seeds: Sequence[int],
    pretrained: Optional[str],
    data_root: Path,
    manifest: Path,
    force: bool = False,
    extra: Sequence[str] = (),
) -> List[Job]:
    """读实验矩阵 CSV，展开成 Job 列表。

    Args:
        stage: ``all`` 或 :data:`STAGES` 里的键。
        force: 为 True 时连已有 result.json 的配置也重跑。

    Returns:
        待跑的 Job（已过滤掉完成的）。
    """
    import csv

    if not matrix.exists():
        raise FileNotFoundError(
            f"找不到实验矩阵 {matrix}。它应该是 10_实验矩阵与排期.csv 的副本，"
            "见 README 的「一键跑完」一节。"
        )

    prefixes = None if stage == "all" else STAGES.get(stage)
    if prefixes is None and stage != "all":
        raise ValueError(f"不认识的 stage={stage}，可选 {['all'] + list(STAGES)}")

    jobs: List[Job] = []
    skipped = 0
    with open(matrix, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            exp_id = (row.get("exp_id") or row.get("id") or "").strip()
            if not exp_id or exp_id.startswith("#"):
                continue
            if prefixes and not any(exp_id.startswith(p) for p in prefixes):
                continue

            which = _stage_of(exp_id)
            row_seeds = _parse_seeds(row.get("seeds"), seeds)

            for seed in row_seeds:
                out_dir = runs_dir / f"{exp_id}_seed{seed}"
                job = Job(exp_id=exp_id, seed=seed, cmd=[], out_dir=out_dir, stage=which)
                if job.done and not force:
                    skipped += 1
                    continue
                job.cmd = _build_cmd(row, exp_id, seed, out_dir, pretrained,
                                     data_root, manifest, extra)
                jobs.append(job)

    LOGGER.info("stage=%s：%d 个任务待跑，%d 个已完成跳过", stage, len(jobs), skipped)
    return jobs


def _stage_of(exp_id: str) -> str:
    for name, prefixes in STAGES.items():
        if any(exp_id.startswith(p) for p in prefixes):
            return name
    return "other"


def _parse_seeds(raw: Optional[str], default: Sequence[int]) -> List[int]:
    """解析矩阵里的 seeds 列。

    支持 ``0;1;2`` / ``0,1,2`` / ``0``（单个 seed 值）。
    留空则用命令行的 ``--seeds``。

    Note:
        单个数字一律当作**seed 值**，不是"seed 个数"。
        早前把它当个数，结果 ``seeds=0`` 被解释成"0 个 seed"，
        整个 sanity 阶段静默变成 0 个任务 —— 而 sanity 恰恰是最该跑的关卡。
    """
    if not raw or not raw.strip():
        return list(default)
    raw = raw.strip()
    for sep in (";", ",", "|"):
        if sep in raw:
            return [int(x) for x in raw.split(sep) if x.strip().lstrip("-").isdigit()]
    if raw.isdigit():
        return [int(raw)]
    LOGGER.warning("seeds 列 %r 解析不了，退回默认 %s", raw, list(default))
    return list(default)


def _build_cmd(
    row: Dict[str, str],
    exp_id: str,
    seed: int,
    out_dir: Path,
    pretrained: Optional[str],
    data_root: Path,
    manifest: Path,
    extra: Sequence[str],
) -> List[str]:
    """把矩阵的一行翻成一条 CLI 命令。

    矩阵里的 ``extra_args`` 列直接透传，所以新增消融只要在 CSV 里加一行、
    在 extra_args 里写上开关，不需要改这个脚本。
    """
    get = lambda k, d="": (row.get(k) or d).strip()  # noqa: E731

    entry = get("entrypoint") or "fedosp.run_fed"
    cmd = [sys.executable, "-m", entry,
           "--exp-id", exp_id,
           "--seed", str(seed),
           "--out", str(out_dir),
           "--manifest", str(manifest)]

    if pretrained:
        cmd += ["--pretrained", pretrained]

    mapping = {
        "strategy": "--strategy",
        "backbone": "--backbone",
        "clients": "--clients",
        "rounds": "--rounds",
        "label_budget": "--label-budget",
        "img_size": "--img-size",
        "batch_size": "--batch-size",
        "lr": "--lr",
        "participation": "--participation",
        "mode": "--mode",
        "epochs": "--epochs",
    }
    for col, flag in mapping.items():
        v = get(col)
        if not v or v.lower() in ("-", "na", "n/a", "default"):
            continue
        if col == "clients":
            cmd += [flag] + v.replace(";", " ").replace(",", " ").split()
        else:
            cmd += [flag, v]

    if get("extra_args"):
        cmd += shlex.split(get("extra_args"))
    cmd += list(extra)
    return cmd


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=ROOT / "configs" / "experiment_matrix.csv")
    ap.add_argument("--stage", default="all",
                    help="all 或 " + " / ".join(STAGES))
    ap.add_argument("--gpus", default="0", help="逗号分隔，如 0,1,2,3")
    ap.add_argument("--jobs-per-gpu", type=int, default=1,
                    help="单卡并发任务数。80GB 卡可设 2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    ap.add_argument("--pretrained", default=os.environ.get("RETFOUND_CKPT") or None)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--timeout-h", type=float, default=12.0)
    ap.add_argument("--force", action="store_true", help="重跑已完成的配置")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="透传给每个实验的额外参数，放在最后")
    args = ap.parse_args()

    if not args.pretrained:
        LOGGER.warning(
            "没有指定 --pretrained（也没设 RETFOUND_CKPT 环境变量）："
            "骨干会是随机初始化的，结果没有意义。正式跑之前请先拿到 RETFound 权重。"
        )

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    total_failed = 0
    for st in stages:
        jobs = build_jobs(
            args.matrix, st, args.runs_dir, seeds, args.pretrained,
            args.data_root, args.manifest, args.force, args.extra,
        )
        if not jobs:
            continue

        LOGGER.info("─" * 68)
        LOGGER.info("阶段 %s", st)
        sched = Scheduler(
            gpus=[g.strip() for g in args.gpus.split(",") if g.strip()],
            jobs_per_gpu=args.jobs_per_gpu, retries=args.retries,
            log_dir=args.log_dir, dry_run=args.dry_run, timeout_h=args.timeout_h,
        )
        sched.submit(jobs)
        rc = sched.run()
        total_failed += len(sched.failed)

        # sanity 阶段是关卡：锚点没对上就别往下跑，省几十个 GPU 小时
        if st == "sanity" and rc != 0 and not args.dry_run:
            LOGGER.error("sanity 阶段失败 —— 这一关是用来拦住数据管线问题的，先修好再继续。")
            LOGGER.error("常见原因：manifest 划分有误、预处理没跑完、RETFound 权重没加载上。")
            return 1

    if total_failed:
        LOGGER.error("总共 %d 个任务失败，详见 %s/failed.txt", total_failed, args.log_dir)
        return 1
    LOGGER.info("全部任务完成 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
