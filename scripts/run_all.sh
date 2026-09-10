#!/usr/bin/env bash
# ============================================================================
# FedOSP 一键复现：从原始数据到论文表格与插图
#
# 用法：
#   bash scripts/run_all.sh                              # 全流程，自动探测 GPU
#   bash scripts/run_all.sh --gpus 0,1,2,3               # 指定 GPU
#   bash scripts/run_all.sh --stage main                 # 只跑主对比
#   bash scripts/run_all.sh --dry-run                    # 只打印命令，不真跑
#   bash scripts/run_all.sh --smoke                      # 5 分钟合成数据冒烟测试
#
# 阶段（--stage）：
#   check     环境与数据自检
#   data      建 manifest + 预处理（约 2-4 小时，只需跑一次）
#   sanity    文献锚点核对 —— **这是关卡，不过就别往下跑**
#   main      11 个基线 + FedOSP 主结果
#   ablation  组件消融 A1-A10
#   label     标签效率曲线
#   robust    参与率压力测试
#   backbone  骨干替换 A11/A12
#   analyze   汇总表格 T1-T6 + 论文插图 F2-F7
#   all       以上全部（默认）
#
# 断点续跑：已有 result.json 的配置自动跳过，中断后重跑同一条命令即可。
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------- 默认参数 ------------------------------- #
STAGE="all"
GPUS=""
JOBS_PER_GPU=1
SEEDS="0,1,2"
RETRIES=1
DRY_RUN=""
FORCE=""
SMOKE=0
DATA_ROOT="${FEDOSP_DATA:-$REPO_ROOT/data}"
MANIFEST="$DATA_ROOT/manifest.csv"
CACHE_DIR="$DATA_ROOT/cache"
PRETRAINED="${RETFOUND_CKPT:-}"
RUNS_DIR="$REPO_ROOT/runs"
LOG_DIR="$REPO_ROOT/logs"
PY="${PYTHON:-python3}"

C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'
C_BLU=$'\033[0;34m'; C_BLD=$'\033[1m'; C_OFF=$'\033[0m'

log()  { printf '%s[%s]%s %s\n' "$C_BLU" "$(date +%H:%M:%S)" "$C_OFF" "$*"; }
ok()   { printf '%s  ✓ %s%s\n' "$C_GRN" "$*" "$C_OFF"; }
warn() { printf '%s  ! %s%s\n' "$C_YEL" "$*" "$C_OFF"; }
die()  { printf '%s  ✗ %s%s\n' "$C_RED" "$*" "$C_OFF" >&2; exit 1; }
hdr()  { printf '\n%s%s%s\n%s\n' "$C_BLD" "$*" "$C_OFF" \
         "────────────────────────────────────────────────────────────────────"; }

# ------------------------------- 解析参数 ------------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)        STAGE="$2"; shift 2 ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --seeds)        SEEDS="$2"; shift 2 ;;
    --retries)      RETRIES="$2"; shift 2 ;;
    --pretrained)   PRETRAINED="$2"; shift 2 ;;
    --data-root)    DATA_ROOT="$2"; MANIFEST="$2/manifest.csv"; CACHE_DIR="$2/cache"; shift 2 ;;
    --manifest)     MANIFEST="$2"; shift 2 ;;
    --runs-dir)     RUNS_DIR="$2"; shift 2 ;;
    --dry-run)      DRY_RUN="--dry-run"; shift ;;
    --force)        FORCE="--force"; shift ;;
    --smoke)        SMOKE=1; shift ;;
    -h|--help)      sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "不认识的参数 $1（用 --help 看说明）" ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

# ============================================================================
# 冒烟测试：不需要任何真实数据，5 分钟内确认代码是通的
# ============================================================================
if [[ $SMOKE -eq 1 ]]; then
  hdr "冒烟测试（合成数据 + 微型 ViT，不需要真实数据集）"
  log "1/3 单元测试"
  $PY -m pytest tests/ -q 2>/dev/null || $PY tests/test_pipeline.py || die "单元测试失败"
  ok "单元测试通过"

  log "2/3 联邦流程能否学到东西（40 轮合成数据）"
  $PY -m fedosp.run_fed --dry-run --rounds 40 --lr 3e-3 \
      --out "$RUNS_DIR/smoke_fedosp" 2>&1 | tail -5 || die "联邦流程失败"
  ok "联邦流程通过"

  log "3/3 全部 8 个策略各跑 3 轮"
  for s in fedavg fedprox fedbn fedper scaffold fedproto feduaa fedosp; do
    if $PY -m fedosp.run_fed --dry-run --rounds 3 --strategy "$s" \
         --out "$RUNS_DIR/smoke_$s" >"$LOG_DIR/smoke_$s.log" 2>&1; then
      ok "策略 $s"
    else
      die "策略 $s 失败，见 $LOG_DIR/smoke_$s.log"
    fi
  done
  hdr "冒烟测试全部通过 ✓"
  echo "下一步：准备真实数据（见 DATA.md），然后 bash scripts/run_all.sh --stage all"
  exit 0
fi

# ============================================================================
# GPU 探测
# ============================================================================
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    N_GPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$N_GPU" -gt 0 ]]; then
      GPUS=$(seq -s, 0 $((N_GPU - 1)))
      log "自动探测到 $N_GPU 张 GPU：$GPUS"
    fi
  fi
  [[ -z "$GPUS" ]] && { GPUS="0"; warn "没探测到 GPU，按单卡跑（CPU 上会非常慢）"; }
fi

# ============================================================================
# stage: check —— 环境与数据自检
# ============================================================================
stage_check() {
  hdr "阶段 check：环境与数据自检"
  local fail=0

  $PY - <<'EOF' || fail=1
import importlib, sys
need = {"torch": "2.0", "timm": "0.9", "numpy": "1.21", "pandas": "1.3",
        "sklearn": None, "PIL": None, "matplotlib": None, "scipy": None, "yaml": None}
missing = []
for mod in need:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  ✓ {mod:12s} {v}")
    except ImportError:
        missing.append(mod); print(f"  ✗ {mod:12s} 未安装")
if missing:
    print(f"\n缺少依赖 {missing}，先跑：pip install -r requirements.txt")
    sys.exit(1)
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  ✓ GPU{i}: {p.name} {p.total_memory/1024**3:.0f} GB")
    else:
        print("  ! CUDA 不可用，只能 CPU 跑（仅适合 --smoke）")
except Exception as e:
    print(f"  ! GPU 检查异常：{e}")
EOF
  [[ $fail -eq 1 ]] && die "依赖检查未通过"
  ok "Python 依赖齐全"

  if [[ -n "$PRETRAINED" && -f "$PRETRAINED" ]]; then
    ok "RETFound 权重：$PRETRAINED ($(du -h "$PRETRAINED" | cut -f1))"
  else
    warn "没有 RETFound 权重。骨干会随机初始化，结果没有科研意义。"
    warn "获取方式：huggingface.co/YukunZhou/RETFound_mae_natureCFP（需申请，1-3 天）"
    warn "拿到后：export RETFOUND_CKPT=/path/to/RETFound_mae_natureCFP.pth"
  fi

  log "数据集目录检查（$DATA_ROOT）"
  for d in eyepacs aptos ddr idrid messidor2; do
    if [[ -d "$DATA_ROOT/raw/$d" ]]; then
      n=$(find "$DATA_ROOT/raw/$d" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \
           -o -iname '*.png' -o -iname '*.tif' \) 2>/dev/null | head -200000 | wc -l | tr -d ' ')
      ok "$d：$n 张图"
    else
      warn "$d：目录不存在（$DATA_ROOT/raw/$d）—— 见 DATA.md"
    fi
  done
  echo
  ok "自检结束"
}

# ============================================================================
# stage: data —— manifest + 预处理
# ============================================================================
stage_data() {
  hdr "阶段 data：构建 manifest 与预处理缓存"

  if [[ -f "$MANIFEST" && -z "$FORCE" ]]; then
    ok "manifest 已存在，跳过（要重建加 --force）：$MANIFEST"
  else
    log "构建统一 manifest（患者级划分、DDR ungradable 过滤、Messidor-2 隔离）"
    $PY -m fedosp.data.build_manifest \
        --data-root "$DATA_ROOT/raw" --out "$MANIFEST" \
        2>&1 | tee "$LOG_DIR/build_manifest.log" || die "manifest 构建失败，见 $LOG_DIR/build_manifest.log"
    ok "manifest → $MANIFEST"
  fi

  if [[ -d "$CACHE_DIR" && -z "$FORCE" ]]; then
    ok "预处理缓存已存在，跳过：$CACHE_DIR"
  else
    log "预处理（圆形裁剪 + 短边 512 缩放），约 2-4 小时"
    $PY -m fedosp.data.preprocess \
        --manifest "$MANIFEST" --out "$CACHE_DIR" --workers 8 \
        2>&1 | tee "$LOG_DIR/preprocess.log" || die "预处理失败，见 $LOG_DIR/preprocess.log"
    ok "缓存 → $CACHE_DIR"
  fi
}

# ============================================================================
# 实验阶段（统一交给调度器）
# ============================================================================
run_stage() {
  local st="$1"
  hdr "阶段 $st"
  [[ -f "$MANIFEST" ]] || die "找不到 manifest（$MANIFEST），先跑 --stage data"

  local args=(--stage "$st" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU"
              --seeds "$SEEDS" --retries "$RETRIES" --manifest "$MANIFEST"
              --runs-dir "$RUNS_DIR" --log-dir "$LOG_DIR"
              --matrix "$REPO_ROOT/configs/experiment_matrix.csv")
  [[ -n "$PRETRAINED" ]] && args+=(--pretrained "$PRETRAINED")
  [[ -n "$DRY_RUN" ]]    && args+=($DRY_RUN)
  [[ -n "$FORCE" ]]      && args+=($FORCE)

  $PY scripts/scheduler.py "${args[@]}"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ "$st" == "sanity" ]]; then
      die "sanity 未通过。这一关专门拦数据管线问题，先修好再往下跑（省几十 GPU 小时）。"
    fi
    warn "阶段 $st 有失败任务，详见 $LOG_DIR/failed.txt（其余已继续）"
  fi
  return 0
}

# ============================================================================
# stage: analyze —— 出表 + 出图
# ============================================================================
stage_analyze() {
  hdr "阶段 analyze：生成论文表格与插图"

  log "汇总表格 T1-T6"
  $PY scripts/aggregate_results.py --runs "$RUNS_DIR" --out "$REPO_ROOT/tables" \
      --manifest "$MANIFEST" 2>&1 | tee "$LOG_DIR/aggregate.log" \
      || warn "汇总有问题，见 $LOG_DIR/aggregate.log"

  log "绘制插图 F2-F7"
  $PY scripts/make_figures.py --runs "$RUNS_DIR" --out "$REPO_ROOT/figures" \
      --manifest "$MANIFEST" 2>&1 | tee "$LOG_DIR/figures.log" \
      || warn "画图有问题，见 $LOG_DIR/figures.log"

  echo
  ok "表格：$REPO_ROOT/tables/（md / csv / tex 三种格式，总览见 all_tables.md）"
  ok "插图：$REPO_ROOT/figures/（pdf 矢量 + png 位图）"
  echo
  printf '%s请优先核对这三处：%s\n' "$C_BLD" "$C_OFF"
  echo "  1. tables/t3.md 里 abl_nofsr 的 ΔMacro —— 若 > -0.5 说明 FSR 收益不足，看可行性文档 5.2 的退路"
  echo "  2. tables/t3.md 里 abl_proto_sample 的 ΔMacro —— 这决定 P3 卖点是否成立"
  echo "  3. figures/F4 标题 —— 若显示 NO ordinal structure，说明 margin 损失没起作用"
}

# ============================================================================
# 主流程
# ============================================================================
START_TS=$(date +%s)
hdr "FedOSP 复现流程　stage=$STAGE　gpus=$GPUS　seeds=$SEEDS"
[[ -n "$DRY_RUN" ]] && warn "dry-run 模式：只打印命令，不实际执行"

case "$STAGE" in
  check)    stage_check ;;
  data)     stage_data ;;
  analyze)  stage_analyze ;;
  sanity|main|ablation|label|robust|backbone) run_stage "$STAGE" ;;
  all)
    stage_check
    stage_data
    for st in sanity main ablation label robust backbone; do
      run_stage "$st"
    done
    stage_analyze
    ;;
  *) die "不认识的 stage=$STAGE（用 --help 看可选值）" ;;
esac

ELAPSED=$(( $(date +%s) - START_TS ))
hdr "完成，总耗时 $((ELAPSED / 3600)) 小时 $(((ELAPSED % 3600) / 60)) 分"
[[ -f "$LOG_DIR/failed.txt" ]] && warn "有失败任务：$LOG_DIR/failed.txt（修好后重跑同一命令，已完成的会跳过）"
exit 0
