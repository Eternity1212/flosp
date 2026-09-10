#!/usr/bin/env bash
# ============================================================================
# 数据获取助手：能自动下的自动下，需要人工申请的给出准确指引并等你确认。
#
#   bash scripts/download_data.sh --all         # 下全部能自动下的（Kaggle 三项）
#   bash scripts/download_data.sh --eyepacs     # 只下 EyePACS
#   bash scripts/download_data.sh --retfound    # 下 RETFound 权重（需先获批）
#   bash scripts/download_data.sh --verify      # 只校验已有数据的完整性
#
# 完整说明与人工申请步骤见 DATA.md。
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${FEDOSP_DATA:-$REPO_ROOT/data}"
RAW="$DATA_ROOT/raw"
WEIGHTS="$REPO_ROOT/weights"

C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'
C_BLU=$'\033[0;34m'; C_BLD=$'\033[1m'; C_OFF=$'\033[0m'
log()  { printf '%s[%s]%s %s\n' "$C_BLU" "$(date +%H:%M:%S)" "$C_OFF" "$*"; }
ok()   { printf '%s  ✓ %s%s\n' "$C_GRN" "$*" "$C_OFF"; }
warn() { printf '%s  ! %s%s\n' "$C_YEL" "$*" "$C_OFF"; }
die()  { printf '%s  ✗ %s%s\n' "$C_RED" "$*" "$C_OFF" >&2; exit 1; }
hdr()  { printf '\n%s%s%s\n%s\n' "$C_BLD" "$*" "$C_OFF" \
         "────────────────────────────────────────────────────────────────────"; }

DO_EYEPACS=0 DO_APTOS=0 DO_DDR=0 DO_IDRID=0 DO_MESSIDOR=0 DO_RETFOUND=0 DO_VERIFY=0
[[ $# -eq 0 ]] && { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)      DO_EYEPACS=1; DO_APTOS=1; DO_MESSIDOR=1; DO_VERIFY=1; shift ;;
    --eyepacs)  DO_EYEPACS=1; shift ;;
    --aptos)    DO_APTOS=1; shift ;;
    --ddr)      DO_DDR=1; shift ;;
    --idrid)    DO_IDRID=1; shift ;;
    --messidor) DO_MESSIDOR=1; shift ;;
    --retfound) DO_RETFOUND=1; shift ;;
    --verify)   DO_VERIFY=1; shift ;;
    -h|--help)  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)          die "不认识的参数 $1" ;;
  esac
done

mkdir -p "$RAW" "$WEIGHTS"

check_kaggle() {
  command -v kaggle &>/dev/null || die "没装 kaggle CLI：pip install kaggle"
  [[ -f "$HOME/.kaggle/kaggle.json" ]] || die \
    "缺 ~/.kaggle/kaggle.json。去 kaggle.com/settings/account 建 API token，见 DATA.md 第 1 节"
  local perm
  perm=$(stat -f "%OLp" "$HOME/.kaggle/kaggle.json" 2>/dev/null \
         || stat -c "%a" "$HOME/.kaggle/kaggle.json" 2>/dev/null)
  [[ "$perm" != "600" ]] && { warn "kaggle.json 权限是 $perm，改成 600"; chmod 600 "$HOME/.kaggle/kaggle.json"; }
}

count_images() {
  find "$1" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
       -o -iname '*.tif' -o -iname '*.tiff' \) 2>/dev/null | wc -l | tr -d ' '
}

# ============================================================================
if [[ $DO_EYEPACS -eq 1 ]]; then
  hdr "EyePACS（Kaggle DR 2015，约 35 GB）"
  check_kaggle
  mkdir -p "$RAW/eyepacs"; cd "$RAW/eyepacs"
  if [[ -d train && $(count_images train) -gt 30000 ]]; then
    ok "已存在，跳过"
  else
    warn "35 GB，视网速可能要几小时。若报 403 → 先去网页接受竞赛规则（DATA.md 第 1 节）"
    kaggle competitions download -c diabetic-retinopathy-detection || die "下载失败"
    log "合并分卷并解压（图像分成 5 个卷）"
    if ls train.zip.00* &>/dev/null; then
      cat train.zip.00* > train.zip && unzip -q -o train.zip && rm -f train.zip train.zip.00*
    fi
    if ls test.zip.00* &>/dev/null; then
      cat test.zip.00* > test.zip && unzip -q -o test.zip && rm -f test.zip test.zip.00*
    fi
    for z in trainLabels.csv.zip retinopathy_solution.csv.zip; do
      [[ -f "$z" ]] && unzip -q -o "$z"
    done
    ok "EyePACS 就绪"
  fi
fi

if [[ $DO_APTOS -eq 1 ]]; then
  hdr "APTOS 2019（约 10 GB）"
  check_kaggle
  mkdir -p "$RAW/aptos"; cd "$RAW/aptos"
  if [[ -d train_images && $(count_images train_images) -gt 3000 ]]; then
    ok "已存在，跳过"
  else
    kaggle competitions download -c aptos2019-blindness-detection || die "下载失败"
    unzip -q -o aptos2019-blindness-detection.zip && rm -f aptos2019-blindness-detection.zip
    ok "APTOS 就绪"
  fi
fi

if [[ $DO_MESSIDOR -eq 1 ]]; then
  hdr "Messidor-2 的 DR 裁定标签（图像要另外向 ADCIS 申请）"
  check_kaggle
  mkdir -p "$RAW/messidor2"; cd "$RAW/messidor2"
  if [[ -f messidor_data.csv ]]; then
    ok "标签已存在"
  else
    kaggle datasets download -d google-brain/messidor2-dr-grades \
      && unzip -q -o messidor2-dr-grades.zip && rm -f messidor2-dr-grades.zip \
      && ok "标签就绪" || warn "标签下载失败，手动下：kaggle.com/datasets/google-brain/messidor2-dr-grades"
  fi
  if [[ ! -d IMAGES ]] || [[ $(count_images IMAGES) -lt 1000 ]]; then
    printf '\n%s图像需要人工申请，无法自动下载：%s\n' "$C_BLD" "$C_OFF"
    echo "  1. 打开 https://www.adcis.net/en/third-party/messidor2/"
    echo "  2. 填个人信息表单并验证邮箱"
    echo "  3. 等审批（数天到数周），拿到链接后解压到 $RAW/messidor2/IMAGES/"
    echo
    warn "这一项有等待期，请**今天**就去填表 —— 拖一天整个项目晚一天"
    warn "万一批不下来，用 leave-one-client-out 协议替代（见 DATA.md 第 6c 节）"
  else
    ok "图像已就绪（$(count_images IMAGES) 张）"
  fi
fi

if [[ $DO_DDR -eq 1 ]]; then
  hdr "DDR"
  printf '%sDDR 放在 Google Drive / 百度网盘，没有直链，需手动下载：%s\n' "$C_BLD" "$C_OFF"
  echo "  1. 打开 https://github.com/nkicsl/DDR-dataset"
  echo "  2. 按 README 的网盘链接下载（可能是分卷 zip）"
  echo "  3. 放到 $RAW/ddr/ 并解压："
  echo "       cat DDR-dataset.zip.0* > DDR-dataset.zip && unzip -q DDR-dataset.zip"
  echo
  warn "记住 DDR 的标签 5 = ungradable，不是第 6 个严重度等级。build_manifest.py 会自动剔除。"
fi

if [[ $DO_IDRID -eq 1 ]]; then
  hdr "IDRiD"
  printf '%sIDRiD 需要 IEEE DataPort 账号（免费），需手动下载：%s\n' "$C_BLD" "$C_OFF"
  echo "  1. 打开 https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"
  echo "  2. 注册登录后只下 'B. Disease Grading' 部分"
  echo "  3. 解压到 $RAW/idrid/"
  echo
  warn "必须用官方 train/test 划分，别自己重新随机划分，否则跟文献锚点对不上。"
fi

if [[ $DO_RETFOUND -eq 1 ]]; then
  hdr "RETFound 预训练权重（约 1.2 GB）"
  command -v huggingface-cli &>/dev/null || die "没装：pip install huggingface_hub"
  if [[ -f "$WEIGHTS/RETFound_mae_natureCFP.pth" ]]; then
    ok "已存在：$WEIGHTS/RETFound_mae_natureCFP.pth"
  else
    warn "这是 gated model：必须先在网页上获批，否则下面会报 401/403"
    echo "  1. https://huggingface.co/YukunZhou/RETFound_mae_natureCFP → Agree and access"
    echo "  2. https://huggingface.co/settings/tokens → 建 read token → huggingface-cli login"
    echo
    if huggingface-cli download YukunZhou/RETFound_mae_natureCFP \
         --local-dir "$WEIGHTS" --local-dir-use-symlinks False; then
      ok "权重就绪"
      echo
      printf '%s把这行加进 ~/.bashrc：%s\n' "$C_BLD" "$C_OFF"
      echo "  export RETFOUND_CKPT=$WEIGHTS/RETFound_mae_natureCFP.pth"
    else
      warn "下载失败。若是 401/403 说明申请还没批下来，等审批邮件。"
      warn "等待期间可用 --imagenet-pretrained 先把管线跑通。"
    fi
  fi
fi

# ============================================================================
if [[ $DO_VERIFY -eq 1 ]]; then
  hdr "完整性校验"
  fail=0
  # 名称 期望图像数 目录 关键文件
  check_one() {
    local name="$1" expect="$2" dir="$3" keyfile="${4:-}"
    if [[ ! -d "$dir" ]]; then
      warn "$name：目录不存在（$dir）"; fail=1; return
    fi
    local n; n=$(count_images "$dir")
    if [[ -n "$keyfile" && ! -f "$keyfile" ]]; then
      warn "$name：缺关键文件 $(basename "$keyfile")"; fail=1
    fi
    if [[ "$n" -ge "$expect" ]]; then
      ok "$name：$n 张（期望 ≥ $expect）"
    else
      warn "$name：只有 $n 张，期望 ≥ $expect —— 可能没下完或解压不全"; fail=1
    fi
  }

  check_one "EyePACS train"  35000 "$RAW/eyepacs/train"      "$RAW/eyepacs/trainLabels.csv"
  check_one "EyePACS test"   53000 "$RAW/eyepacs/test"
  check_one "APTOS"           3600 "$RAW/aptos/train_images" "$RAW/aptos/train.csv"
  check_one "DDR"            12000 "$RAW/ddr"
  check_one "IDRiD"            500 "$RAW/idrid"
  check_one "Messidor-2"      1700 "$RAW/messidor2/IMAGES"   "$RAW/messidor2/messidor_data.csv"

  if [[ -f "$WEIGHTS/RETFound_mae_natureCFP.pth" ]]; then
    ok "RETFound 权重：$(du -h "$WEIGHTS/RETFound_mae_natureCFP.pth" | cut -f1)"
  else
    warn "RETFound 权重缺失 —— 骨干会随机初始化，结果没有科研意义"; fail=1
  fi

  echo
  if [[ $fail -eq 0 ]]; then
    ok "全部就绪，下一步：bash scripts/run_all.sh --stage data"
  else
    warn "有缺失项。缺 Messidor-2 或 RETFound 属正常（都要等审批），"
    warn "其余四个数据集齐了就可以先跑 --stage data 和 --stage sanity。"
  fi
fi

exit 0
