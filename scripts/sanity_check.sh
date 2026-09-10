#!/usr/bin/env bash
# 第 2 周的关卡：数据管线健全性检查（方案 9.3 + 第 14 节自查清单）
#
# 这一步的目的不是拿名次，是回答「我的数据管线和权重加载是不是对的」。
# 对不上文献锚点就停下来查，**不要往联邦实验走**。
#
# 用法：
#   bash scripts/sanity_check.sh /path/to/raw /path/to/cache /path/to/RETFound_mae_natureCFP.pth

set -euo pipefail

DATA_ROOT="${1:?用法: sanity_check.sh <原始数据根目录> <缓存目录> <RETFound权重路径>}"
CACHE_DIR="${2:?缺少缓存目录}"
PRETRAINED="${3:?缺少 RETFound 权重路径}"
OUT_DIR="${OUT_DIR:-runs/sanity}"
WORKERS="${WORKERS:-16}"

echo "=============================================="
echo " 步骤 0/4  代码自检（不需要数据）"
echo "=============================================="
python tests/test_pipeline.py

echo
echo "=============================================="
echo " 步骤 1/4  构建统一 manifest + 泄漏检查"
echo "=============================================="
python -m fedosp.data.build_manifest \
    --data-root "${DATA_ROOT}" \
    --out data/manifest.csv \
    --strict            # ← 任何一项校验不过直接退出

echo
echo "=============================================="
echo " 步骤 2/4  预处理与缓存（约 2-4 小时）"
echo "=============================================="
python -m fedosp.data.preprocess \
    --manifest data/manifest.csv \
    --cache-dir "${CACHE_DIR}" \
    --workers "${WORKERS}"

echo
echo "=============================================="
echo " 步骤 3/4  再校验一次缓存后的 manifest"
echo "=============================================="
python -m fedosp.data.build_manifest \
    --data-root "${DATA_ROOT}" \
    --out data/manifest_cached.csv \
    --verify-only --strict
md5sum data/manifest_cached.csv 2>/dev/null || md5 data/manifest_cached.csv
echo ">>> 把上面这个 md5 记进实验日志：所有方法必须读同一份 manifest"

echo
echo "=============================================="
echo " 步骤 4/4  集中式基线 vs 文献锚点"
echo "   期望：APTOS AUROC≈0.943 / IDRiD≈0.822 / Messidor-2≈0.884"
echo "=============================================="
python -m fedosp.run_central \
    --manifest data/manifest_cached.csv \
    --mode local --clients aptos idrid ddr \
    --pretrained "${PRETRAINED}" \
    --epochs 50 --patience 10 --amp \
    --check-anchors --anchor-tol 0.05 \
    --out "${OUT_DIR}"

echo
echo "结果在 ${OUT_DIR}/result.json 的 anchor_check 字段。"
echo "全 OK  -> 放行，进入第 3 周的联邦基线"
echo "有 OFF -> 按这个顺序查：1) 划分是否泄漏 2) 标签列是否读错 3) 预处理是否把视野裁坏"
