#!/usr/bin/env bash
# Phase 2 Fusion 预训练一键启动脚本
#
# 用法：
#   默认双卡（GPU 6,7）：bash scripts/run_phase2_fusion.sh
#   指定 GPU：       NUM_GPUS=2 GPU_IDS=3,5 bash scripts/run_phase2_fusion.sh
#   覆盖配置：        bash scripts/run_phase2_fusion.sh --override train.phase2_max_epochs=1
#
# 环境变量：
#   NUM_GPUS        使用 GPU 数量（默认 2）
#   GPU_IDS         CUDA_VISIBLE_DEVICES（默认 6,7）
#   CONFIG          配置文件路径（默认 config/default.yaml）

set -euo pipefail

# ── GPU 配置 ──
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-6,7}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# ── 路径 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"

echo "[Phase2Fusion] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_processes=${NUM_GPUS}"
echo "[Phase2Fusion] Config: ${CONFIG}"
echo "[Phase2Fusion] Project: ${PROJECT_ROOT}"

# ── 检查预压缩数据目录 ──
PARQUET_DIR="${PROJECT_ROOT}/data/compressed/v2"
if [ ! -d "${PARQUET_DIR}" ]; then
    echo "[ERROR] 预压缩 FineWeb-Edu 数据目录不存在: ${PARQUET_DIR}"
    echo "[INFO]  Phase 2 复用 Phase 1 数据目录（data/compressed/v2/）"
    echo "[INFO]  请先准备预压缩数据，或通过 --override data.phase1_parquet_dir=<path> 指定"
    exit 1
fi

PARQUET_COUNT=$(find "${PARQUET_DIR}" -name "*.parquet" | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    echo "[ERROR] ${PARQUET_DIR} 中没有找到 .parquet 文件"
    exit 1
fi
echo "[Phase2Fusion] 找到 ${PARQUET_COUNT} 个 Parquet 文件于 ${PARQUET_DIR}"

# ── 检查检查点目录 ──
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
echo "[Phase2Fusion] Checkpoint 目录: ${CHECKPOINT_DIR}"

# ── 激活 Conda 环境 ──
eval "$(conda shell.bash hook)"
conda activate ExplicitLLM

# ── 检查依赖 ──
python -c "import accelerate; import swanlab" 2>/dev/null || {
    echo "[INFO] 安装缺失依赖..."
    pip install accelerate swanlab -q
}

# ── Accelerate 启动 ──
echo "[Phase2Fusion] 启动训练..."
accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port 29501 \
    "${PROJECT_ROOT}/main.py" \
    --config "${CONFIG}" \
    --device cuda \
    train --phase 2 \
    "$@"
