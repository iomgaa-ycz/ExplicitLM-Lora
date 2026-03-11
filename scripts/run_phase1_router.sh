#!/usr/bin/env bash
# Phase 1 Router 训练一键启动脚本
#
# 用法：
#   单卡（默认 GPU 6）：bash scripts/run_phase1_router.sh
#   双卡：NUM_GPUS=2 GPU_IDS=6,7 bash scripts/run_phase1_router.sh
#   覆盖配置：bash scripts/run_phase1_router.sh --override train.phase1_max_epochs=5
#
# 环境变量：
#   NUM_GPUS        使用 GPU 数量（默认 1）
#   GPU_IDS         CUDA_VISIBLE_DEVICES（默认 6）
#   CONFIG          配置文件路径（默认 config/default.yaml）

set -euo pipefail

# ── GPU 配置 ──
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-6}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# ── 路径 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"

echo "[Phase1Router] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_processes=${NUM_GPUS}"
echo "[Phase1Router] Config: ${CONFIG}"
echo "[Phase1Router] Project: ${PROJECT_ROOT}"

# ── 检查预压缩数据目录 ──
PARQUET_DIR="${PROJECT_ROOT}/data/compressed/v2"
if [ ! -d "${PARQUET_DIR}" ]; then
    echo "[ERROR] 预压缩 FineWeb-Edu 数据目录不存在: ${PARQUET_DIR}"
    echo "[INFO]  请先运行数据预处理脚本或将数据放入 data/compressed/v2/"
    echo "[INFO]  也可通过 --override data.phase1_parquet_dir=<path> 指定其他路径"
    exit 1
fi

PARQUET_COUNT=$(find "${PARQUET_DIR}" -name "*.parquet" | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    echo "[ERROR] ${PARQUET_DIR} 中没有找到 .parquet 文件"
    exit 1
fi
echo "[Phase1Router] 找到 ${PARQUET_COUNT} 个 Parquet 文件于 ${PARQUET_DIR}"

# ── 检查检查点目录 ──
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
echo "[Phase1Router] Checkpoint 目录: ${CHECKPOINT_DIR}"

# ── 激活 Conda 环境 ──
eval "$(conda shell.bash hook)"
conda activate ExplicitLLM

# ── 检查依赖 ──
python -c "import accelerate; import swanlab" 2>/dev/null || {
    echo "[INFO] 安装缺失依赖..."
    pip install accelerate swanlab -q
}

# ── Accelerate 启动 ──
echo "[Phase1Router] 启动训练..."
accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port 29500 \
    "${PROJECT_ROOT}/main.py" \
    --config "${CONFIG}" \
    --device cuda \
    train --phase 1 \
    "$@"
