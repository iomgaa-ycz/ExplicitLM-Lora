#!/usr/bin/env bash
# Phase 3 MedQA SFT 一键启动脚本
#
# 用法：
#   默认双卡（GPU 6,7），从 phase2_best 加载：
#       bash scripts/run_phase3_sft.sh
#   指定 Phase 2 权重：
#       FROM_PHASE2=checkpoints/phase2_best bash scripts/run_phase3_sft.sh
#   指定 GPU 与 Phase 2 权重：
#       NUM_GPUS=2 GPU_IDS=3,5 FROM_PHASE2=checkpoints/phase2_best bash scripts/run_phase3_sft.sh
#   覆盖配置参数：
#       bash scripts/run_phase3_sft.sh --override train.phase3_max_epochs=1
#
# 环境变量：
#   NUM_GPUS        使用 GPU 数量（默认 2）
#   GPU_IDS         CUDA_VISIBLE_DEVICES（默认 6,7）
#   FROM_PHASE2     Phase 2 最优 checkpoint 目录（默认 checkpoints/phase2_best）
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
FROM_PHASE2="${FROM_PHASE2:-${PROJECT_ROOT}/checkpoints/phase2_best}"

echo "[Phase3SFT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_processes=${NUM_GPUS}"
echo "[Phase3SFT] Config: ${CONFIG}"
echo "[Phase3SFT] Phase 2 Checkpoint: ${FROM_PHASE2}"
echo "[Phase3SFT] Project: ${PROJECT_ROOT}"

# ── 检查 Phase 2 Checkpoint ──
if [ ! -d "${FROM_PHASE2}" ]; then
    echo "[WARN] Phase 2 checkpoint 目录不存在: ${FROM_PHASE2}"
    echo "[INFO] 将使用随机初始化注入模块（建议先完成 Phase 2 训练）"
    echo "[INFO] 如已有 Phase 2 权重，请设置 FROM_PHASE2=<路径>"
fi

# ── 检查 MedQA 数据 ──
MEDQA_DIR="${PROJECT_ROOT}/data/medqa/hf_dataset"
if [ ! -d "${MEDQA_DIR}" ]; then
    echo "[ERROR] MedQA HF dataset 目录不存在: ${MEDQA_DIR}"
    echo "[INFO]  请将 MedQA HuggingFace dataset 放入 data/medqa/hf_dataset/"
    exit 1
fi
echo "[Phase3SFT] MedQA dataset: ${MEDQA_DIR}"

TRAIN_KM="${PROJECT_ROOT}/data/medqa_knowledge_train.jsonl"
if [ ! -f "${TRAIN_KM}" ]; then
    echo "[WARN] 训练知识映射不存在: ${TRAIN_KM}（将使用空知识兜底）"
fi

VAL_KM="${PROJECT_ROOT}/data/medqa_knowledge_validation.jsonl"
if [ ! -f "${VAL_KM}" ]; then
    echo "[WARN] 验证知识映射不存在: ${VAL_KM}（将使用空知识兜底）"
fi

# ── 检查检查点目录 ──
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
echo "[Phase3SFT] Checkpoint 目录: ${CHECKPOINT_DIR}"

# ── 激活 Conda 环境 ──
eval "$(conda shell.bash hook)"
conda activate ExplicitLLM

# ── 检查依赖 ──
python -c "import accelerate; import swanlab; import datasets" 2>/dev/null || {
    echo "[INFO] 安装缺失依赖..."
    pip install accelerate swanlab datasets -q
}

# ── Accelerate 启动 ──
echo "[Phase3SFT] 启动训练..."
accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port 29502 \
    "${PROJECT_ROOT}/main.py" \
    --config "${CONFIG}" \
    --device cuda \
    train --phase 3 --from-phase2 "${FROM_PHASE2}" \
    "$@"
