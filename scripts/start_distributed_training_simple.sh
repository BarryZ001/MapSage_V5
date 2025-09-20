#!/bin/bash
# scripts/start_distributed_training_simple.sh - 使用torchrun启动分布式训练

set -e

# 配置参数
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu.py"
NPROC_PER_NODE=8  # 8个GCU
MASTER_PORT=29500

echo "🚀 启动DINOv3分布式训练 - 使用${NPROC_PER_NODE}个GCU"
echo "📝 配置文件: ${CONFIG_FILE}"

# 设置环境变量
export TORCH_DEVICE=xla
export XLA_USE_BF16=1
export MMENGINE_DEVICE=xla
export CUDA_VISIBLE_DEVICES=""

# 使用torchrun启动分布式训练
python3 -m torch.distributed.run \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    scripts/train_distributed.py ${CONFIG_FILE} \
    --launcher pytorch

echo "🎉 分布式训练已启动"