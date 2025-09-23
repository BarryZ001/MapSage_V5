#!/bin/bash
# gloo后端分布式训练启动脚本

echo "🚀 启动gloo后端分布式训练..."

# 设置环境变量
source scripts/setup_eccl_env.sh

# 检查参数
SCRIPT_PATH=${1:-"scripts/train_distributed_gcu_fixed.py"}
NUM_GPUS=${2:-8}

echo "📋 训练参数:"
echo "  - 训练脚本: $SCRIPT_PATH"
echo "  - GPU数量: $NUM_GPUS"

# 使用torchrun启动分布式训练（强制使用gloo后端）
export TORCH_DISTRIBUTED_BACKEND=gloo

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    $SCRIPT_PATH \
    --backend=gloo \
    --launcher=pytorch

echo "🎯 分布式训练完成！"
