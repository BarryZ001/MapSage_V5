#!/bin/bash
# gloo后端分布式训练启动脚本

echo "🚀 启动gloo后端分布式训练..."

# 设置环境变量
source scripts/setup_eccl_env.sh

# 检查参数
SCRIPT_PATH=${1:-"scripts/train_distributed_gcu_fixed.py"}
NUM_GPUS=${2:-8}
CONFIG_FILE=${3:-"configs/train_dinov3_mmrs1m_t20_gcu.py"}

echo "📋 训练参数:"
echo "  - 训练脚本: $SCRIPT_PATH"
echo "  - GPU数量: $NUM_GPUS"
echo "  - 配置文件: $CONFIG_FILE"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "💡 请使用以下格式："
    echo "   bash scripts/start_gloo_training.sh [训练脚本] [GPU数量] [配置文件]"
    echo "   例如: bash scripts/start_gloo_training.sh scripts/train_distributed_gcu_fixed.py 8 configs/train_dinov3_mmrs1m_t20_gcu.py"
    exit 1
fi

# 使用torchrun启动分布式训练（强制使用gloo后端）
export TORCH_DISTRIBUTED_BACKEND=gloo

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    $SCRIPT_PATH \
    $CONFIG_FILE \
    --backend=gloo \
    --launcher=pytorch

echo "🎯 分布式训练完成！"
