#!/bin/bash
# 单进程GCU训练脚本（分布式失败时的回退方案）

set -e

echo "🚀 启动单进程GCU训练"

# 设置环境变量
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 单进程模式
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# GCU环境变量
export ECCL_BACKEND=gloo
export ECCL_DEVICE_TYPE=GCU
export ECCL_DEBUG=0
export CUDA_VISIBLE_DEVICES=""
export TOPS_VISIBLE_DEVICES=0

# 训练参数
CONFIG_FILE="configs/dinov3/dinov3_vit-l16_mmrs1m_t20_gcu.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu_single"
TRAIN_SCRIPT="scripts/train_distributed_pytorch_ddp_8card_gcu.py"

# 创建工作目录
mkdir -p "$WORK_DIR"

echo "📋 单进程训练配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  工作目录: $WORK_DIR"
echo "  训练脚本: $TRAIN_SCRIPT"
echo "  模式: 单进程"

# 启动单进程训练
echo "🚀 启动单进程GCU训练..."

python3 "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --launcher none

echo "✅ 单进程训练启动完成"
