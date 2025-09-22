#!/bin/bash

# T20 8卡分布式训练启动脚本 - 修复版
# 使用torch_gcu分布式支持

echo "🚀 启动T20 8卡分布式训练 (torch_gcu修复版)"

# 设置环境变量
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=8
export NPROC_PER_NODE=8

# 设置GCU相关环境变量
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=""

# 设置Python路径
export PYTHONPATH=$PWD:$PYTHONPATH

echo "📋 环境变量设置:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  TOPS_VISIBLE_DEVICES: $TOPS_VISIBLE_DEVICES"

# 检查配置文件是否存在
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu.py"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "📝 使用配置文件: $CONFIG_FILE"
echo "🔧 使用修复版训练脚本: scripts/train_distributed_gcu_fixed.py"

# 启动分布式训练
python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_distributed_gcu_fixed.py \
    $CONFIG_FILE \
    --launcher pytorch

echo "✅ 训练启动完成"