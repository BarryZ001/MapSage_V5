#!/bin/bash
# 8卡分布式训练启动脚本 - 交互式版本（输出到屏幕）
# 燧原T20 GCU版本

set -e

echo "🚀 启动8卡分布式训练（交互式）"
echo "📅 时间: $(date)"

# 设置环境变量
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 设置GCU相关环境变量
export CUDA_VISIBLE_DEVICES=""  # 禁用CUDA
export GCU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 配置文件路径
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
SCRIPT_FILE="scripts/train_distributed_8card_gcu.py"

# 检查文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ 训练脚本不存在: $SCRIPT_FILE"
    exit 1
fi

echo "📄 配置文件: $CONFIG_FILE"
echo "📜 训练脚本: $SCRIPT_FILE"
echo "🔢 使用设备数: $WORLD_SIZE"

# 创建工作目录
WORK_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu_8card"
mkdir -p "$WORK_DIR"

echo ""
echo "🔥 使用torchrun启动8卡分布式训练..."
echo "📺 输出将直接显示在屏幕上"
echo ""

# 使用torchrun启动分布式训练（前台运行，输出到屏幕）
python3 -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    "$SCRIPT_FILE" "$CONFIG_FILE" \
    --launcher pytorch

echo ""
echo "🎉 8卡分布式训练完成!"