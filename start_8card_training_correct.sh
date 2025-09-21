#!/bin/bash
# 正确的8卡分布式训练启动脚本 - 燧原T20 GCU版本
# 确保所有8张GCU卡都参与训练

set -e

echo "🚀 启动MapSage V5 - 8卡分布式训练"
echo "📅 时间: $(date)"
echo "🖥️  主机: $(hostname)"

# 检查GCU设备
echo "🔍 检查GCU设备状态..."
efsmi

# 配置环境变量
export WORLD_SIZE=8
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

# 配置GCU相关环境变量
export ECCL_BACKEND=eccl
export ECCL_DEVICE_TYPE=gcu
export ECCL_DEBUG=1

# 配置训练参数
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu_8card_correct"

echo "📄 配置文件: $CONFIG_FILE"
echo "📁 工作目录: $WORK_DIR"

# 创建工作目录
mkdir -p "$WORK_DIR"

# 停止之前的训练进程
echo "🛑 停止之前的训练进程..."
pkill -f "train_distributed_8card_gcu.py" || true
sleep 2

# 使用torchrun启动8卡分布式训练
echo "🚀 使用torchrun启动8卡分布式训练..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    scripts/train_distributed_8card_gcu.py \
    "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch

echo "✅ 8卡分布式训练启动完成"
echo "📊 请使用 'efsmi' 命令监控GCU设备使用情况"
echo "📝 训练日志保存在: $WORK_DIR/logs/"