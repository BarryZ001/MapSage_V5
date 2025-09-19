#!/bin/bash

# DINOv3 + LoveDA T20 GCU训练脚本
# 在T20服务器上运行LoveDA数据集的微调训练

set -e  # 遇到错误立即退出

echo "🚀 开始DINOv3 + LoveDA T20 GCU训练"
echo "⏰ 训练开始时间: $(date)"

# 配置参数
CONFIG_FILE="configs/train_dinov3_loveda_t20_gcu.py"
WORK_DIR="./work_dirs/dinov3_loveda_t20_gcu"
NUM_GPUS=4  # T20服务器GPU数量
PORT=29500

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "📋 使用配置文件: $CONFIG_FILE"
echo "💾 工作目录: $WORK_DIR"
echo "🔥 GPU数量: $NUM_GPUS"

# 创建工作目录
mkdir -p $WORK_DIR

# 检查数据集
echo "🔍 检查数据集..."
if [ ! -d "/workspace/data/loveda" ]; then
    echo "❌ LoveDA数据集不存在: /workspace/data/loveda"
    exit 1
fi

# 检查权重文件
echo "🔍 检查权重文件..."
if [ ! -f "/workspace/weights/best_mIoU_iter_6000.pth" ]; then
    echo "❌ MMRS1M训练权重不存在: /workspace/weights/best_mIoU_iter_6000.pth"
    echo "💡 请先完成MMRS1M数据集的训练"
    exit 1
fi

# 环境验证
echo "🔍 运行环境验证..."
python scripts/validate_t20_environment.py

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动训练
echo "🚀 启动分布式训练..."
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG_FILE \
    --work-dir $WORK_DIR \
    --launcher pytorch \
    --seed 42 \
    --deterministic

echo "✅ 训练完成！"
echo "📊 训练日志: $WORK_DIR/"
echo "💾 模型权重: $WORK_DIR/latest.pth"
echo "📈 TensorBoard日志: $WORK_DIR/tf_logs/"
echo "⏰ 训练结束时间: $(date)"

# 显示最佳模型信息
if [ -f "$WORK_DIR/best_mIoU.pth" ]; then
    echo "🏆 最佳模型: $WORK_DIR/best_mIoU.pth"
fi

echo "🎉 DINOv3 + LoveDA T20 GCU训练完成！"