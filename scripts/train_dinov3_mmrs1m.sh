#!/bin/bash
# DINOv3 + MMRS-1M 阶段一训练启动脚本
# 适用于T20服务器8卡A100环境

set -e

# 配置参数
CONFIG_FILE="configs/train_dinov3_mmrs1m.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_stage1"
GPUS=8
PORT=${PORT:-29500}

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建工作目录
mkdir -p "$WORK_DIR"

# 检查MMRS-1M数据集
if [ ! -d "/workspace/data/mmrs1m" ]; then
    echo "❌ MMRS-1M数据集不存在: /workspace/data/mmrs1m"
    echo "请确保数据集已正确挂载到T20服务器"
    exit 1
fi

# 检查DINOv3预训练权重
if [ ! -f "/workspace/weights/pretrained/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" ]; then
    echo "⚠️  DINOv3预训练权重不存在，将使用官方权重"
    echo "建议下载遥感预训练权重以获得更好效果"
fi

# 打印训练信息
echo "🚀 开始DINOv3 + MMRS-1M阶段一训练"
echo "📁 配置文件: $CONFIG_FILE"
echo "💾 工作目录: $WORK_DIR"
echo "🔧 GPU数量: $GPUS"
echo "🌐 端口: $PORT"
echo "📊 数据集: MMRS-1M (/workspace/data/mmrs1m)"
echo "🏗️ 模型: DINOv3-ViT-L/16"
echo "⏱️  预计训练时间: 5-7天"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=$PORT

# 启动分布式训练
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
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