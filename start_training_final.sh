#!/bin/bash
# MapSage V5 统一训练启动脚本
# 适用于T20服务器8卡GCU环境
# 这是唯一的训练启动脚本，避免混乱

set -e

echo "🚀 MapSage V5 统一训练启动脚本"
echo "=================================="

# 基础配置
PROJECT_ROOT="/workspace/MapSage_V5"
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_final"
GPUS=8
PORT=${PORT:-29500}

# 检查环境
echo "🔍 检查训练环境..."

# 1. 检查项目目录
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "❌ 项目目录不存在: $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# 2. 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "可用的配置文件:"
    ls -la configs/train_dinov3_*.py
    exit 1
fi

# 3. 检查训练脚本
TRAIN_SCRIPT="scripts/train_distributed_pytorch_ddp_8card_gcu.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 4. 检查数据集
DATA_ROOT="/workspace/data/mmrs1m"
if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  数据集目录不存在: $DATA_ROOT"
    echo "请确保数据集已正确挂载"
fi

# 5. 创建工作目录
mkdir -p "$WORK_DIR"

# 6. 检查GCU环境
echo "🔧 检查GCU环境..."
python3 -c "
import torch
try:
    import torch_gcu
    print('✅ torch_gcu 可用')
    if hasattr(torch_gcu, 'device_count'):
        device_count = torch_gcu.device_count()
        print(f'✅ 检测到 {device_count} 个GCU设备')
    else:
        print('✅ torch_gcu 已导入（设备数量检测不可用）')
except ImportError:
    print('❌ torch_gcu 不可用')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GCU环境检查失败"
    exit 1
fi

# 打印训练信息
echo ""
echo "📋 训练配置信息:"
echo "  项目目录: $PROJECT_ROOT"
echo "  配置文件: $CONFIG_FILE"
echo "  工作目录: $WORK_DIR"
echo "  训练脚本: $TRAIN_SCRIPT"
echo "  GPU数量: $GPUS"
echo "  端口: $PORT"
echo "  数据集: $DATA_ROOT"
echo ""

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export MASTER_PORT=$PORT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 启动分布式训练
echo "🎯 启动分布式训练..."
echo "命令: python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT $TRAIN_SCRIPT --config $CONFIG_FILE --work-dir $WORK_DIR --launcher pytorch"
echo ""

python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    --seed 42 \
    --deterministic

TRAIN_EXIT_CODE=$?

echo ""
echo "=================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
    echo "📊 训练日志: $WORK_DIR/"
    echo "💾 模型权重: $WORK_DIR/latest.pth"
    echo "📈 TensorBoard: tensorboard --logdir=$WORK_DIR/tf_logs/"
else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "请检查训练日志: $WORK_DIR/"
fi
echo "=================================="

exit $TRAIN_EXIT_CODE