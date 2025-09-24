#!/bin/bash
# DINOv3 + MMRS-1M 8卡分布式训练启动脚本
# 基于成功的燧原T20 GCU经验，适配DINOv3训练

set -e  # 遇到错误立即退出

echo "🚀 DINOv3 + MMRS-1M 8卡分布式训练启动脚本"
echo "=" * 60

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📁 项目根目录: $PROJECT_ROOT"

# 配置参数
CONFIG_FILE="${1:-configs/train_dinov3_mmrs1m_t20_gcu_8card.py}"
WORK_DIR="${2:-./work_dirs/dinov3_mmrs1m_8card_gcu}"
MASTER_PORT="${3:-29500}"
NUM_GPUS=8

echo "⚙️ 训练配置:"
echo "   - 配置文件: $CONFIG_FILE"
echo "   - 工作目录: $WORK_DIR"
echo "   - 主端口: $MASTER_PORT"
echo "   - GPU数量: $NUM_GPUS"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查训练脚本
TRAIN_SCRIPT="scripts/train_dinov3_distributed_8card_gcu.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 创建工作目录
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/logs"

echo "✅ 预检查完成"

# 设置燧原GCU环境变量 - 基于成功的demo_deepspeed_xla.py经验
echo "🔧 设置燧原GCU环境变量..."

export PYTORCH_GCU_ALLOC_CONF="backend:topsMallocAsync"
export TORCH_ECCL_AVOID_RECORD_STREAMS="false"
export TORCH_ECCL_ASYNC_ERROR_HANDLING="3"

# 分布式训练环境变量
export MASTER_ADDR="localhost"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE="$NUM_GPUS"

# 设置CUDA相关环境变量（即使使用GCU，某些库仍需要这些变量）
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Python路径设置
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "✅ 环境变量设置完成"

# 检查torch_gcu环境
echo "🔍 检查torch_gcu环境..."
python3 -c "
import torch
import torch_gcu
print(f'PyTorch版本: {torch.__version__}')
print(f'torch_gcu可用: {torch_gcu.is_available()}')
if torch_gcu.is_available():
    print(f'GCU设备数: {torch_gcu.device_count()}')
else:
    print('❌ torch_gcu不可用')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ torch_gcu环境检查失败"
    exit 1
fi

echo "✅ torch_gcu环境检查通过"

# 检查数据集路径（如果是T20服务器）
if [ -d "/workspace/data/mmrs1m/data" ]; then
    echo "✅ 检测到T20服务器环境，MMRS-1M数据集路径存在"
    ls -la /workspace/data/mmrs1m/data/ | head -10
else
    echo "⚠️ 未检测到T20服务器MMRS-1M数据集路径，将使用配置文件中的路径"
fi

# 显示训练命令
TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TRAIN_SCRIPT $CONFIG_FILE --work-dir $WORK_DIR"

echo ""
echo "🚀 即将执行训练命令:"
echo "   $TRAIN_CMD"
echo ""

# 询问用户确认
read -p "是否开始训练？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 用户取消训练"
    exit 0
fi

echo "🚀 开始DINOv3 8卡分布式训练..."
echo "=" * 60

# 记录训练开始时间
START_TIME=$(date)
echo "⏰ 训练开始时间: $START_TIME"

# 创建日志文件
LOG_FILE="$WORK_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件: $LOG_FILE"

# 执行训练命令，同时输出到控制台和日志文件
echo "执行命令: $TRAIN_CMD" | tee "$LOG_FILE"
echo "开始时间: $START_TIME" | tee -a "$LOG_FILE"
echo "=" * 60 | tee -a "$LOG_FILE"

# 使用exec重定向，确保能捕获所有输出
exec > >(tee -a "$LOG_FILE") 2>&1

# 执行训练
$TRAIN_CMD

# 记录训练结束时间
END_TIME=$(date)
echo "=" * 60
echo "✅ 训练完成!"
echo "⏰ 开始时间: $START_TIME"
echo "⏰ 结束时间: $END_TIME"
echo "📝 日志文件: $LOG_FILE"
echo "📁 工作目录: $WORK_DIR"

# 显示训练结果
if [ -d "$WORK_DIR" ]; then
    echo ""
    echo "📊 训练结果:"
    echo "   - 工作目录: $WORK_DIR"
    echo "   - 日志文件: $LOG_FILE"
    
    # 显示最新的检查点
    if [ -d "$WORK_DIR" ]; then
        LATEST_CKPT=$(find "$WORK_DIR" -name "*.pth" -type f -exec ls -t {} + | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "   - 最新检查点: $LATEST_CKPT"
        fi
    fi
    
    # 显示TensorBoard命令
    if [ -d "$WORK_DIR/tf_logs" ]; then
        echo ""
        echo "📈 查看训练曲线:"
        echo "   tensorboard --logdir=$WORK_DIR/tf_logs --port=6006"
    fi
fi

echo ""
echo "🎉 DINOv3 8卡分布式训练脚本执行完成!"