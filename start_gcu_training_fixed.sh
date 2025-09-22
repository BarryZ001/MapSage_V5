#!/bin/bash
# GCU分布式训练启动脚本（修复版）

set -e  # 遇到错误立即退出

echo "🚀 启动GCU分布式训练（修复版）"

# 检查GCU设备
echo "🔍 检查GCU设备..."
python3 -c "
try:
    import torch_gcu
    if torch_gcu.is_available():
        device_count = torch_gcu.device_count()
        print(f'✅ GCU设备可用，数量: {device_count}')
        if device_count < 8:
            print(f'⚠️  警告: 检测到{device_count}个GCU设备，少于8个')
    else:
        print('❌ GCU设备不可用')
        exit(1)
except ImportError:
    print('❌ torch_gcu未安装')
    exit(1)
"

# 设置环境变量
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# GCU相关环境变量
export ECCL_BACKEND=gloo
export ECCL_DEVICE_TYPE=GCU
export ECCL_DEBUG=0
export CUDA_VISIBLE_DEVICES=""

# 分布式训练环境变量
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 网络配置
export GLOO_SOCKET_IFNAME=lo
export GLOO_TIMEOUT_SECONDS=300

# 训练参数
CONFIG_FILE="configs/dinov3/dinov3_vit-l16_mmrs1m_t20_gcu.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu"
TRAIN_SCRIPT="scripts/train_distributed_pytorch_ddp_8card_gcu.py"

# 检查必要文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 创建工作目录
mkdir -p "$WORK_DIR"

# 停止之前的训练进程
echo "🛑 停止之前的训练进程..."
pkill -f "train_distributed_pytorch_ddp_8card_gcu.py" || true
sleep 2

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "📋 训练配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  工作目录: $WORK_DIR"
echo "  训练脚本: $TRAIN_SCRIPT"
echo "  设备数量: $WORLD_SIZE"
echo "  主节点: $MASTER_ADDR:$MASTER_PORT"

# 启动分布式训练
echo "🚀 启动8卡GCU分布式训练..."

# 使用torchrun启动分布式训练
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29500 \
    "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch

echo "✅ 训练启动完成"
echo "📊 监控GCU设备使用情况:"
echo "   使用命令: watch -n 1 'python3 -c "import torch_gcu; print(f\"GCU设备数量: {torch_gcu.device_count()}\")"'"
echo "📁 日志保存在: $WORK_DIR"
