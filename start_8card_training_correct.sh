#!/bin/bash
# 正确的8卡分布式训练启动脚本 - 燧原T20 GCU版本
# 基于成功经验优化，确保所有8张GCU卡都参与训练

set -e

echo "🚀 启动MapSage V5 - 8卡分布式训练"
echo "📅 时间: $(date)"
echo "🖥️  主机: $(hostname)"
echo "📂 工作目录: $(pwd)"

# 预检查：验证环境
echo "🔍 运行环境预检查..."
if [ -f "validate_t20_environment_complete.py" ]; then
    echo "📋 执行完整环境验证..."
    python3 validate_t20_environment_complete.py
    if [ $? -ne 0 ]; then
        echo "❌ 环境验证失败，请先修复环境问题"
        exit 1
    fi
    echo "✅ 环境验证通过"
else
    echo "⚠️  环境验证脚本不存在，跳过预检查"
fi

# 检查GCU设备（容器内使用Python检查）
echo "🔍 检查GCU设备状态..."
python3 -c "
import sys
try:
    import torch_gcu
    if hasattr(torch_gcu, 'is_available') and torch_gcu.is_available():
        device_count = torch_gcu.device_count()
        print(f'✅ 检测到 {device_count} 张GCU设备')
        if device_count < 8:
            print(f'⚠️  警告: 检测到设备数量({device_count})少于8张')
        for i in range(min(device_count, 8)):
            try:
                device = torch_gcu.device(i)
                print(f'   - GCU设备 {i}: 可用 (device: {device})')
            except Exception as e:
                print(f'   - GCU设备 {i}: 错误 - {e}')
                sys.exit(1)
    else:
        print('❌ torch_gcu不可用或未检测到GCU设备')
        sys.exit(1)
except ImportError:
    print('❌ torch_gcu模块未安装')
    sys.exit(1)
except Exception as e:
    print(f'❌ GCU设备检查失败: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GCU设备检查失败，无法继续训练"
    exit 1
fi

# 配置分布式训练环境变量
echo "⚙️  配置分布式训练环境..."
export WORLD_SIZE=8
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

# 配置GCU相关环境变量（基于成功经验）
export ECCL_BACKEND=gloo
export ECCL_DEVICE_TYPE=gcu
export ECCL_DEBUG=0  # 生产环境关闭调试
export ECCL_TIMEOUT=1800  # 30分钟超时

# 网络配置优化
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# 配置训练参数
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_final"
TRAIN_SCRIPT="scripts/train_distributed_pytorch_ddp_8card_gcu.py"

echo "📄 配置文件: $CONFIG_FILE"
echo "📁 工作目录: $WORK_DIR"
echo "🐍 训练脚本: $TRAIN_SCRIPT"

# 检查必要文件
echo "🔍 检查必要文件..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "📝 可用的配置文件:"
    ls -la configs/*.py | head -5
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 训练脚本不存在: $TRAIN_SCRIPT"
    echo "📝 可用的训练脚本:"
    ls -la scripts/train*.py | head -5
    exit 1
fi

# 创建工作目录
echo "📁 准备工作目录..."
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/logs"

# 停止之前的训练进程
echo "🛑 停止之前的训练进程..."
pkill -f "train_distributed_pytorch_ddp_8card_gcu.py" || true
pkill -f "train_distributed_8card_gcu.py" || true
pkill -f "torchrun.*train" || true
sleep 3

# 设置Python路径
export PYTHONPATH="${PWD}:${PYTHONPATH}"
echo "🐍 Python路径: $PYTHONPATH"

# 检查分布式后端可用性
echo "🌐 检查分布式后端..."
python3 -c "
import torch.distributed as dist
print('分布式可用:', dist.is_available())
try:
    if hasattr(dist, 'is_gloo_available'):
        print('Gloo后端:', dist.is_gloo_available())
    if hasattr(dist, 'is_nccl_available'):
        print('NCCL后端:', dist.is_nccl_available())
except:
    pass
try:
    import eccl
    print('ECCL后端: 可用')
except ImportError:
    print('ECCL后端: 不可用')
"

# 记录启动信息
echo "📝 记录启动信息到日志..."
{
    echo "=== 训练启动信息 ==="
    echo "时间: $(date)"
    echo "主机: $(hostname)"
    echo "用户: $(whoami)"
    echo "工作目录: $(pwd)"
    echo "Python版本: $(python3 --version)"
    echo "配置文件: $CONFIG_FILE"
    echo "工作目录: $WORK_DIR"
    echo "环境变量:"
    echo "  WORLD_SIZE=$WORLD_SIZE"
    echo "  MASTER_ADDR=$MASTER_ADDR"
    echo "  MASTER_PORT=$MASTER_PORT"
    echo "  ECCL_BACKEND=$ECCL_BACKEND"
    echo "  ECCL_DEVICE_TYPE=$ECCL_DEVICE_TYPE"
    echo "===================="
} > "$WORK_DIR/logs/training_start_$(date +%Y%m%d_%H%M%S).log"

# 使用torchrun启动8卡分布式训练
echo "🚀 使用torchrun启动8卡分布式训练..."
echo "⏰ 启动时间: $(date)"

# 启动训练（添加更多参数和错误处理）
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --max_restarts=3 \
    --rdzv_backend=c10d \
    "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    --seed 42 \
    --deterministic \
    2>&1 | tee "$WORK_DIR/logs/training_output_$(date +%Y%m%d_%H%M%S).log"

# 检查训练结果
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 8卡分布式训练启动成功"
    echo "⏰ 完成时间: $(date)"
else
    echo "❌ 训练启动失败，退出码: $TRAIN_EXIT_CODE"
    echo "📝 请检查日志文件: $WORK_DIR/logs/"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "📊 监控和调试信息:"
echo "📊 容器外监控GCU: 'efsmi'"
echo "📊 容器内检查设备: 'python3 -c \"import torch_gcu; print(torch_gcu.device_count())\"'"
echo "📊 查看进程: 'ps aux | grep train'"
echo "📊 查看GPU使用: 'watch -n 1 efsmi'"
echo "📝 训练日志: $WORK_DIR/logs/"
echo "📝 工作目录: $WORK_DIR"
echo ""
echo "🎉 训练启动完成！"