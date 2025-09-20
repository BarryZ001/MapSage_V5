#!/bin/bash
# 8卡分布式训练启动脚本 - 燧原T20 GCU版本

set -e

echo "🚀 启动8卡分布式训练"
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

# 创建日志目录
LOG_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu_8card/logs"
mkdir -p "$LOG_DIR"

# 启动8卡分布式训练
echo "🔥 启动8个训练进程..."

for i in {0..7}; do
    export RANK=$i
    export LOCAL_RANK=$i
    
    echo "🚀 启动进程 $i/8"
    
    # 每个进程的日志文件
    LOG_FILE="$LOG_DIR/train_rank_${i}.log"
    
    # 启动训练进程（后台运行）
    python "$SCRIPT_FILE" "$CONFIG_FILE" --launcher pytorch --local_rank $i > "$LOG_FILE" 2>&1 &
    
    # 记录进程ID
    PID=$!
    echo "📝 进程 $i PID: $PID"
    echo $PID > "$LOG_DIR/train_rank_${i}.pid"
    
    # 短暂延迟，避免同时启动造成冲突
    sleep 2
done

echo "✅ 所有训练进程已启动"
echo "📊 监控训练进度:"
echo "  - 日志目录: $LOG_DIR"
echo "  - 主进程日志: $LOG_DIR/train_rank_0.log"
echo "  - 工作目录: ./work_dirs/dinov3_mmrs1m_t20_gcu_8card"

echo ""
echo "🔍 实时监控命令:"
echo "  tail -f $LOG_DIR/train_rank_0.log"
echo ""
echo "🛑 停止训练命令:"
echo "  pkill -f train_distributed_8card_gcu.py"
echo ""

# 等待所有进程完成
wait

echo "🎉 8卡分布式训练完成!"