#!/bin/bash
# scripts/start_distributed_manual.sh - 手动启动分布式训练

set -e

# 配置参数
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu.py"
WORLD_SIZE=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

echo "🚀 手动启动DINOv3分布式训练 - 使用${WORLD_SIZE}个GCU"
echo "📝 配置文件: ${CONFIG_FILE}"
echo "🌐 Master地址: ${MASTER_ADDR}:${MASTER_PORT}"

# 设置环境变量
export TORCH_DEVICE=xla
export XLA_USE_BF16=1
export MMENGINE_DEVICE=xla
export CUDA_VISIBLE_DEVICES=""
export WORLD_SIZE=${WORLD_SIZE}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

# 创建日志目录
mkdir -p logs

# 启动分布式训练进程
for i in $(seq 0 $((WORLD_SIZE-1))); do
    echo "🔄 启动进程 ${i}/${WORLD_SIZE}..."
    
    # 在后台启动每个进程
    RANK=${i} LOCAL_RANK=${i} python3 scripts/train_distributed.py ${CONFIG_FILE} --launcher pytorch > "logs/train_rank_${i}.log" 2>&1 &
    
    # 记录进程ID
    echo $! > "logs/train_rank_${i}.pid"
    
    echo "✅ 进程 ${i} 已启动 (PID: $!)"
    
    # 短暂延迟避免同时启动
    sleep 3
done

echo "🎉 所有${WORLD_SIZE}个训练进程已启动"
echo "📊 监控命令: tail -f logs/train_rank_*.log"
echo "🛑 停止命令: ./scripts/stop_distributed_training.sh"

# 等待所有进程
wait