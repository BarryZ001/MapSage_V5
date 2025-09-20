#!/bin/bash

# 启动8卡分布式训练脚本
cd /workspace/code/MapSage_V5

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "🚀 启动DINOv3分布式训练 - 8卡GCU"
echo "📊 配置文件: configs/train_dinov3_mmrs1m_t20_gcu.py"
echo "🌐 Master地址: $MASTER_ADDR:$MASTER_PORT"

# 启动8个训练进程
for i in {0..7}; do
    export RANK=$i
    export LOCAL_RANK=$i
    export WORLD_SIZE=8
    
    nohup python scripts/train_distributed.py configs/train_dinov3_mmrs1m_t20_gcu.py --launcher pytorch > logs/train_rank_$i.log 2>&1 &
    PID=$!
    echo "✅ 启动进程 $i，PID: $PID"
done

echo "🎉 所有8个训练进程已启动"
echo "📝 监控命令: tail -f logs/train_rank_0.log"
echo "🛑 停止命令: pkill -f train_distributed"

# 等待所有进程
wait