#!/bin/bash
# scripts/stop_distributed_training.sh - 停止分布式训练

set -e

echo "🛑 停止分布式训练进程..."

# 停止所有训练进程
for pid_file in logs/train_rank_*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        rank=$(echo "$pid_file" | sed 's/.*train_rank_\([0-9]*\)\.pid/\1/')
        
        if kill -0 "$pid" 2>/dev/null; then
            echo "🔄 停止进程 Rank ${rank} (PID: ${pid})"
            kill "$pid"
            sleep 1
            
            # 如果进程仍在运行，强制终止
            if kill -0 "$pid" 2>/dev/null; then
                echo "⚠️ 强制终止进程 Rank ${rank} (PID: ${pid})"
                kill -9 "$pid"
            fi
        else
            echo "ℹ️ 进程 Rank ${rank} (PID: ${pid}) 已经停止"
        fi
        
        # 删除PID文件
        rm -f "$pid_file"
    fi
done

# 清理任何残留的训练进程
pkill -f "train_distributed.py" 2>/dev/null || true

echo "✅ 所有分布式训练进程已停止"