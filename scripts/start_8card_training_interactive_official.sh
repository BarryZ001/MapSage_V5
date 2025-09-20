#!/bin/bash
# -*- coding: utf-8 -*-
# T20服务器8卡训练交互式启动脚本 - 基于燧原官方文档

echo "🚀 T20服务器8卡训练交互式启动 (燧原官方配置)"
echo "=================================================="

# 1. 检查torch_gcu环境
echo "📋 检查torch_gcu环境..."
python3 -c "
import torch
import torch_gcu
print('✅ PyTorch版本:', torch.__version__)
print('✅ torch_gcu可用:', torch_gcu.is_available())
if torch_gcu.is_available():
    print('✅ GCU设备数:', torch_gcu.device_count())
else:
    print('❌ torch_gcu不可用，请检查安装')
    exit(1)
" || {
    echo "❌ torch_gcu环境检查失败，请先运行环境修复脚本"
    exit 1
}

# 2. 设置环境变量 (根据官方文档)
echo ""
echo "🔧 设置训练环境变量..."

# 基础分布式环境变量
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export WORLD_SIZE=${WORLD_SIZE:-"8"}

# 燧原专用环境变量 (根据官方文档)
export PYTORCH_GCU_ALLOC_CONF=${PYTORCH_GCU_ALLOC_CONF:-"backend:topsMallocAsync"}
export TORCH_ECCL_AVOID_RECORD_STREAMS=${TORCH_ECCL_AVOID_RECORD_STREAMS:-"false"}
export TORCH_ECCL_ASYNC_ERROR_HANDLING=${TORCH_ECCL_ASYNC_ERROR_HANDLING:-"3"}

# 调试信息 (可选)
# export ENFLAME_LOG_DEBUG_LEVEL="DEBUG"
# export ENFLAME_LOG_DEBUG_MOD="TORCH_GCU/OP,TORCH_GCU/FALLBACK"

echo "  - MASTER_ADDR: $MASTER_ADDR"
echo "  - MASTER_PORT: $MASTER_PORT"
echo "  - WORLD_SIZE: $WORLD_SIZE"
echo "  - PYTORCH_GCU_ALLOC_CONF: $PYTORCH_GCU_ALLOC_CONF"

# 3. 选择配置文件
echo ""
echo "📝 选择训练配置:"
echo "1. DINOv3 + LoveDA (推荐)"
echo "2. DINOv3 + MMRS1M"
echo "3. 自定义配置文件"
echo ""
read -p "请选择配置 (1-3): " config_choice

case $config_choice in
    1)
        CONFIG_FILE="configs/train_dinov3_loveda_t20_gcu.py"
        echo "✅ 选择配置: DINOv3 + LoveDA"
        ;;
    2)
        CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
        echo "✅ 选择配置: DINOv3 + MMRS1M"
        ;;
    3)
        echo ""
        echo "可用配置文件:"
        ls -1 configs/*t20*gcu*.py 2>/dev/null || echo "  (未找到T20 GCU配置文件)"
        echo ""
        read -p "请输入配置文件路径: " CONFIG_FILE
        ;;
    *)
        echo "❌ 无效选择，使用默认配置"
        CONFIG_FILE="configs/train_dinov3_loveda_t20_gcu.py"
        ;;
esac

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "📄 使用配置文件: $CONFIG_FILE"

# 4. 设置工作目录
WORK_DIR="test_work_dir/$(basename $CONFIG_FILE .py)_$(date +%Y%m%d_%H%M%S)"
echo "📁 工作目录: $WORK_DIR"

# 5. 选择启动方式
echo ""
echo "🎯 选择启动方式:"
echo "1. 交互式启动 (推荐，可实时查看输出)"
echo "2. 后台启动 (日志保存到文件)"
echo ""
read -p "请选择启动方式 (1-2): " start_choice

# 6. 构建训练命令
TRAIN_CMD="python3 scripts/train_distributed_8card_gcu.py \\
    --config $CONFIG_FILE \\
    --work-dir $WORK_DIR \\
    --launcher pytorch"

echo ""
echo "🚀 准备启动8卡分布式训练..."
echo "命令: $TRAIN_CMD"
echo ""

case $start_choice in
    1)
        echo "📺 交互式启动 - 按Ctrl+C停止训练"
        echo "=================================="
        sleep 2
        
        # 交互式启动
        eval $TRAIN_CMD
        ;;
    2)
        LOG_FILE="$WORK_DIR/training.log"
        mkdir -p $(dirname $LOG_FILE)
        
        echo "📝 后台启动 - 日志文件: $LOG_FILE"
        echo "使用以下命令查看日志:"
        echo "  tail -f $LOG_FILE"
        echo ""
        
        # 后台启动
        nohup bash -c "$TRAIN_CMD" > $LOG_FILE 2>&1 &
        TRAIN_PID=$!
        
        echo "✅ 训练已启动，进程ID: $TRAIN_PID"
        echo "📊 实时监控日志:"
        echo "=================================="
        
        # 显示前几行日志
        sleep 3
        tail -f $LOG_FILE
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 训练启动完成！"

# 7. 训练后操作提示
echo ""
echo "💡 训练管理命令:"
echo "  - 查看进程: ps aux | grep train_distributed"
echo "  - 停止训练: pkill -f train_distributed"
echo "  - 查看GPU使用: watch -n 1 'python3 -c \"import torch_gcu; print(torch_gcu.device_count())\"'"
echo "  - 查看日志: tail -f $WORK_DIR/training.log"