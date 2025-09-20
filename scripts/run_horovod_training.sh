#!/bin/bash

# TopsDL官方推荐的OpenMPI+Horovod分布式训练启动脚本
# 基于官方文档的hostfile和mpirun配置

set -e

# 默认参数
CONFIG_FILE=""
WORK_DIR=""
RESUME=""
SLOTS_PER_NODE=8
TOTAL_PROCESSES=8
ENABLE_AMP=false
AUTO_SCALE_LR=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --slots-per-node)
            SLOTS_PER_NODE="$2"
            shift 2
            ;;
        --total-processes)
            TOTAL_PROCESSES="$2"
            shift 2
            ;;
        --amp)
            ENABLE_AMP=true
            shift
            ;;
        --auto-scale-lr)
            AUTO_SCALE_LR=true
            shift
            ;;
        --help)
            echo "Usage: $0 --config CONFIG_FILE [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config CONFIG_FILE        训练配置文件路径 (必需)"
            echo "  --work-dir WORK_DIR         工作目录"
            echo "  --resume RESUME             恢复训练的检查点路径"
            echo "  --slots-per-node SLOTS      每个节点的slots数量 (默认: 8)"
            echo "  --total-processes TOTAL     总进程数 (默认: 8)"
            echo "  --amp                       启用自动混合精度"
            echo "  --auto-scale-lr             自动缩放学习率"
            echo "  --help                      显示此帮助信息"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=== TopsDL Horovod分布式训练启动脚本 ==="
echo "配置文件: $CONFIG_FILE"
echo "每节点slots: $SLOTS_PER_NODE"
echo "总进程数: $TOTAL_PROCESSES"
echo "工作目录: ${WORK_DIR:-"自动生成"}"
echo "恢复训练: ${RESUME:-"否"}"
echo "自动混合精度: $ENABLE_AMP"
echo "自动缩放学习率: $AUTO_SCALE_LR"
echo "=========================================="

# 创建MPI hostfile目录
echo "创建MPI配置目录..."
mkdir -p /etc/mpi

# 生成hostfile（按照TopsDL官方文档格式）
echo "生成hostfile..."
echo "localhost slots=$SLOTS_PER_NODE" > /etc/mpi/hostfile

# 如果有worker节点，添加到hostfile
if [[ -f "/etc/volcano/worker.host" ]]; then
    echo "发现worker节点配置，添加到hostfile..."
    cat /etc/volcano/worker.host | sed "s/$/& slots=$SLOTS_PER_NODE/g" >> /etc/mpi/hostfile
fi

echo "Hostfile内容:"
cat /etc/mpi/hostfile
echo ""

# 构建训练命令参数
TRAIN_ARGS="$CONFIG_FILE"

if [[ -n "$WORK_DIR" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --work-dir $WORK_DIR"
fi

if [[ -n "$RESUME" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME"
fi

if [[ "$ENABLE_AMP" == "true" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --amp"
fi

if [[ "$AUTO_SCALE_LR" == "true" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --auto-scale-lr"
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_distributed_horovod_8card_gcu.py"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

echo "训练脚本: $TRAIN_SCRIPT"
echo "训练参数: $TRAIN_ARGS"
echo ""

# 设置环境变量
export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"

# 按照TopsDL官方文档格式执行mpirun命令
echo "启动Horovod分布式训练..."
echo "执行命令:"
echo "mpirun --hostfile /etc/mpi/hostfile --allow-run-as-root -mca btl ^openib -np $TOTAL_PROCESSES python3 $TRAIN_SCRIPT $TRAIN_ARGS"
echo ""

# 执行训练
mpirun \
    --hostfile /etc/mpi/hostfile \
    --allow-run-as-root \
    -mca btl ^openib \
    -np $TOTAL_PROCESSES \
    python3 "$TRAIN_SCRIPT" $TRAIN_ARGS

echo ""
echo "训练完成!"