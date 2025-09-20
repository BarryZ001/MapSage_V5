#!/bin/bash

# PyTorch DDP分布式训练启动脚本
# 适用于燧原T20 GCU环境
# 不依赖Horovod，使用原生PyTorch分布式训练

set -e

# 默认参数
CONFIG_FILE=""
WORK_DIR="./work_dirs/pytorch_ddp_training"
RESUME=""
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
AMP=false
AUTO_SCALE_LR=false

# 显示帮助信息
show_help() {
    echo "PyTorch DDP分布式训练启动脚本"
    echo ""
    echo "用法: $0 --config CONFIG_FILE [选项]"
    echo ""
    echo "必需参数:"
    echo "  --config CONFIG_FILE        训练配置文件路径"
    echo ""
    echo "选项:"
    echo "  --work-dir WORK_DIR         工作目录 (默认: ./work_dirs/pytorch_ddp_training)"
    echo "  --resume CHECKPOINT         恢复训练的检查点路径"
    echo "  --nnodes NNODES             节点数量 (默认: 1)"
    echo "  --nproc-per-node NPROC      每个节点的进程数 (默认: 8)"
    echo "  --node-rank NODE_RANK       当前节点的rank (默认: 0)"
    echo "  --master-addr MASTER_ADDR   主节点地址 (默认: localhost)"
    echo "  --master-port MASTER_PORT   主节点端口 (默认: 29500)"
    echo "  --amp                       启用自动混合精度训练"
    echo "  --auto-scale-lr             根据GPU数量自动缩放学习率"
    echo "  --help                      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 单节点8卡训练"
    echo "  $0 --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py --amp"
    echo ""
    echo "  # 多节点训练 (节点0)"
    echo "  $0 --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \\"
    echo "     --nnodes 2 --node-rank 0 --master-addr 192.168.1.100 --amp"
    echo ""
    echo "  # 多节点训练 (节点1)"
    echo "  $0 --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \\"
    echo "     --nnodes 2 --node-rank 1 --master-addr 192.168.1.100 --amp"
}

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
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --amp)
            AMP=true
            shift
            ;;
        --auto-scale-lr)
            AUTO_SCALE_LR=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$CONFIG_FILE" ]]; then
    echo "错误: 必须指定配置文件 --config"
    show_help
    exit 1
fi

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 显示配置信息
echo "=== PyTorch DDP分布式训练启动脚本 ==="
echo "配置文件: $CONFIG_FILE"
echo "工作目录: $WORK_DIR"
echo "节点数量: $NNODES"
echo "每节点进程数: $NPROC_PER_NODE"
echo "当前节点rank: $NODE_RANK"
echo "主节点地址: $MASTER_ADDR"
echo "主节点端口: $MASTER_PORT"
echo "恢复训练: ${RESUME:-否}"
echo "自动混合精度: $AMP"
echo "自动缩放学习率: $AUTO_SCALE_LR"
echo "=========================================="

# 创建工作目录
mkdir -p "$WORK_DIR"

# 构建训练脚本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_distributed_pytorch_ddp_8card_gcu.py"

# 检查训练脚本是否存在
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "错误: 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

echo "训练脚本: $TRAIN_SCRIPT"

# 构建训练参数
TRAIN_ARGS="$CONFIG_FILE --work-dir $WORK_DIR"

if [[ -n "$RESUME" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME"
fi

if [[ "$AMP" == "true" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --amp"
fi

if [[ "$AUTO_SCALE_LR" == "true" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --auto-scale-lr"
fi

echo "训练参数: $TRAIN_ARGS"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动分布式训练
echo ""
echo "启动PyTorch DDP分布式训练..."
echo "执行命令:"
echo "python3 -m torch.distributed.launch \\"
echo "    --nnodes=$NNODES \\"
echo "    --nproc_per_node=$NPROC_PER_NODE \\"
echo "    --node_rank=$NODE_RANK \\"
echo "    --master_addr=$MASTER_ADDR \\"
echo "    --master_port=$MASTER_PORT \\"
echo "    $TRAIN_SCRIPT $TRAIN_ARGS"
echo ""

python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$TRAIN_SCRIPT" $TRAIN_ARGS