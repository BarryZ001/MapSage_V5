#!/bin/bash

# MMRS-1M DINOv3 8卡分布式训练启动脚本
# 专门适配燧原T20 GCU计算环境
# 基于PyTorch分布式训练框架

echo "🚀 启动 MMRS-1M DINOv3 8卡分布式训练"
echo "🔥 计算环境: 燧原T20 GCU"
echo "📊 数据集: MMRS-1M 多模态遥感数据集"
echo "🏗️ 模型: DINOv3-ViT-L/16 + VisionTransformerUpHead"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 8卡GCU设备
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0

# 设置GCU相关环境变量
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOPS_LAUNCH_MODE=pytorch
export ECCL_DEBUG=INFO
export ECCL_TIMEOUT=1800

# 设置MMSegmentation环境变量
export MMCV_WITH_OPS=1
export MAX_JOBS=8

# 配置文件路径
CONFIG_FILE="configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu_8card"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 创建工作目录
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/tf_logs"

# 检查数据集路径
DATA_ROOT="/workspace/data/mmrs1m/data"
LOCAL_DATA_ROOT="./data"

if [ -d "$DATA_ROOT" ]; then
    echo "✅ 使用服务器数据路径: $DATA_ROOT"
    ACTUAL_DATA_ROOT="$DATA_ROOT"
elif [ -d "$LOCAL_DATA_ROOT" ]; then
    echo "✅ 使用本地数据路径: $LOCAL_DATA_ROOT"
    ACTUAL_DATA_ROOT="$LOCAL_DATA_ROOT"
else
    echo "❌ 错误: 数据集路径不存在，请检查数据集配置"
    echo "   服务器路径: $DATA_ROOT"
    echo "   本地路径: $LOCAL_DATA_ROOT"
    exit 1
fi

# 打印训练配置信息
echo ""
echo "📋 训练配置信息:"
echo "   配置文件: $CONFIG_FILE"
echo "   工作目录: $WORK_DIR"
echo "   数据路径: $ACTUAL_DATA_ROOT"
echo "   设备数量: 8 GCUs"
echo "   批次大小: 2 x 8 = 16"
echo "   最大迭代: 40000"
echo ""

# 启动8卡分布式训练
echo "🔄 启动分布式训练..."

# 使用torchrun启动分布式训练
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train.py \
    "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    --seed 42 \
    --deterministic \
    --diff-seed \
    --cfg-options \
        data_root="$ACTUAL_DATA_ROOT" \
        train_dataloader.dataset.data_root="$ACTUAL_DATA_ROOT" \
        val_dataloader.dataset.data_root="$ACTUAL_DATA_ROOT" \
        test_dataloader.dataset.data_root="$ACTUAL_DATA_ROOT" \
    2>&1 | tee "$WORK_DIR/training.log"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 8卡分布式训练完成!"
    echo "📁 训练日志: $WORK_DIR/training.log"
    echo "📁 检查点: $WORK_DIR/"
    echo "📊 TensorBoard日志: $WORK_DIR/tf_logs/"
    echo ""
    echo "🔍 查看训练日志:"
    echo "   tail -f $WORK_DIR/training.log"
    echo ""
    echo "📈 启动TensorBoard:"
    echo "   tensorboard --logdir=$WORK_DIR/tf_logs --port=6006"
else
    echo ""
    echo "❌ 8卡分布式训练失败!"
    echo "📁 检查训练日志: $WORK_DIR/training.log"
    echo "🔍 查看错误信息:"
    echo "   tail -50 $WORK_DIR/training.log"
    exit 1
fi