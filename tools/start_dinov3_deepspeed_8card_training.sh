#!/bin/bash

# DINOv3 + MMRS-1M 8卡分布式训练启动脚本 (DeepSpeed版本)
# 使用DeepSpeed框架进行GCU环境下的分布式训练

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 脚本信息
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "DINOv3 + MMRS-1M 8卡分布式训练启动脚本 (DeepSpeed版本)"
print_info "项目根目录: $PROJECT_ROOT"
echo "=" * 60

# 默认配置
DEFAULT_CONFIG="$PROJECT_ROOT/configs/dinov3/dinov3_vit-base_mmrs1m_8xb2-160k_512x512.py"
DEFAULT_WORK_DIR="$PROJECT_ROOT/work_dirs/dinov3_deepspeed_8card_gcu"
DEFAULT_STEPS=1000
NUM_GPUS=8

# 解析命令行参数
CONFIG_FILE="$DEFAULT_CONFIG"
WORK_DIR="$DEFAULT_WORK_DIR"
STEPS="$DEFAULT_STEPS"

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
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --config CONFIG_FILE    配置文件路径 (默认: $DEFAULT_CONFIG)"
            echo "  --work-dir WORK_DIR     工作目录 (默认: $DEFAULT_WORK_DIR)"
            echo "  --steps STEPS           训练步数 (默认: $DEFAULT_STEPS)"
            echo "  --num-gpus NUM_GPUS     GPU数量 (默认: 8)"
            echo "  -h, --help              显示帮助信息"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 验证配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 训练脚本路径
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_dinov3_deepspeed_8card_gcu.py"

# 验证训练脚本
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    print_error "训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 显示配置信息
print_info "训练配置:"
echo "   📝 配置文件: $CONFIG_FILE"
echo "   📁 工作目录: $WORK_DIR"
echo "   🚀 训练脚本: $TRAIN_SCRIPT"
echo "   🔢 训练步数: $STEPS"
echo "   🎯 GPU数量: $NUM_GPUS"
echo ""

# 创建工作目录
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/logs"

# 检查Python环境
print_info "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3未找到，请确保已安装Python3"
    exit 1
fi

# 检查必要的Python包
print_info "检查必要的Python包..."
python3 -c "
import sys
try:
    import torch
    import torch_gcu
    import deepspeed
    import mmengine
    import mmseg
    print('✅ 所有必要的包都已安装')
    print(f'   - PyTorch: {torch.__version__}')
    print(f'   - DeepSpeed: {deepspeed.__version__}')
    print(f'   - torch_gcu可用: {torch_gcu.is_available()}')
    if torch_gcu.is_available():
        print(f'   - GCU设备数: {torch_gcu.device_count()}')
except ImportError as e:
    print(f'❌ 缺少必要的包: {e}')
    sys.exit(1)
"

if [[ $? -ne 0 ]]; then
    print_error "Python环境检查失败"
    exit 1
fi

# 设置环境变量
export PYTORCH_GCU_ALLOC_CONF="backend:topsMallocAsync"
export TORCH_ECCL_AVOID_RECORD_STREAMS="false"
export TORCH_ECCL_ASYNC_ERROR_HANDLING="3"

print_info "环境变量设置:"
echo "   - PYTORCH_GCU_ALLOC_CONF: $PYTORCH_GCU_ALLOC_CONF"
echo "   - TORCH_ECCL_AVOID_RECORD_STREAMS: $TORCH_ECCL_AVOID_RECORD_STREAMS"
echo "   - TORCH_ECCL_ASYNC_ERROR_HANDLING: $TORCH_ECCL_ASYNC_ERROR_HANDLING"
echo ""

# 构建DeepSpeed启动命令
DEEPSPEED_CMD="deepspeed --num_gpus=$NUM_GPUS $TRAIN_SCRIPT $CONFIG_FILE --work-dir $WORK_DIR --steps $STEPS"

print_info "即将执行训练命令:"
echo "   $DEEPSPEED_CMD"
echo ""

# 询问用户确认
read -p "是否开始训练？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "用户取消训练"
    exit 0
fi

print_success "开始DINOv3 8卡分布式训练 (DeepSpeed)..."
echo "=" * 60

# 记录训练开始时间
START_TIME=$(date)
print_info "训练开始时间: $START_TIME"

# 创建日志文件
LOG_FILE="$WORK_DIR/logs/train_deepspeed_$(date +%Y%m%d_%H%M%S).log"
print_info "日志文件: $LOG_FILE"

# 执行训练命令，同时输出到控制台和日志文件
echo "执行命令: $DEEPSPEED_CMD" | tee "$LOG_FILE"
echo "开始时间: $START_TIME" | tee -a "$LOG_FILE"
echo "=" * 60 | tee -a "$LOG_FILE"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 执行DeepSpeed训练命令
eval "$DEEPSPEED_CMD" 2>&1 | tee -a "$LOG_FILE"

# 获取训练命令的退出状态
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# 记录训练结束时间
END_TIME=$(date)
echo "=" * 60 | tee -a "$LOG_FILE"
echo "结束时间: $END_TIME" | tee -a "$LOG_FILE"

# 检查训练结果
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    print_success "训练完成!"
    echo "⏰ 开始时间: $START_TIME"
    echo "⏰ 结束时间: $END_TIME"
    echo "📁 工作目录: $WORK_DIR"
    echo "📝 日志文件: $LOG_FILE"
    
    # 显示最新的检查点
    if [[ -d "$WORK_DIR" ]]; then
        LATEST_CHECKPOINT=$(find "$WORK_DIR" -name "*.pth" -type f -exec ls -t {} + | head -n 1)
        if [[ -n "$LATEST_CHECKPOINT" ]]; then
            print_success "最新检查点: $LATEST_CHECKPOINT"
        fi
    fi
    
    # 显示TensorBoard命令
    if [[ -d "$WORK_DIR" ]]; then
        print_info "查看训练日志:"
        echo "   tensorboard --logdir $WORK_DIR"
    fi
    
else
    print_error "训练失败，退出码: $TRAIN_EXIT_CODE"
    print_info "请查看日志文件: $LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

print_success "脚本执行完成!"