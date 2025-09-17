#!/bin/bash

# 燧原T20 DINOv3 + MMRS-1M 训练启动脚本
# 基于之前成功的训练经验

echo "🚀 燧原T20 DINOv3 + MMRS-1M 训练启动"
echo "================================================"

# 检查是否在T20容器环境中
if [ ! -f "/opt/tops/bin/tops-smi" ] && [ ! -d "/opt/tops" ]; then
    echo "❌ 未检测到T20环境，请确保在正确的容器中运行"
    exit 1
fi

echo "✅ 检测到T20容器环境"

# 设置环境变量
export PATH="/opt/tops/bin:$PATH"
export LD_LIBRARY_PATH="/opt/tops/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/workspace/code/MapSage_V5:$PYTHONPATH"

# 进入项目目录
cd /workspace/code/MapSage_V5

echo "\n🔍 快速环境检查..."

# 检查torch-gcu
python3 -c "
try:
    import torch
    if hasattr(torch, 'gcu'):
        print('✅ torch-gcu框架可用')
    else:
        print('❌ torch-gcu框架不可用，请重启容器')
        exit(1)
except Exception as e:
    print(f'❌ PyTorch检查失败: {e}')
    exit(1)
" || exit 1

# 检查ptex
python3 -c "
try:
    import ptex
    device = ptex.device('xla')
    print('✅ ptex模块可用')
except Exception as e:
    print(f'❌ ptex检查失败: {e}')
    exit(1)
" || exit 1

# 检查配置文件
if [ ! -f "configs/train_dinov3_mmrs1m.py" ]; then
    echo "❌ 训练配置文件不存在: configs/train_dinov3_mmrs1m.py"
    exit 1
fi
echo "✅ 训练配置文件存在"

# 检查数据路径
if [ ! -d "/workspace/data/mmrs1m/data" ]; then
    echo "❌ 数据路径不存在: /workspace/data/mmrs1m/data"
    exit 1
fi
echo "✅ 数据路径存在"

# 检查预训练权重
if [ ! -f "/workspace/weights/pretrained/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth" ]; then
    echo "❌ 预训练权重不存在"
    exit 1
fi
echo "✅ 预训练权重存在"

# 创建工作目录
mkdir -p work_dirs/dinov3_mmrs1m_stage1
echo "✅ 工作目录已创建"

echo "\n🎯 开始训练..."
echo "配置文件: configs/train_dinov3_mmrs1m.py"
echo "工作目录: work_dirs/dinov3_mmrs1m_stage1"
echo "最大迭代: 80000"
echo "批次大小: 8 x 2 = 16"

echo "\n⏰ 预计训练时间: 5-7天"
echo "📊 可以通过以下命令监控训练进度:"
echo "   tail -f work_dirs/dinov3_mmrs1m_stage1/$(date +%Y%m%d_%H%M%S).log"
echo "   tops-smi (查看GPU使用情况)"

echo "\n🚀 启动训练..."

# 启动训练
python3 tools/train.py configs/train_dinov3_mmrs1m.py \
    --work-dir work_dirs/dinov3_mmrs1m_stage1 \
    --seed 42 \
    --deterministic