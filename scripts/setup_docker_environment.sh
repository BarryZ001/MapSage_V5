#!/bin/bash
# T20服务器Docker环境配置脚本
# 用于在Docker容器中安装PyTorch和CUDA环境

set -e  # 遇到错误立即退出

echo "🐳 T20服务器Docker环境配置开始"
echo "================================================"

# 检查是否在Docker容器中
if [ ! -f /.dockerenv ]; then
    echo "⚠️  警告: 此脚本应在Docker容器中运行"
fi

# 更新系统包
echo "📦 更新系统包..."
apt-get update
apt-get install -y wget curl git vim

# 检查Python版本
echo "🐍 检查Python版本..."
python3 --version
pip3 --version

# 升级pip
echo "⬆️  升级pip..."
pip3 install --upgrade pip

# 检查CUDA版本
echo "🔧 检查CUDA版本..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ CUDA版本: $CUDA_VERSION"
else
    echo "❌ CUDA未安装或不可用"
    exit 1
fi

# 安装PyTorch (CUDA 11.7版本)
echo "🔥 安装PyTorch (CUDA 11.7)..."
pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# 验证PyTorch安装
echo "✅ 验证PyTorch安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 安装MMSegmentation依赖
echo "🛠️  安装MMSegmentation依赖..."
pip3 install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
pip3 install mmsegmentation==0.30.0

# 安装其他必要依赖
echo "📚 安装其他依赖..."
pip3 install opencv-python pillow numpy scipy matplotlib seaborn
pip3 install transformers timm einops
pip3 install tensorboard wandb

# 验证MMSegmentation安装
echo "✅ 验证MMSegmentation安装..."
python3 -c "import mmseg; print(f'MMSegmentation版本: {mmseg.__version__}')"

# 创建必要目录
echo "📁 创建工作目录..."
mkdir -p /workspace/data/mmrs1m/data
mkdir -p /weights/pretrained/dinov3
mkdir -p /workspace/code/MapSage_V5/work_dirs

# 设置权限
echo "🔐 设置目录权限..."
chmod -R 755 /workspace
chmod -R 755 /weights

# 验证环境
echo "🔍 最终环境验证..."
echo "Python版本: $(python3 --version)"
echo "PyTorch版本: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA可用性: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU数量: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "MMSegmentation版本: $(python3 -c 'import mmseg; print(mmseg.__version__)')"

echo "================================================"
echo "✅ T20服务器Docker环境配置完成！"
echo "📝 接下来请:"
echo "   1. 确保数据已挂载到 /workspace/data/mmrs1m/data"
echo "   2. 确保预训练权重已放置到 /weights/pretrained/dinov3/"
echo "   3. 运行环境验证: python3 scripts/validate_training_env.py"
echo "   4. 开始训练: python3 tools/train.py configs/train_dinov3_mmrs1m.py"
echo "================================================"