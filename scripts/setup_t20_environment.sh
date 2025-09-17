#!/bin/bash

# T20服务器环境配置脚本
# 安装MapSage V5训练所需的所有Python依赖

echo "🔧 开始配置T20服务器训练环境..."

# 检查Python版本
echo "📋 检查Python版本..."
python --version
python3 --version

# 升级pip
echo "⬆️ 升级pip..."
python -m pip install --upgrade pip

# 安装PyTorch (适配T20/燧原芯片)
echo "🔥 安装PyTorch..."
# 根据T20服务器的具体配置选择合适的PyTorch版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装MMEngine和相关依赖
echo "⚙️ 安装MMEngine生态系统..."
pip install mmengine
pip install mmcv>=2.0.0
pip install mmsegmentation

# 安装其他核心依赖
echo "📦 安装核心依赖包..."
pip install numpy
pip install opencv-python
pip install pillow
pip install matplotlib
pip install scipy
pip install scikit-learn
pip install tqdm
pip install tensorboard
pip install wandb

# 安装数据处理相关依赖
echo "📊 安装数据处理依赖..."
pip install pandas
pip install h5py
pip install imageio
pip install albumentations

# 安装配置文件处理
echo "⚙️ 安装配置处理依赖..."
pip install pyyaml
pip install addict

# 验证安装
echo "✅ 验证关键包安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import mmengine; print(f'MMEngine版本: {mmengine.__version__}')"
python -c "import mmcv; print(f'MMCV版本: {mmcv.__version__}')"
python -c "import mmseg; print(f'MMSegmentation版本: {mmseg.__version__}')"

# 检查CUDA可用性（如果适用）
echo "🔍 检查计算设备..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'设备数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# 创建必要的目录
echo "📁 创建工作目录..."
mkdir -p checkpoints
mkdir -p work_dirs
mkdir -p data
mkdir -p logs

echo "🎉 T20服务器环境配置完成！"
echo "💡 下一步操作:"
echo "   1. 验证环境: python scripts/validate_training_env.py"
echo "   2. 开始训练: bash scripts/train_dinov3_mmrs1m.sh"
echo "="*60