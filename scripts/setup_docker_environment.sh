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

# 检查GCU版本（燧原T20专用）
echo "🔧 检查GCU版本..."
if command -v gcu-smi &> /dev/null; then
    gcu-smi
    echo "✅ GCU环境可用"
elif [ -d "/usr/local/gcu" ]; then
    echo "✅ GCU环境已安装"
else
    echo "⚠️ GCU环境未检测到，但继续安装（T20服务器可能使用特殊配置）"
fi

# 安装PyTorch (CPU版本，适配燧原T20 GCU环境)
echo "🔥 安装PyTorch (CPU版本，适配T20 GCU环境)..."
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 验证PyTorch安装
echo "✅ 验证PyTorch安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'设备类型: {torch.device('cpu')}'); print('✅ PyTorch CPU版本安装成功')"

# 安装MMSegmentation依赖（CPU版本）
echo "🛠️  安装MMSegmentation依赖..."
pip3 install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
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
echo "计算设备: CPU (适配燧原T20 GCU环境)"
echo "MMSegmentation版本: $(python3 -c 'import mmseg; print(mmseg.__version__)')"

echo "================================================"
echo "✅ T20服务器Docker环境配置完成！"
echo "🔥 燧原T20 GCU环境特别说明:"
echo "   - 已安装CPU版本PyTorch，适配T20 GCU计算环境"
echo "   - GCU加速将通过燧原专用驱动和运行时实现"
echo "📝 接下来请:"
echo "   1. 确保数据已挂载到 /workspace/data/mmrs1m/data"
echo "   2. 确保预训练权重已放置到 /weights/pretrained/dinov3/"
echo "   3. 运行环境验证: python3 scripts/validate_training_env.py"
echo "   4. 开始训练: python3 tools/train.py configs/train_dinov3_mmrs1m.py"
echo "================================================"