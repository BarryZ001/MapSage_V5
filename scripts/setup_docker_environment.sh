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
echo "⬆️ 检查pip版本..."
current_pip_version=$(pip3 --version | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
echo "当前pip版本: $current_pip_version"

# 检查是否存在版本冲突的包
if pip3 list | grep -E "torch-gcu|horovod" | grep -E "gcu-|115\.gcu"; then
    echo "⚠️ 检测到GCU相关包的版本格式问题，降级pip到兼容版本..."
    python3 -m pip install --force-reinstall pip==21.3.1
else
    echo "✅ 升级pip到最新版本..."
    pip3 install --upgrade pip
fi

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
echo "🔧 检查现有PyTorch-GCU环境..."
# 检查是否已安装torch-gcu
if python3 -c "import torch; print('torch version:', torch.__version__)" 2>/dev/null; then
    echo "✅ PyTorch已安装"
    # 检查是否为GCU版本
    if python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "gcu\|GCU"; then
        echo "✅ 检测到GCU版本的PyTorch"
    else
        echo "⚠️ 检测到标准版本PyTorch，在GCU环境中可能需要特殊处理"
    fi
else
    echo "🔥 安装PyTorch (CPU版本，适配T20 GCU环境)..."
    # 在GCU环境中，优先使用已有的torch-gcu
    if python3 -c "import sys; sys.path.append('/usr/local/topsrider'); import torch" 2>/dev/null; then
        echo "✅ 使用系统预装的torch-gcu"
    else
        pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu || {
            echo "⚠️ 使用清华源安装PyTorch..."
            pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
        }
    fi
fi

# 验证PyTorch安装
echo "✅ 验证PyTorch安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'设备类型: {torch.device('cpu')}'); print('✅ PyTorch CPU版本安装成功')"

# 安装MMSegmentation依赖（CPU版本）
echo "🛠️  安装MMSegmentation依赖..."
pip3 install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
pip3 install mmsegmentation==0.30.0

# 安装其他必要依赖
echo "📚 安装其他依赖..."
# 检查pip版本，如果是旧版本则使用兼容模式
current_pip_major=$(pip3 --version | grep -o '[0-9]\+' | head -1)
if [ "$current_pip_major" -lt "22" ]; then
    echo "使用pip兼容模式安装依赖..."
    # 使用--no-deps避免版本冲突
    pip3 install --no-deps numpy==1.21.6 || echo "⚠️ numpy安装失败"
    pip3 install --no-deps scipy==1.7.3 || echo "⚠️ scipy安装失败"
    pip3 install --no-deps matplotlib==3.5.3 || echo "⚠️ matplotlib安装失败"
    pip3 install --no-deps pillow==9.5.0 || echo "⚠️ pillow安装失败"
    pip3 install --no-deps opencv-python==4.8.1.78 || echo "⚠️ opencv-python安装失败"
    pip3 install --no-deps seaborn==0.12.2 || echo "⚠️ seaborn安装失败"
    pip3 install --no-deps transformers==4.21.3 || echo "⚠️ transformers安装失败"
    pip3 install --no-deps timm==0.6.12 || echo "⚠️ timm安装失败"
    pip3 install --no-deps einops==0.6.1 || echo "⚠️ einops安装失败"
else
    echo "使用标准模式安装依赖..."
    pip3 install numpy scipy matplotlib pillow opencv-python seaborn transformers timm einops || {
        echo "⚠️ 标准安装失败，切换到兼容模式..."
        pip3 install --no-deps numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3 pillow==9.5.0 opencv-python==4.8.1.78 seaborn==0.12.2 transformers==4.21.3 timm==0.6.12 einops==0.6.1
    }
fi

# 安装监控和日志工具
echo "安装监控工具..."
pip3 install tensorboard wandb || {
    echo "⚠️ 使用清华源安装监控工具..."
    pip3 install tensorboard wandb -i https://pypi.tuna.tsinghua.edu.cn/simple/
}

# 单独处理可能有版本冲突的包
echo "处理特殊依赖包..."
pip3 install --no-deps mmcv-full==1.7.1 || echo "⚠️ mmcv-full安装跳过"
pip3 install --no-deps mmengine || echo "⚠️ mmengine安装跳过"

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
echo "pip版本: $(pip3 --version)"

echo "PyTorch版本和设备支持:"
python3 -c "try:
    import torch
    print('PyTorch版本:', torch.__version__)
    print('CPU支持: 可用')
    # 检查是否为GCU版本
    if 'gcu' in torch.__version__.lower():
        print('✅ 检测到GCU版本PyTorch')
    else:
        print('⚠️ 标准版本PyTorch')
except Exception as e:
    print('❌ PyTorch验证失败:', str(e))
"

echo "GCU环境验证:"
python3 -c "try:
    import ptex
    print('✅ ptex可用, XLA设备数量:', ptex.device_count())
    print('✅ GCU环境就绪')
except ImportError:
    print('⚠️ ptex不可用 - 可能需要正确配置TopsRider环境')
    print('💡 提示: 请确保已正确安装TopsRider软件栈')
except Exception as e:
    print('⚠️ GCU环境检测异常:', str(e))
"

echo "核心依赖验证:"
for package in numpy scipy matplotlib pillow opencv-python; do
    python3 -c "try: import \${package//-/_}; print('✅ $package: 可用')" 2>/dev/null || echo "❌ $package: 不可用"
done

echo "AI/ML依赖验证:"
for package in transformers timm einops; do
    python3 -c "try: import $package; print('✅ $package: 可用')" 2>/dev/null || echo "❌ $package: 不可用"
done

echo "监控工具验证:"
for package in tensorboard wandb; do
    python3 -c "try: import $package; print('✅ $package: 可用')" 2>/dev/null || echo "❌ $package: 不可用"
done

echo "计算设备: CPU (适配燧原T20 GCU环境)"
echo "MMSegmentation版本: $(python3 -c 'import mmseg; print(mmseg.__version__)' 2>/dev/null || echo '未安装')"

echo "================================================"
echo "✅ T20服务器Docker环境配置完成！"
echo "🔥 燧原T20 GCU环境特别说明:"
echo "   - 已安装CPU版本PyTorch，适配T20 GCU计算环境"
echo "   - GCU加速将通过燧原专用驱动和运行时实现"
echo "📝 接下来请:"
echo "   1. 确保数据已挂载到 /workspace/data/mmrs1m/data"
echo "   2. 确保预训练权重已放置到 /weights/pretrained/dinov3/"
echo "   3. 运行环境验证: python3 scripts/validate_training_env.py"
echo "   4. 开始训练: python3 scripts/train.py configs/train_dinov3_mmrs1m.py"
echo "================================================"