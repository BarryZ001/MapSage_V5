#!/bin/bash

# T20服务器MMEngine安装脚本
# 适用于燧原T20 GCU环境

set -e

echo "=== T20服务器MMEngine安装脚本 ==="
echo "开始安装MMEngine及其依赖..."

# 检查当前环境
echo "检查当前Python环境..."
python3 --version
pip3 --version

# 更新pip
echo "更新pip..."
python3 -m pip install --upgrade pip

# 设置pip源（使用清华源加速）
echo "配置pip源..."
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
EOF

# 安装基础依赖
echo "安装基础依赖..."
pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    pillow \
    matplotlib \
    tqdm \
    tensorboard \
    pyyaml \
    termcolor \
    yapf

# 安装MMEngine
echo "安装MMEngine..."
pip3 install --no-cache-dir mmengine

# 安装MMCV
echo "安装MMCV..."
pip3 install --no-cache-dir "mmcv>=2.0.0"

# 安装MMSegmentation
echo "安装MMSegmentation..."
pip3 install --no-cache-dir mmsegmentation

# 验证安装
echo "验证MMEngine安装..."
python3 -c "import mmengine; print('✅ MMEngine version:', mmengine.__version__)"

echo "验证MMCV安装..."
python3 -c "import mmcv; print('✅ MMCV version:', mmcv.__version__)"

echo "验证MMSegmentation安装..."
python3 -c "import mmseg; print('✅ MMSegmentation version:', mmseg.__version__)"

# 检查关键模块
echo "检查关键模块..."
python3 -c "from mmengine.config import Config; print('✅ mmengine.config 可用')"
python3 -c "from mmengine.runner import Runner; print('✅ mmengine.runner 可用')"
python3 -c "from mmseg.apis import init_model; print('✅ mmseg.apis 可用')"

# 检查与torch_gcu的兼容性
echo "检查与torch_gcu的兼容性..."
python3 -c "
import torch
import torch_gcu  # type: ignore
import mmengine
print('✅ torch + torch_gcu + mmengine 兼容性正常')
print(f'PyTorch: {torch.__version__}')
print(f'MMEngine: {mmengine.__version__}')
"

echo "=== MMEngine安装完成 ==="
echo "现在可以运行训练脚本了！"

# 提供测试命令
echo ""
echo "可以使用以下命令测试："
echo "python3 -c \"from mmengine.config import Config; print('MMEngine配置模块正常')\""
echo "python3 scripts/diagnose_t20_mmengine.py"