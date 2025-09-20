#!/bin/bash
# -*- coding: utf-8 -*-

# T20服务器缺失模块安装脚本
# 安装torch_gcu和ptex等缺失模块

set -e

echo "=== T20服务器缺失模块安装脚本 ==="
echo "安装torch_gcu和ptex等缺失模块..."

# 检查当前环境
echo "检查当前Python环境..."
python3 --version
pip3 --version

# 检查torch_gcu
echo "检查torch_gcu模块..."
python3 -c "
try:
    import torch_gcu
    print('torch_gcu already available, version:', torch_gcu.__version__)
except ImportError:
    print('torch_gcu not available - this is expected in T20 environment')
    print('torch_gcu should be provided by TopsRider installation')
except Exception as e:
    print('torch_gcu import error:', e)
"

# 检查ptex
echo "检查ptex模块..."
python3 -c "
try:
    import ptex
    print('ptex already available')
except ImportError:
    print('ptex not available - will attempt to install')
except Exception as e:
    print('ptex import error:', e)
"

# 尝试安装ptex
echo "尝试安装ptex..."
pip3 install --no-cache-dir ptex || echo "Warning: ptex installation failed"

# 检查其他可能缺失的依赖
echo "检查其他依赖..."
python3 -c "
import sys
missing_modules = []

# 检查常用模块
modules_to_check = [
    'numpy', 'torch', 'torchvision', 'PIL', 'cv2', 
    'mmcv', 'mmseg', 'mmengine', 'transformers'
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f'✓ {module} available')
    except ImportError:
        print(f'✗ {module} missing')
        missing_modules.append(module)

if missing_modules:
    print(f'Missing modules: {missing_modules}')
else:
    print('All core modules available')
"

# 安装可能缺失的基础依赖
echo "安装基础依赖..."
pip3 install --no-cache-dir \
    numpy \
    pillow \
    opencv-python-headless \
    tqdm \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas

# 验证关键模块
echo "验证关键模块..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)

import mmcv
print('MMCV version:', mmcv.__version__)

import mmseg
print('MMSeg version:', mmseg.__version__)

import mmengine
print('MMEngine version:', mmengine.__version__)

print('All key modules verified successfully!')
"

echo "=== 缺失模块安装完成 ==="
echo "注意: torch_gcu需要通过TopsRider官方安装包提供"