#!/bin/bash
# -*- coding: utf-8 -*-

# T20服务器MMCV版本修复脚本
# 解决MMCV版本兼容性问题

set -e

echo "=== T20服务器MMCV版本修复脚本 ==="
echo "修复MMCV版本兼容性问题..."

# 检查当前环境
echo "检查当前Python环境..."
python3 --version
pip3 --version

# 检查当前MMCV版本
echo "检查当前MMCV版本..."
python3 -c "import mmcv; print('Current MMCV version:', mmcv.__version__)" || echo "MMCV not installed or import failed"

# 卸载现有的MMCV相关包
echo "卸载现有的MMCV相关包..."
pip3 uninstall -y mmcv mmcv-full mmcv-lite || echo "No existing MMCV packages to uninstall"

# 清理pip缓存
echo "清理pip缓存..."
pip3 cache purge

# 安装兼容的MMCV版本
echo "安装兼容的MMCV版本..."
# 对于mmseg 1.x，需要mmcv>=2.0.0rc4
pip3 install --no-cache-dir mmcv>=2.0.0rc4,<2.1.0

# 验证安装
echo "验证MMCV安装..."
python3 -c "
import mmcv
print('MMCV version:', mmcv.__version__)

# 检查版本兼容性
import mmseg
print('MMSeg version:', mmseg.__version__)

# 测试基本功能
from mmcv import Config
print('MMCV Config import successful')

from mmcv.runner import BaseRunner
print('MMCV BaseRunner import successful')
"

# 检查torch_gcu兼容性
echo "检查torch_gcu兼容性..."
python3 -c "
try:
    import torch_gcu
    print('torch_gcu available')
except ImportError:
    print('torch_gcu not available - this is expected in non-GCU environments')

try:
    import ptex
    print('ptex available')
except ImportError:
    print('ptex not available - installing...')
"

# 尝试安装ptex（如果需要）
echo "尝试安装ptex..."
pip3 install --no-cache-dir ptex || echo "Warning: ptex installation failed, continuing..."

echo "=== MMCV版本修复完成 ==="
echo "请重新运行训练脚本验证修复效果"