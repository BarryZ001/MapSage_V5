#!/bin/bash

# T20环境OpenCV OpenGL依赖修复脚本
# 解决libGL.so.1缺失导致的MMCV导入错误

set -e

echo "🔧 T20环境OpenCV OpenGL依赖修复开始"
echo "================================================"

# 检查当前错误
echo "📋 检查当前MMCV导入错误..."
python3 -c "import mmcv; print('MMCV导入成功')" 2>&1 || echo "❌ 确认存在MMCV导入错误"

# 更新包管理器
echo "📦 更新包管理器..."
apt-get update

# 安装OpenGL相关依赖
echo "🎨 安装OpenGL相关依赖..."
apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0

# 安装额外的图形库依赖
echo "🖼️ 安装图形库依赖..."
apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libxrender1

# 检查libGL.so.1是否存在
echo "🔍 检查OpenGL库..."
ldconfig -p | grep libGL || echo "⚠️ libGL库检查完成"

# 创建软链接（如果需要）
if [ ! -f "/usr/lib/x86_64-linux-gnu/libGL.so.1" ]; then
    echo "🔗 创建libGL软链接..."
    if [ -f "/usr/lib/x86_64-linux-gnu/libGL.so" ]; then
        ln -sf /usr/lib/x86_64-linux-gnu/libGL.so /usr/lib/x86_64-linux-gnu/libGL.so.1
    elif [ -f "/usr/lib/x86_64-linux-gnu/mesa/libGL.so.1" ]; then
        ln -sf /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so.1
    fi
fi

# 重新安装opencv-python（headless版本，避免GUI依赖）
echo "📷 重新安装opencv-python（headless版本）..."
pip3 uninstall -y opencv-python opencv-contrib-python 2>/dev/null || echo "原opencv包已卸载"
pip3 install --no-deps opencv-python-headless==4.8.1.78

# 验证修复结果
echo "🧪 验证修复结果..."
echo "测试OpenCV导入:"
python3 -c "import cv2; print('✅ OpenCV导入成功, 版本:', cv2.__version__)" || echo "❌ OpenCV导入失败"

echo "测试MMCV导入:"
python3 -c "import mmcv; print('✅ MMCV导入成功, 版本:', mmcv.__version__)" || echo "❌ MMCV导入失败"

echo "测试MMEngine导入:"
python3 -c "import mmengine; print('✅ MMEngine导入成功, 版本:', mmengine.__version__)" || echo "❌ MMEngine导入失败"

# 测试基本图像操作（无GUI）
echo "测试基本图像操作:"
python3 -c "
import cv2
import numpy as np
# 创建测试图像
img = np.zeros((100, 100, 3), dtype=np.uint8)
# 基本操作测试
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('✅ OpenCV基本操作测试成功')
" || echo "❌ OpenCV基本操作测试失败"

echo "================================================"
echo "🎉 OpenCV OpenGL依赖修复完成！"
echo "💡 已安装headless版本的OpenCV，避免GUI依赖问题"
echo "💡 如果仍有问题，请检查容器的图形库配置"