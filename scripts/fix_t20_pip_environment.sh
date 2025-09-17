#!/bin/bash

# 燧原T20 GCU环境pip版本冲突修复脚本
# 解决torch-gcu和horovod包版本号格式不符合pip标准的问题

echo "🔧 T20 GCU环境pip版本冲突修复开始"
echo "================================================"

# 检查当前pip版本
echo "📦 当前pip版本:"
pip3 --version

# 降级pip到兼容版本
echo "⬇️ 降级pip到兼容版本..."
python3 -m pip install --force-reinstall pip==21.3.1

# 验证pip版本
echo "✅ 验证pip版本:"
pip3 --version

# 检查问题包
echo "🔍 检查问题包..."
echo "torch-gcu版本:"
python3 -c "try: import torch; print('torch version:', torch.__version__); except: print('torch未安装')" 2>/dev/null

echo "horovod版本:"
python3 -c "try: import horovod; print('horovod version:', horovod.__version__); except: print('horovod未安装')" 2>/dev/null

echo "ptex版本:"
python3 -c "try: import ptex; print('ptex version:', ptex.__version__); except: print('ptex未安装')" 2>/dev/null

# 尝试修复包依赖
echo "🔧 修复包依赖..."
# 使用--no-deps安装基础包避免版本冲突
echo "安装基础科学计算包..."
pip3 install --no-deps --force-reinstall numpy==1.21.6
pip3 install --no-deps --force-reinstall scipy==1.7.3
pip3 install --no-deps --force-reinstall matplotlib==3.5.3
pip3 install --no-deps --force-reinstall pillow==9.5.0
pip3 install --no-deps --force-reinstall opencv-python==4.8.1.78

# 安装其他必要包
echo "安装其他必要包..."
pip3 install --no-deps tqdm==4.66.1
pip3 install --no-deps seaborn==0.12.2
pip3 install --no-deps transformers==4.21.3
pip3 install --no-deps timm==0.6.12
pip3 install --no-deps einops==0.6.1

# 验证安装
echo "🧪 验证修复结果..."
echo "Python包导入测试:"
python3 -c "import numpy; print('✅ numpy:', numpy.__version__)" || echo "❌ numpy导入失败"
python3 -c "import torch; print('✅ torch:', torch.__version__)" || echo "❌ torch导入失败"
python3 -c "import cv2; print('✅ opencv-python导入成功')" || echo "❌ opencv-python导入失败"
python3 -c "import PIL; print('✅ pillow导入成功')" || echo "❌ pillow导入失败"
python3 -c "import matplotlib; print('✅ matplotlib导入成功')" || echo "❌ matplotlib导入失败"

# GCU环境测试
echo "GCU环境测试:"
python3 -c "try: import ptex; print('✅ ptex可用, XLA设备数量:', ptex.device_count()); except Exception as e: print('⚠️ ptex测试:', str(e))" || echo "❌ GCU环境测试失败"

echo "================================================"
echo "🎉 T20 GCU环境修复完成！"
echo "💡 如果仍有问题，请检查TopsRider软件栈是否正确安装"
echo "💡 建议重启容器后再次运行环境验证脚本"