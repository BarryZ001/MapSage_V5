#!/bin/bash

# 燧原T20 GCU环境pip版本冲突修复脚本
# 解决torch-gcu和horovod包版本号格式不符合pip标准的问题

echo "🔧 T20 GCU环境pip版本冲突修复开始"
echo "================================================"

# 检查当前pip版本
echo "📦 当前pip版本:"
pip3 --version

# 首先尝试卸载有问题的包
echo "🗑️ 卸载有问题的包..."
echo "尝试卸载horovod..."
pip3 uninstall -y horovod 2>/dev/null || echo "horovod未安装或卸载失败"
echo "尝试卸载torch-gcu..."
pip3 uninstall -y torch-gcu 2>/dev/null || echo "torch-gcu未安装或卸载失败"

# 清理pip缓存
echo "🧹 清理pip缓存..."
pip3 cache purge 2>/dev/null || echo "缓存清理完成"

# 降级pip到兼容版本
echo "⬇️ 降级pip到兼容版本..."
python3 -m pip install --force-reinstall --no-deps pip==21.3.1

# 验证pip版本
echo "✅ 验证pip版本:"
pip3 --version

# 检查系统预装的torch环境
echo "🔍 检查系统torch环境..."
if [ -d "/usr/local/topsrider" ]; then
    echo "✅ 发现TopsRider环境"
    export PYTHONPATH="/usr/local/topsrider:$PYTHONPATH"
    echo "已添加TopsRider到Python路径"
else
    echo "⚠️ 未发现TopsRider环境"
fi

# 测试torch和ptex导入
echo "torch导入测试:"
python3 -c "try: import torch; print('✅ torch可用, 版本:', torch.__version__); except Exception as e: print('❌ torch导入失败:', str(e))"

echo "ptex导入测试:"
python3 -c "try: import ptex; print('✅ ptex可用, 设备数量:', ptex.device_count()); except Exception as e: print('❌ ptex导入失败:', str(e))"

# 安装基础科学计算包（使用兼容版本）
echo "🔧 安装基础科学计算包..."
echo "安装numpy..."
pip3 install --no-deps numpy==1.21.6 || echo "❌ numpy安装失败"

echo "安装scipy..."
pip3 install --no-deps scipy==1.7.3 || echo "❌ scipy安装失败"

echo "安装matplotlib..."
pip3 install --no-deps matplotlib==3.5.3 || echo "❌ matplotlib安装失败"

echo "安装pillow..."
pip3 install --no-deps pillow==9.5.0 || echo "❌ pillow安装失败"

echo "安装opencv-python..."
pip3 install --no-deps opencv-python==4.8.1.78 || echo "❌ opencv-python安装失败"

# 安装其他必要包
echo "📚 安装其他必要包..."
pip3 install --no-deps tqdm==4.66.1 || echo "❌ tqdm安装失败"
pip3 install --no-deps seaborn==0.12.2 || echo "❌ seaborn安装失败"
pip3 install --no-deps transformers==4.21.3 || echo "❌ transformers安装失败"
pip3 install --no-deps timm==0.6.12 || echo "❌ timm安装失败"
pip3 install --no-deps einops==0.6.1 || echo "❌ einops安装失败"

# 安装监控工具
echo "📊 安装监控工具..."
pip3 install --no-deps tensorboard || echo "❌ tensorboard安装失败"
pip3 install --no-deps wandb || echo "❌ wandb安装失败"

# 验证安装
echo "🧪 验证修复结果..."
echo "Python包导入测试:"
python3 -c "import numpy; print('✅ numpy:', numpy.__version__)" || echo "❌ numpy导入失败"
python3 -c "import scipy; print('✅ scipy:', scipy.__version__)" || echo "❌ scipy导入失败"
python3 -c "import cv2; print('✅ opencv-python导入成功')" || echo "❌ opencv-python导入失败"
python3 -c "import PIL; print('✅ pillow导入成功')" || echo "❌ pillow导入失败"
python3 -c "import matplotlib; print('✅ matplotlib导入成功')" || echo "❌ matplotlib导入失败"
python3 -c "import transformers; print('✅ transformers导入成功')" || echo "❌ transformers导入失败"
python3 -c "import timm; print('✅ timm导入成功')" || echo "❌ timm导入失败"
python3 -c "import einops; print('✅ einops导入成功')" || echo "❌ einops导入失败"

# 最终GCU环境测试
echo "🎯 最终GCU环境测试:"
python3 -c "try: import torch; print('✅ torch版本:', torch.__version__); except Exception as e: print('❌ torch测试失败:', str(e))"
python3 -c "try: import ptex; print('✅ ptex可用, XLA设备数量:', ptex.device_count()); except Exception as e: print('⚠️ ptex测试结果:', str(e))"

echo "================================================"
echo "🎉 T20 GCU环境修复完成！"
echo "💡 如果torch或ptex仍有问题，请检查TopsRider软件栈是否正确安装"
echo "💡 建议运行: python3 scripts/validate_training_env.py 进行完整验证"
echo "💡 如需重新配置完整环境，请运行: bash scripts/setup_docker_environment.sh"