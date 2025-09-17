#!/bin/bash

# T20环境torch-gcu和ptex模块修复脚本
# 此脚本需要在T20服务器的容器内执行

echo "🔧 T20环境torch-gcu和ptex模块修复脚本"
echo "================================================"

# 检查是否在容器内
if [ ! -d "/usr/local/topsrider" ]; then
    echo "❌ 错误: 此脚本必须在T20服务器的容器内执行"
    echo "请先登录T20服务器并进入容器后再运行此脚本"
    exit 1
fi

echo "✅ 检测到T20容器环境"

# 1. 检查torch-gcu框架状态
echo "
🔍 检查torch-gcu框架状态..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('torch.gcu available:', hasattr(torch, 'gcu'))" 2>/dev/null

if ! python3 -c "import torch; assert hasattr(torch, 'gcu')" 2>/dev/null; then
    echo "❌ torch-gcu框架不可用，需要重新安装"
    echo "请手动执行: ./TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu"
else
    echo "✅ torch-gcu框架可用"
fi

# 2. 查找并安装ptex模块
echo "
🔍 查找ptex wheel包..."
PTEX_WHEEL=$(find /usr/local/topsrider -name "ptex*.whl" -type f | head -1)

if [ -z "$PTEX_WHEEL" ]; then
    echo "❌ 未找到ptex wheel包"
    echo "请检查TopsRider软件栈是否完整安装"
    exit 1
fi

echo "✅ 找到ptex wheel包: $PTEX_WHEEL"

# 检查ptex是否已安装
if python3 -c "import ptex" 2>/dev/null; then
    echo "✅ ptex模块已安装"
    python3 -c "import ptex; print('ptex version:', ptex.__version__)"
else
    echo "🔧 安装ptex模块..."
    pip3 install "$PTEX_WHEEL" --force-reinstall
    
    if python3 -c "import ptex" 2>/dev/null; then
        echo "✅ ptex模块安装成功"
        python3 -c "import ptex; print('ptex version:', ptex.__version__)"
    else
        echo "❌ ptex模块安装失败"
        exit 1
    fi
fi

# 3. 验证ptex功能
echo "
🧪 验证ptex功能..."
python3 -c "
import ptex
import torch

print('ptex version:', ptex.__version__)
print('XLA device count:', ptex.device_count())

# 测试设备创建
try:
    device = ptex.device('xla')
    print('XLA device:', device)
    
    # 测试张量操作
    x = torch.randn(2, 3).to(device)
    y = torch.randn(2, 3).to(device)
    z = x + y
    print('✅ 张量操作成功:', z.shape)
    print('✅ 结果设备:', z.device)
except Exception as e:
    print('❌ ptex功能测试失败:', str(e))
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ ptex功能验证成功"
else
    echo "❌ ptex功能验证失败"
    exit 1
fi

# 4. 运行完整环境验证
echo "
🔍 运行完整环境验证..."
cd /workspace/code/MapSage_V5
python3 scripts/validate_training_env.py

echo "
================================================"
echo "🎉 T20环境修复完成！"
echo "如果验证脚本仍显示错误，请检查:"
echo "1. TopsRider软件栈是否完整安装"
echo "2. 是否在正确的容器环境中"
echo "3. 权限设置是否正确"