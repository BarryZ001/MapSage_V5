#!/bin/bash

# T20环境torch-gcu和ptex模块修复脚本
# 此脚本需要在T20服务器的容器内执行

echo "🔧 T20环境torch-gcu和ptex模块修复脚本"
echo "================================================"

# 检查是否在容器内
if [ ! -d "/usr/local/topsrider" ]; then
    echo "❌ 错误: 此脚本必须在T20服务器的容器内执行"
    echo "请使用以下命令进入容器:"
    echo "docker exec -it t20_mapsage_env /bin/bash"
    echo "然后再运行此脚本"
    exit 1
fi

echo "✅ 检测到T20容器环境"

# 1. 检查torch-gcu框架状态
echo "
🔍 检查torch-gcu框架状态..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('torch.gcu available:', hasattr(torch, 'gcu'))" 2>/dev/null

if ! python3 -c "import torch; assert hasattr(torch, 'gcu')" 2>/dev/null; then
    echo "❌ torch-gcu框架不可用，需要重新安装TopsRider软件栈"
    echo "🔧 开始重新安装TopsRider软件栈..."
    
    # 查找TopsRider安装程序（多个位置查找）
    TOPSRIDER_INSTALLER=$(find /root -name "TopsRider*.run" -type f 2>/dev/null | head -1)
    
    if [ -z "$TOPSRIDER_INSTALLER" ]; then
        # 查找/usr/local/topsrider目录
        TOPSRIDER_INSTALLER=$(find /usr/local/topsrider -name "TopsRider*.run" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "$TOPSRIDER_INSTALLER" ]; then
        # 查找当前工作目录（可能从主机拷贝到这里）
        TOPSRIDER_INSTALLER=$(find /workspace/code/MapSage_V5 -name "TopsRider*.run" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "$TOPSRIDER_INSTALLER" ]; then
        echo "❌ 未找到TopsRider安装程序"
        echo "请从主机拷贝TopsRider安装文件到容器中:"
        echo "方法1: docker cp /root/TopsRider_t2x_2.5.136_deb_amd64.run t20_mapsage_env:/root/"
        echo "方法2: cp /root/TopsRider_t2x_2.5.136_deb_amd64.run /root/mapsage_project/code/"
        echo "然后重新运行此脚本"
        exit 1
    fi
    
    echo "✅ 找到TopsRider安装程序: $TOPSRIDER_INSTALLER"
    echo # 按照官方手册分两步安装TopsRider
    echo "🔧 第一步：安装基础软件栈..."
    chmod +x "$TOPSRIDER_INSTALLER"
    "$TOPSRIDER_INSTALLER" -y
    
    echo "🔧 第二步：安装torch-gcu框架..."
    "$TOPSRIDER_INSTALLER" -y -C torch-gcu
    
    # 设置环境变量
    echo "🔧 设置环境变量..."
    export PATH="/opt/tops/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/tops/lib:$LD_LIBRARY_PATH"
    
    # 添加到bashrc以持久化
    echo 'export PATH="/opt/tops/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="/opt/tops/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
    
    # 重新加载环境
    source ~/.bashrc
    
    # 验证安装结果
    echo "🔍 重新验证torch-gcu状态..."
    if python3 -c "import torch; print('torch.gcu available:', torch.gcu.is_available())" 2>/dev/null | grep -q "True"; then
        echo "✅ torch-gcu框架重新安装成功"
    else
        echo "⚠️  torch-gcu框架安装完成，但可能需要重启容器或重新登录"
        echo "请尝试以下步骤:"
        echo "1. 退出容器: exit"
        echo "2. 重新进入容器: docker exec -it t20_mapsage_env /bin/bash"
        echo "3. 重新运行验证: python3 -c 'import torch; print(torch.gcu.is_available())'"
        echo "或手动执行: $TOPSRIDER_INSTALLER -y -C torch-gcu"
    fi
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
    python3 -c "import ptex; print('ptex模块导入成功')"
else
    echo "🔧 安装ptex模块..."
    pip3 install "$PTEX_WHEEL" --force-reinstall
    
    if python3 -c "import ptex" 2>/dev/null; then
        echo "✅ ptex模块安装成功"
        python3 -c "import ptex; print('ptex模块导入成功')"
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

print('ptex模块导入成功')
print('ptex可用函数:', [attr for attr in dir(ptex) if not attr.startswith('_')][:5])

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

# 4. 修复QuantStub兼容性问题
echo "
🔧 修复QuantStub兼容性问题..."
cd /workspace/code/MapSage_V5
python3 scripts/fix_quantstub_compatibility.py

echo "
🔍 诊断torch-gcu问题..."
python3 scripts/diagnose_torch_gcu.py

# 5. 运行完整环境验证
echo "
🔍 运行完整环境验证..."
validation_result=$(python3 scripts/validate_training_env.py 2>&1)
echo "$validation_result"

echo "================================================"
echo "🎉 T20环境修复完成！"
echo "
# 检查验证结果中是否包含torch-gcu不可用的信息
if echo "$validation_result" | grep -q "torch-gcu框架不可用"; then
    echo "⚠️  torch-gcu框架仍不可用，需要重启容器"
    echo "请运行重启指导脚本:"
    echo "bash scripts/restart_container_guide.sh"
else
    echo "✅ 环境修复成功，可以开始训练！"
fi

echo ""
echo "💡 如果验证脚本仍显示错误，请检查:"
echo "1. TopsRider软件栈是否完整安装"
echo "2. 是否在正确的容器环境中"
echo "3. 权限设置是否正确"
echo "4. 是否需要重启容器让torch-gcu生效"