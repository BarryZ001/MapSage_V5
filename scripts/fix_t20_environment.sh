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
    echo "🔧 按照官方手册执行两步安装流程..."
    chmod +x "$TOPSRIDER_INSTALLER"
    
    echo "📋 第一步：安装除框架之外的基础软件栈"
    echo "   执行命令: $TOPSRIDER_INSTALLER -y"
    "$TOPSRIDER_INSTALLER" -y
    
    if [ $? -ne 0 ]; then
        echo "❌ 基础软件栈安装失败"
        exit 1
    fi
    
    echo "📋 第二步：安装torch-gcu框架"
    echo "   执行命令: $TOPSRIDER_INSTALLER -y -C torch-gcu"
    "$TOPSRIDER_INSTALLER" -y -C torch-gcu
    
    if [ $? -ne 0 ]; then
        echo "❌ torch-gcu框架安装失败"
        exit 1
    fi
    
    # 按照官方手册设置环境变量和动态链接器
    echo "🔧 设置环境变量和动态链接器配置..."
    export PATH="/opt/tops/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/tops/lib:$LD_LIBRARY_PATH"
    
    # 添加到bashrc以持久化
    echo 'export PATH="/opt/tops/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="/opt/tops/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
    
    # 按照官方手册运行ldconfig更新动态链接器配置
    echo "🔧 更新动态链接器配置..."
    ldconfig
    
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

# 2. 按照官方手册和成功经验安装ptex模块
echo "
🔧 安装ptex模块（torch-gcu框架核心组件）..."

# 查找ptex wheel包（按照成功经验的路径）
PTEX_WHEEL=$(find /usr/local/topsrider -name "ptex*.whl" -type f 2>/dev/null | head -1)

if [ -z "$PTEX_WHEEL" ]; then
    echo "❌ 未找到ptex wheel包"
    echo "请检查TopsRider软件栈是否完整安装"
    echo "预期路径: /usr/local/topsrider/ai_development_toolkit/pytorch-gcu/ptex-*.whl"
    exit 1
fi

echo "✅ 找到ptex wheel包: $PTEX_WHEEL"

# 检查ptex是否已安装
if python3 -c "import ptex" 2>/dev/null; then
    echo "✅ ptex模块已安装"
    python3 -c "import ptex; print('ptex模块导入成功')"
else
    echo "📋 安装ptex模块（负责XLA设备管理和张量操作）"
    pip3 install "$PTEX_WHEEL" --force-reinstall
    
    if python3 -c "import ptex" 2>/dev/null; then
        echo "✅ ptex模块安装成功"
        
        # 按照成功经验进行基础验证
        echo "🔍 验证ptex模块安装..."
        python3 -c "import ptex; print('ptex version:', ptex.__version__); print('XLA devices:', ptex.device_count())" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "✅ ptex模块验证成功"
        else
            echo "⚠️  ptex模块导入成功但功能验证失败（可能需要重启容器）"
        fi
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

# 检查是否需要修复TOPS软件栈
echo "
🔧 检查TOPS软件栈状态..."
if [ ! -f "/opt/tops/bin/tops-smi" ] || [ ! -f "/opt/tops/lib/libtops.so" ]; then
    echo "⚠️ 检测到TOPS软件栈核心文件缺失"
    echo "是否需要尝试修复TOPS软件栈？(需要root权限)"
    echo "输入 'y' 继续修复，或按任意键跳过:"
    read -t 10 -n 1 response
    echo
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "🔧 开始修复TOPS软件栈..."
        sudo python3 scripts/fix_tops_stack.py
        if [ $? -eq 0 ]; then
            echo "✅ TOPS软件栈修复完成"
            # 重新加载环境变量
            source /etc/profile 2>/dev/null || true
        else
            echo "❌ TOPS软件栈修复失败，请手动检查"
        fi
    else
        echo "⏭️ 跳过TOPS软件栈修复"
    fi
else
    echo "✅ TOPS软件栈核心文件存在"
fi

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