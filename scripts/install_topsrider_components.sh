#!/bin/bash

# TopsRider组件安装脚本
# 基于已安装的ECCL包信息进行环境配置

set -e  # 遇到错误时退出

echo "🚀 TopsRider组件安装和配置脚本"
echo "=================================="

# 检查是否在容器环境中
if [ -f /.dockerenv ]; then
    echo "✅ 检测到Docker容器环境"
    IN_CONTAINER=true
else
    echo "⚠️ 未检测到容器环境，可能在宿主机上运行"
    IN_CONTAINER=false
fi

# 检查ECCL包是否已安装
check_eccl_installation() {
    echo "🔍 检查ECCL安装状态..."
    
    if command -v dpkg >/dev/null 2>&1; then
        if dpkg -l | grep -q "tops-eccl"; then
            echo "✅ 发现已安装的ECCL包:"
            dpkg -l | grep eccl
            return 0
        else
            echo "❌ 未发现ECCL包"
            return 1
        fi
    else
        echo "⚠️ dpkg命令不可用，跳过包检查"
        return 1
    fi
}

# 验证ECCL文件
verify_eccl_files() {
    echo "📁 验证ECCL文件..."
    
    # 检查库文件
    if [ -f "/usr/lib/libeccl.so" ]; then
        echo "✅ 找到ECCL库文件: /usr/lib/libeccl.so"
        ls -la /usr/lib/libeccl.so
    else
        echo "❌ 未找到ECCL库文件"
        return 1
    fi
    
    # 检查头文件
    if [ -f "/usr/include/eccl/eccl.h" ]; then
        echo "✅ 找到ECCL头文件: /usr/include/eccl/eccl.h"
        ls -la /usr/include/eccl/eccl.h
    else
        echo "❌ 未找到ECCL头文件"
        return 1
    fi
    
    # 检查性能测试工具
    echo "🔧 检查ECCL性能测试工具:"
    local tools_found=0
    for tool in eccl_all_gather_perf eccl_all_reduce_perf eccl_broadcast_perf; do
        if [ -f "/usr/local/bin/$tool" ]; then
            echo "✅ 找到工具: /usr/local/bin/$tool"
            tools_found=$((tools_found + 1))
        fi
    done
    
    if [ $tools_found -gt 0 ]; then
        echo "✅ 找到 $tools_found 个ECCL性能测试工具"
    else
        echo "⚠️ 未找到ECCL性能测试工具"
    fi
    
    return 0
}

# 配置环境变量
configure_environment() {
    echo "🌍 配置ECCL环境变量..."
    
    # 创建环境配置文件
    cat > /tmp/eccl_env.sh << 'EOF'
# ECCL环境配置
export ECCL_DEBUG=0
export ECCL_LOG_LEVEL=INFO
export ECCL_SOCKET_IFNAME=eth0
export ECCL_IB_DISABLE=1

# 库路径配置
export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH}"

# 工具路径配置
export PATH="/usr/local/bin:${PATH}"

# GCU设备配置
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=""

echo "✅ ECCL环境变量已配置"
EOF

    echo "✅ 环境配置文件已创建: /tmp/eccl_env.sh"
    echo "📝 要应用配置，请运行: source /tmp/eccl_env.sh"
}

# 测试ECCL功能
test_eccl_functionality() {
    echo "🧪 测试ECCL功能..."
    
    # 检查库是否可以加载
    if command -v ldd >/dev/null 2>&1; then
        echo "🔍 检查ECCL库依赖:"
        if ldd /usr/lib/libeccl.so 2>/dev/null; then
            echo "✅ ECCL库依赖检查通过"
        else
            echo "⚠️ ECCL库依赖检查失败"
        fi
    fi
    
    # 测试性能工具（如果存在）
    if [ -f "/usr/local/bin/eccl_all_reduce_perf" ]; then
        echo "🔧 测试ECCL性能工具..."
        if /usr/local/bin/eccl_all_reduce_perf --help >/dev/null 2>&1; then
            echo "✅ ECCL性能工具可用"
        else
            echo "⚠️ ECCL性能工具测试失败"
        fi
    fi
}

# 创建Python验证脚本
create_python_verification() {
    echo "🐍 创建Python验证脚本..."
    
    cat > /tmp/verify_eccl_python.py << 'EOF'
#!/usr/bin/env python3
import os
import sys

def test_eccl_import():
    """测试ECCL Python模块导入"""
    try:
        import eccl
        print("✅ ECCL Python模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ ECCL Python模块导入失败: {e}")
        return False

def test_torch_gcu():
    """测试torch_gcu模块"""
    try:
        import torch_gcu
        print("✅ torch_gcu模块导入成功")
        print(f"   可用设备数: {torch_gcu.device_count()}")
        return True
    except ImportError as e:
        print(f"❌ torch_gcu模块导入失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Python模块验证")
    print("=" * 30)
    
    eccl_ok = test_eccl_import()
    torch_gcu_ok = test_torch_gcu()
    
    if eccl_ok or torch_gcu_ok:
        print("✅ 至少一个关键模块可用")
        sys.exit(0)
    else:
        print("❌ 关键模块都不可用")
        sys.exit(1)
EOF

    chmod +x /tmp/verify_eccl_python.py
    echo "✅ Python验证脚本已创建: /tmp/verify_eccl_python.py"
}

# 主安装流程
main() {
    echo "开始TopsRider组件安装和配置..."
    
    # 1. 检查ECCL安装
    if check_eccl_installation; then
        echo "✅ ECCL包检查通过"
    else
        echo "⚠️ ECCL包检查失败，但继续进行文件验证"
    fi
    
    # 2. 验证ECCL文件
    if verify_eccl_files; then
        echo "✅ ECCL文件验证通过"
    else
        echo "❌ ECCL文件验证失败"
        exit 1
    fi
    
    # 3. 配置环境
    configure_environment
    
    # 4. 测试功能
    test_eccl_functionality
    
    # 5. 创建Python验证脚本
    create_python_verification
    
    echo ""
    echo "🎉 TopsRider组件配置完成！"
    echo "=================================="
    echo "📋 后续步骤:"
    echo "1. 应用环境配置: source /tmp/eccl_env.sh"
    echo "2. 运行Python验证: python /tmp/verify_eccl_python.py"
    echo "3. 测试分布式训练: 使用 train_distributed_gcu_robust.py"
    echo ""
    echo "📁 重要文件位置:"
    echo "   - ECCL库: /usr/lib/libeccl.so"
    echo "   - ECCL头文件: /usr/include/eccl/eccl.h"
    echo "   - 性能工具: /usr/local/bin/eccl_*_perf"
    echo "   - 环境配置: /tmp/eccl_env.sh"
}

# 执行主函数
main "$@"