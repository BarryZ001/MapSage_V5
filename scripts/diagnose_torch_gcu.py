#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
燧原T20 torch-gcu诊断脚本
用于深入分析torch-gcu不可用的原因
"""

import os
import sys
import subprocess
import importlib.util

def check_environment_variables():
    """检查环境变量"""
    print("🔍 环境变量检查:")
    
    env_vars = {
        'PATH': '/opt/tops/bin',
        'LD_LIBRARY_PATH': '/opt/tops/lib',
        'PYTHONPATH': '/workspace/code/MapSage_V5'
    }
    
    for var, expected_path in env_vars.items():
        value = os.environ.get(var, '')
        if expected_path in value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: {value} (缺少 {expected_path})")

def check_tops_installation():
    """检查TOPS软件栈安装"""
    print("\n🔍 TOPS软件栈检查:")
    
    # 检查关键文件和目录
    paths_to_check = [
        '/opt/tops',
        '/opt/tops/bin',
        '/opt/tops/lib',
        '/opt/tops/bin/tops-smi',
        '/opt/tops/lib/libtops.so'
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"  ✅ {path} 存在")
        else:
            print(f"  ❌ {path} 不存在")
    
    # 检查tops-smi命令
    try:
        result = subprocess.run(['tops-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ tops-smi 命令可用")
            print(f"    输出: {result.stdout.strip()[:100]}...")
        else:
            print(f"  ❌ tops-smi 命令失败: {result.stderr}")
    except Exception as e:
        print(f"  ❌ tops-smi 命令异常: {e}")

def check_pytorch_installation():
    """检查PyTorch安装"""
    print("\n🔍 PyTorch安装检查:")
    
    try:
        import torch
        print(f"  ✅ PyTorch版本: {torch.__version__}")
        print(f"  📍 PyTorch路径: {torch.__file__}")
        
        # 检查torch模块的属性
        torch_attrs = dir(torch)
        gcu_related = [attr for attr in torch_attrs if 'gcu' in attr.lower()]
        if gcu_related:
            print(f"  🔍 torch中的GCU相关属性: {gcu_related}")
        else:
            print("  ❌ torch中未找到GCU相关属性")
        
        # 尝试导入torch.gcu
        try:
            torch_gcu = importlib.import_module('torch.gcu')
            print("  ✅ torch.gcu模块可导入")
            print(f"  📍 torch.gcu路径: {torch_gcu.__file__}")
        except ImportError as e:
            print(f"  ❌ torch.gcu模块导入失败: {e}")
        
        # 检查hasattr
        has_gcu = hasattr(torch, 'gcu')
        print(f"  🔍 hasattr(torch, 'gcu'): {has_gcu}")
        
    except ImportError as e:
        print(f"  ❌ PyTorch导入失败: {e}")

def check_ptex_installation():
    """检查ptex安装"""
    print("\n🔍 ptex安装检查:")
    
    try:
        ptex = importlib.import_module('ptex')
        print("  ✅ ptex模块可导入")
        print(f"  📍 ptex路径: {ptex.__file__}")
        
        # 检查ptex设备
        try:
            device = ptex.device('xla')
            print(f"  ✅ XLA设备可用: {device}")
        except Exception as e:
            print(f"  ❌ XLA设备创建失败: {e}")
            
    except ImportError as e:
        print(f"  ❌ ptex导入失败: {e}")

def check_library_dependencies():
    """按照官方手册检查共享库依赖"""
    print("\n🔍 共享库依赖检查（按照官方手册指导）:")
    
    # 检查关键的共享库
    libs_to_check = [
        '/opt/tops/lib/libtops.so',
        '/opt/tops/lib/libtorch_gcu.so',
        '/opt/tops/lib/libptex.so',
        '/usr/local/topsrider/lib/libtorch_gcu.so',
        '/usr/local/topsrider/ai_development_toolkit/pytorch-gcu/lib/libtorch_gcu.so'
    ]
    
    for lib in libs_to_check:
        if os.path.exists(lib):
            print(f"  ✅ {lib} 存在")
            # 按照官方手册使用ldd检查共享库依赖项
            try:
                result = subprocess.run(['ldd', lib], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    missing_deps = [line for line in result.stdout.split('\n') if 'not found' in line]
                    if missing_deps:
                        print(f"    ❌ 缺少依赖项:")
                        for dep in missing_deps:
                            print(f"      {dep.strip()}")
                        print("    💡 建议运行 'ldconfig' 命令更新动态链接器配置")
                    else:
                        print(f"    ✅ 依赖项完整")
                else:
                    print(f"    ❌ ldd命令执行失败: {result.stderr}")
            except Exception as e:
                print(f"    ❌ 依赖检查失败: {e}")
        else:
            print(f"  ❌ {lib} 不存在")
    
    # 检查ldconfig配置
    print("\n🔍 动态链接器配置检查:")
    try:
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=10)
        if 'libtops' in result.stdout:
            print("  ✅ libtops 已在动态链接器缓存中")
        else:
            print("  ❌ libtops 未在动态链接器缓存中")
            print("  💡 建议运行 'ldconfig' 命令更新配置")
    except Exception as e:
        print(f"  ❌ 检查动态链接器配置失败: {e}")

def suggest_solutions():
    """建议解决方案"""
    print("\n💡 可能的解决方案:")
    print("1. 重新安装TopsRider软件栈")
    print("2. 检查容器镜像是否包含完整的torch-gcu支持")
    print("3. 验证环境变量设置")
    print("4. 重启容器以重新加载环境")
    print("5. 联系T20环境管理员")

def main():
    print("🔧 燧原T20 torch-gcu深度诊断")
    print("=" * 50)
    
    check_environment_variables()
    check_tops_installation()
    check_pytorch_installation()
    check_ptex_installation()
    check_library_dependencies()
    suggest_solutions()
    
    print("\n" + "=" * 50)
    print("🎯 诊断完成")

if __name__ == '__main__':
    main()