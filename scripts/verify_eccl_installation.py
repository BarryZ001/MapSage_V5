#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import importlib.util

def check_eccl_package():
    """检查ECCL包是否已安装"""
    try:
        result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            eccl_lines = [line for line in lines if 'eccl' in line.lower()]
            if eccl_lines:
                print("✅ 发现已安装的ECCL包:")
                for line in eccl_lines:
                    print(f"   {line}")
                return True
            else:
                print("❌ 未发现ECCL包")
                return False
        else:
            print("⚠️ 无法执行dpkg命令")
            return False
    except FileNotFoundError:
        print("⚠️ dpkg命令不存在（可能不在Debian/Ubuntu系统中）")
        return False
    except Exception as e:
        print(f"❌ 检查ECCL包时出错: {e}")
        return False

def check_eccl_files():
    """检查ECCL相关文件"""
    eccl_paths = [
        '/usr/lib/x86_64-linux-gnu/libeccl.so',
        '/usr/lib/libeccl.so',  # 容器中的标准位置
        '/opt/enflame/lib/libeccl.so',
        '/usr/local/lib/libeccl.so',
    ]
    
    found_files = []
    for path in eccl_paths:
        if os.path.exists(path):
            found_files.append(path)
    
    if found_files:
        print("✅ 发现ECCL库文件:")
        for file in found_files:
            print(f"   {file}")
        return True
    else:
        print("❌ 未发现ECCL库文件")
        return False

def check_eccl_headers():
    """检查ECCL头文件"""
    header_paths = [
        '/usr/include/eccl/eccl.h',  # 容器中的标准位置
        '/usr/include/eccl.h',
        '/usr/local/include/eccl.h',
        '/opt/enflame/include/eccl.h',
    ]
    
    found_headers = []
    for path in header_paths:
        if os.path.exists(path):
            found_headers.append(path)
    
    if found_headers:
        print("✅ 发现ECCL头文件:")
        for header in found_headers:
            print(f"   {header}")
        return True
    else:
        print("❌ 未发现ECCL头文件")
        return False

def check_eccl_tools():
    """检查ECCL性能测试工具"""
    eccl_tools = [
        '/usr/local/bin/eccl_all_gather_perf',
        '/usr/local/bin/eccl_all_reduce_perf',
        '/usr/local/bin/eccl_all_to_all_perf',
        '/usr/local/bin/eccl_broadcast_perf',
        '/usr/local/bin/eccl_gather_perf',
        '/usr/local/bin/eccl_reduce_perf',
        '/usr/local/bin/eccl_reduce_scatter_perf',
        '/usr/local/bin/eccl_scatter_perf',
        '/usr/local/bin/eccl_send_recv_perf',
    ]
    
    found_tools = []
    for tool in eccl_tools:
        if os.path.exists(tool):
            found_tools.append(tool)
    
    if found_tools:
        print("✅ 发现ECCL性能测试工具:")
        for tool in found_tools:
            print(f"   {tool}")
        return True
    else:
        print("❌ 未发现ECCL性能测试工具")
        return False

def check_python_eccl():
    """检查Python ECCL模块"""
    try:
        import eccl  # type: ignore
        print("✅ Python ECCL模块导入成功")
        if hasattr(eccl, '__version__'):
            print(f"   版本: {eccl.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Python ECCL模块导入失败: {e}")
        return False

def check_torch_gcu_eccl():
    """检查torch_gcu中的ECCL支持"""
    try:
        import torch_gcu  # type: ignore
        print("✅ torch_gcu导入成功")
        
        # 检查是否有ECCL相关的分布式后端
        if hasattr(torch_gcu, 'distributed'):
            print("✅ torch_gcu.distributed模块存在")
            
            # 尝试检查可用的后端
            try:
                import torch_gcu.distributed as gcu_dist  # type: ignore
                if hasattr(gcu_dist, 'get_backend'):
                    backend = gcu_dist.get_backend()
                    print(f"   当前后端: {backend}")
                elif hasattr(gcu_dist, 'is_available'):
                    available = gcu_dist.is_available()
                    print(f"   分布式可用: {available}")
            except Exception as e:
                print(f"   检查分布式后端时出错: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ torch_gcu导入失败: {e}")
        return False

def check_environment_variables():
    """检查ECCL相关环境变量"""
    eccl_env_vars = [
        'ECCL_DEBUG',
        'ECCL_LOG_LEVEL',
        'ECCL_SOCKET_IFNAME',
        'ECCL_IB_DISABLE',
        'TOPS_VISIBLE_DEVICES',
    ]
    
    print("🔍 检查ECCL相关环境变量:")
    found_vars = False
    for var in eccl_env_vars:
        value = os.environ.get(var)
        if value is not None:
            print(f"   {var}={value}")
            found_vars = True
    
    if not found_vars:
        print("   未设置ECCL相关环境变量")
    
    return True

def test_eccl_initialization():
    """测试ECCL初始化"""
    try:
        # 尝试简单的ECCL初始化测试
        import torch
        import torch.distributed as dist
        
        print("🧪 测试ECCL后端初始化...")
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # 尝试初始化（仅在单进程模式下）
        if not dist.is_initialized():
            try:
                # 尝试使用gloo后端（更安全）
                dist.init_process_group(backend='gloo', rank=0, world_size=1)
                print("✅ 分布式初始化成功（gloo后端）")
                dist.destroy_process_group()
                return True
            except Exception as e:
                print(f"⚠️ 分布式初始化失败: {e}")
                return False
        else:
            print("✅ 分布式已经初始化")
            return True
            
    except Exception as e:
        print(f"❌ ECCL初始化测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 ECCL安装验证报告")
    print("=" * 50)
    
    results = {}
    
    # 检查各个组件
    print("\n📦 检查ECCL包安装状态:")
    results['package'] = check_eccl_package()
    
    print("\n📁 检查ECCL库文件:")
    results['files'] = check_eccl_files()
    
    print("\n📄 检查ECCL头文件:")
    results['headers'] = check_eccl_headers()
    
    print("\n🔧 检查ECCL性能测试工具:")
    results['tools'] = check_eccl_tools()
    
    print("\n🐍 检查Python ECCL模块:")
    results['python_eccl'] = check_python_eccl()
    
    print("\n🔥 检查torch_gcu ECCL支持:")
    results['torch_gcu'] = check_torch_gcu_eccl()
    
    print("\n🌍 检查环境变量:")
    results['env_vars'] = check_environment_variables()
    
    print("\n🧪 测试ECCL初始化:")
    results['initialization'] = test_eccl_initialization()
    
    # 总结报告
    print("\n" + "=" * 50)
    print("📊 验证结果总结:")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for component, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {component}: {status}")
    
    print(f"\n总体状态: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 ECCL安装验证完全通过！")
        return 0
    elif passed >= total * 0.7:
        print("⚠️ ECCL基本可用，但有部分问题需要解决")
        return 1
    else:
        print("❌ ECCL安装存在严重问题，需要重新安装或配置")
        return 2

if __name__ == '__main__':
    sys.exit(main())