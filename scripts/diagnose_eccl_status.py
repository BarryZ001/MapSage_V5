#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECCL状态诊断脚本
用于检查eccl库的安装、配置和可用性
"""

import os
import sys
import subprocess
import importlib
import torch
import torch.distributed as dist

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def check_environment_variables():
    """检查ECCL相关环境变量"""
    print_section("环境变量检查")
    
    eccl_vars = [
        'ECCL_ROOT',
        'ECCL_HOME', 
        'ECCL_PATH',
        'ECCL_LIBRARY_PATH',
        'ECCL_INCLUDE_PATH',
        'TOPS_ECCL_ROOT',
        'TOPS_ECCL_HOME',
        'LD_LIBRARY_PATH',
        'PYTHONPATH',
        'PATH'
    ]
    
    found_vars = {}
    for var in eccl_vars:
        value = os.environ.get(var)
        if value:
            found_vars[var] = value
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 未设置")
    
    return found_vars

def check_eccl_files():
    """检查ECCL相关文件和库"""
    print_section("ECCL文件和库检查")
    
    # 常见的ECCL安装路径
    possible_paths = [
        '/usr/local/eccl',
        '/opt/eccl',
        '/usr/eccl',
        '/home/eccl',
        '/workspace/eccl',
        '/usr/local/lib',
        '/usr/lib',
        '/opt/lib'
    ]
    
    # 检查环境变量中的路径
    eccl_root = os.environ.get('ECCL_ROOT')
    if eccl_root:
        possible_paths.insert(0, eccl_root)
    
    tops_eccl_root = os.environ.get('TOPS_ECCL_ROOT')
    if tops_eccl_root:
        possible_paths.insert(0, tops_eccl_root)
    
    found_files = []
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到路径: {path}")
            
            # 查找ECCL相关文件
            try:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if 'eccl' in file.lower():
                            full_path = os.path.join(root, file)
                            found_files.append(full_path)
                            print(f"  📁 {full_path}")
                    
                    # 限制搜索深度
                    if root.count(os.sep) - path.count(os.sep) >= 3:
                        dirs.clear()
                        
            except PermissionError:
                print(f"  ⚠️ 无权限访问: {path}")
        else:
            print(f"❌ 路径不存在: {path}")
    
    return found_files

def check_library_path():
    """检查库路径中的ECCL库"""
    print_section("库路径检查")
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        paths = ld_library_path.split(':')
        print(f"LD_LIBRARY_PATH包含 {len(paths)} 个路径:")
        
        for path in paths:
            if path and os.path.exists(path):
                print(f"✅ {path}")
                try:
                    files = os.listdir(path)
                    eccl_files = [f for f in files if 'eccl' in f.lower()]
                    if eccl_files:
                        for f in eccl_files:
                            print(f"  📚 {f}")
                except PermissionError:
                    print(f"  ⚠️ 无权限访问")
            else:
                print(f"❌ {path} (不存在)")
    else:
        print("❌ LD_LIBRARY_PATH未设置")

def check_python_imports():
    """检查Python中的ECCL相关导入"""
    print_section("Python导入检查")
    
    # 尝试导入可能的ECCL模块
    eccl_modules = [
        'eccl',
        'torch_eccl',
        'tops_eccl',
        'enflame_eccl',
        'torch_gcu.distributed.eccl',
        'torch_gcu.eccl'
    ]
    
    for module_name in eccl_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}: 导入成功")
            
            # 尝试获取模块信息
            if hasattr(module, '__version__'):
                print(f"  版本: {module.__version__}")
            if hasattr(module, '__file__'):
                print(f"  路径: {module.__file__}")
                
        except ImportError as e:
            print(f"❌ {module_name}: 导入失败 - {e}")
        except Exception as e:
            print(f"⚠️ {module_name}: 导入异常 - {e}")

def check_torch_backends():
    """检查PyTorch支持的分布式后端"""
    print_section("PyTorch分布式后端检查")
    
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查可用的后端
    available_backends = []
    
    backends_to_check = ['gloo', 'nccl', 'mpi', 'eccl']
    
    for backend in backends_to_check:
        try:
            # 使用hasattr检查方法是否存在，然后调用
            if hasattr(dist, 'is_backend_available') and dist.is_backend_available(backend):  # type: ignore
                available_backends.append(backend)
                print(f"✅ {backend}: 可用")
            else:
                print(f"❌ {backend}: 不可用")
        except Exception as e:
            print(f"⚠️ {backend}: 检查失败 - {e}")
    
    return available_backends

def check_torch_gcu_eccl():
    """检查torch_gcu中的ECCL支持"""
    print_section("torch_gcu ECCL支持检查")
    
    try:
        import torch_gcu  # type: ignore
        print(f"✅ torch_gcu版本: {torch_gcu.__version__ if hasattr(torch_gcu, '__version__') else '未知'}")
        
        # 检查torch_gcu.distributed
        try:
            import torch_gcu.distributed as gcu_dist  # type: ignore
            print("✅ torch_gcu.distributed: 导入成功")
            
            # 检查是否有eccl相关属性或方法
            eccl_attrs = [attr for attr in dir(gcu_dist) if 'eccl' in attr.lower()]
            if eccl_attrs:
                print(f"  ECCL相关属性: {eccl_attrs}")
            else:
                print("  ❌ 未找到ECCL相关属性")
                
        except ImportError as e:
            print(f"❌ torch_gcu.distributed: 导入失败 - {e}")
            
    except ImportError as e:
        print(f"❌ torch_gcu: 导入失败 - {e}")

def check_system_commands():
    """检查系统命令和工具"""
    print_section("系统命令检查")
    
    commands = [
        'ldd --version',
        'ldconfig -p | grep eccl',
        'find /usr -name "*eccl*" 2>/dev/null | head -10',
        'pkg-config --list-all | grep eccl',
        'which eccl',
        'eccl --version'
    ]
    
    for cmd in commands:
        try:
            print(f"\n🔧 执行: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                print(f"✅ 输出:\n{result.stdout.strip()}")
            else:
                print(f"❌ 无输出或失败 (返回码: {result.returncode})")
                if result.stderr.strip():
                    print(f"错误: {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ 命令超时")
        except Exception as e:
            print(f"⚠️ 执行失败: {e}")

def test_eccl_initialization():
    """测试ECCL后端初始化"""
    print_section("ECCL初始化测试")
    
    # 设置测试环境
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29501')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
        
        print("🧪 尝试初始化ECCL后端...")
        dist.init_process_group(
            backend='eccl',
            init_method='tcp://127.0.0.1:29501',
            world_size=1,
            rank=0,
            timeout=dist.default_pg_timeout  # type: ignore
        )
        
        print("✅ ECCL后端初始化成功！")
        
        # 清理
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"❌ ECCL后端初始化失败: {e}")
        print(f"错误类型: {type(e).__name__}")

def generate_report():
    """生成诊断报告"""
    print_section("诊断报告生成")
    
    report = {
        'environment_vars': check_environment_variables(),
        'available_backends': check_torch_backends(),
    }
    
    # 生成建议
    print("\n📋 诊断建议:")
    
    if 'eccl' not in report['available_backends']:
        print("❌ ECCL后端不可用，可能的解决方案:")
        print("   1. 检查ECCL库是否正确安装")
        print("   2. 确认环境变量设置正确")
        print("   3. 检查LD_LIBRARY_PATH包含ECCL库路径")
        print("   4. 验证PyTorch版本与ECCL兼容性")
        print("   5. 查看torch_gcu是否支持ECCL")
    else:
        print("✅ ECCL后端可用")
    
    if not report['environment_vars']:
        print("⚠️ 未找到ECCL相关环境变量，建议设置:")
        print("   export ECCL_ROOT=/path/to/eccl")
        print("   export LD_LIBRARY_PATH=$ECCL_ROOT/lib:$LD_LIBRARY_PATH")

def main():
    """主函数"""
    print("🔍 ECCL状态诊断工具")
    print("=" * 60)
    
    try:
        # 执行各项检查
        check_environment_variables()
        check_eccl_files()
        check_library_path()
        check_python_imports()
        check_torch_backends()
        check_torch_gcu_eccl()
        check_system_commands()
        test_eccl_initialization()
        generate_report()
        
        print(f"\n{'='*60}")
        print("🎯 诊断完成！请查看上述结果来确定ECCL状态。")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ 诊断被用户中断")
    except Exception as e:
        print(f"\n❌ 诊断过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()