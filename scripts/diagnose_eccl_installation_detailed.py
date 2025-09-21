#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECCL安装详细诊断脚本
检查ECCL库的安装状态、路径配置和Python模块可用性
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def run_command(cmd, capture_output=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_eccl_package():
    """检查ECCL系统包安装状态"""
    print("=== 检查ECCL系统包安装状态 ===")
    
    # 检查dpkg包状态
    ret, stdout, stderr = run_command("dpkg -l | grep eccl")
    if ret == 0 and stdout.strip():
        print("✓ ECCL系统包已安装:")
        print(stdout.strip())
    else:
        print("✗ ECCL系统包未找到")
        return False
    
    # 检查包文件列表
    ret, stdout, stderr = run_command("dpkg -L tops-eccl")
    if ret == 0:
        print("\n✓ ECCL包文件列表:")
        files = stdout.strip().split('\n')
        for file in files[:20]:  # 只显示前20个文件
            print(f"  {file}")
        if len(files) > 20:
            print(f"  ... 还有 {len(files) - 20} 个文件")
    
    return True

def check_eccl_libraries():
    """检查ECCL库文件"""
    print("\n=== 检查ECCL库文件 ===")
    
    # 常见的ECCL库路径
    possible_paths = [
        "/usr/lib/x86_64-linux-gnu/",
        "/usr/local/lib/",
        "/opt/enflame/lib/",
        "/usr/lib/",
        "/lib/x86_64-linux-gnu/"
    ]
    
    eccl_libs = []
    for path in possible_paths:
        if os.path.exists(path):
            try:
                for file in os.listdir(path):
                    if 'eccl' in file.lower() and (file.endswith('.so') or file.endswith('.a')):
                        eccl_libs.append(os.path.join(path, file))
            except PermissionError:
                continue
    
    if eccl_libs:
        print("✓ 找到ECCL库文件:")
        for lib in eccl_libs:
            print(f"  {lib}")
    else:
        print("✗ 未找到ECCL库文件")
    
    return eccl_libs

def check_eccl_headers():
    """检查ECCL头文件"""
    print("\n=== 检查ECCL头文件 ===")
    
    # 常见的头文件路径
    possible_paths = [
        "/usr/include/",
        "/usr/local/include/",
        "/opt/enflame/include/"
    ]
    
    eccl_headers = []
    for path in possible_paths:
        if os.path.exists(path):
            try:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if 'eccl' in file.lower() and file.endswith('.h'):
                            eccl_headers.append(os.path.join(root, file))
            except PermissionError:
                continue
    
    if eccl_headers:
        print("✓ 找到ECCL头文件:")
        for header in eccl_headers:
            print(f"  {header}")
    else:
        print("✗ 未找到ECCL头文件")
    
    return eccl_headers

def check_python_eccl_module():
    """检查Python ECCL模块"""
    print("\n=== 检查Python ECCL模块 ===")
    
    # 检查eccl模块是否可导入
    try:
        spec = importlib.util.find_spec("eccl")
        if spec is not None:
            print("✓ Python eccl模块可用")
            print(f"  模块路径: {spec.origin}")
            
            # 尝试导入
            try:
                eccl_module = importlib.import_module("eccl")
                print("✓ eccl模块导入成功")
                if hasattr(eccl_module, '__version__'):
                    print(f"  版本: {eccl_module.__version__}")
                if hasattr(eccl_module, '__file__'):
                    print(f"  文件路径: {eccl_module.__file__}")
            except Exception as e:
                print(f"✗ eccl模块导入失败: {e}")
        else:
            print("✗ Python eccl模块不可用")
    except Exception as e:
        print(f"✗ 检查eccl模块时出错: {e}")

def check_environment_variables():
    """检查相关环境变量"""
    print("\n=== 检查环境变量 ===")
    
    env_vars = [
        'LD_LIBRARY_PATH',
        'PYTHONPATH',
        'PATH',
        'TOPS_VISIBLE_DEVICES',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '')
        if value:
            print(f"✓ {var}={value}")
        else:
            print(f"- {var}: 未设置")

def check_torch_distributed_backends():
    """检查PyTorch分布式后端"""
    print("\n=== 检查PyTorch分布式后端 ===")
    
    try:
        import torch
        import torch.distributed as dist
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"分布式可用: {dist.is_available()}")
        
        # 检查可用后端
        backends = []
        if hasattr(dist, 'Backend'):
            for backend in ['GLOO', 'NCCL', 'MPI']:
                if hasattr(dist.Backend, backend):
                    backends.append(backend)
        
        print(f"可用后端: {backends}")
        
        # 尝试检查eccl后端
        try:
            # 这里不实际初始化，只是检查是否可以创建
            print("尝试检查eccl后端支持...")
            # 注意：这里不能直接测试eccl，因为需要分布式环境
        except Exception as e:
            print(f"eccl后端检查失败: {e}")
            
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")

def generate_fix_suggestions():
    """生成修复建议"""
    print("\n=== 修复建议 ===")
    
    suggestions = [
        "1. 确认ECCL库路径是否在LD_LIBRARY_PATH中",
        "2. 检查是否需要安装ECCL的Python绑定",
        "3. 验证ECCL版本与PyTorch版本的兼容性",
        "4. 确认GCU驱动是否正确安装",
        "5. 检查是否需要重启或重新加载环境"
    ]
    
    for suggestion in suggestions:
        print(suggestion)
    
    print("\n建议的修复命令:")
    print("# 1. 检查并设置库路径")
    print("export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
    print("export LD_LIBRARY_PATH=/opt/enflame/lib:$LD_LIBRARY_PATH")
    print()
    print("# 2. 重新安装ECCL（如果需要）")
    print("sudo dpkg --purge tops-eccl")
    print("sudo dpkg -i tops-eccl_2.5.136-1_amd64.deb")
    print()
    print("# 3. 检查Python路径")
    print("python3 -c \"import sys; print('\\n'.join(sys.path))\"")

def main():
    """主函数"""
    print("ECCL安装详细诊断")
    print("=" * 50)
    
    # 执行各项检查
    check_eccl_package()
    check_eccl_libraries()
    check_eccl_headers()
    check_python_eccl_module()
    check_environment_variables()
    check_torch_distributed_backends()
    generate_fix_suggestions()
    
    print("\n诊断完成！")

if __name__ == '__main__':
    main()