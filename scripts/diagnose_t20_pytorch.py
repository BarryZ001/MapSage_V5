#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T20环境PyTorch诊断脚本
检查PyTorch安装状态和分布式模块可用性
"""

import sys
import os
import subprocess

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_python_environment():
    """检查Python环境"""
    print("=== Python环境检查 ===")
    print("Python版本:", sys.version)
    print("Python路径:", sys.executable)
    print("Python路径(which):", end=" ")
    
    code, stdout, stderr = run_command("which python3")
    if code == 0:
        print(stdout.strip())
    else:
        print("未找到python3")
    
    print()

def check_pytorch_installation():
    """检查PyTorch安装"""
    print("=== PyTorch安装检查 ===")
    
    # 检查torch模块
    try:
        import torch
        print("✅ torch模块导入成功")
        print("PyTorch版本:", torch.__version__)
        print("PyTorch安装路径:", torch.__file__)
    except ImportError as e:
        print("❌ torch模块导入失败:", e)
        return False
    
    # 检查torch.distributed模块
    try:
        import torch.distributed as dist
        print("✅ torch.distributed模块导入成功")
        print("分布式后端支持:")
        
        # 检查支持的后端
        backends = []
        if dist.is_available():
            print("  - 分布式训练可用")
            if dist.is_nccl_available():
                backends.append("nccl")
            if dist.is_gloo_available():
                backends.append("gloo")
            if dist.is_mpi_available():
                backends.append("mpi")
            print("  - 支持的后端:", backends)
        else:
            print("  - 分布式训练不可用")
            
    except ImportError as e:
        print("❌ torch.distributed模块导入失败:", e)
        return False
    
    print()
    return True

def check_torch_gcu():
    """检查torch_gcu模块"""
    print("=== torch_gcu模块检查 ===")
    
    try:
        import torch_gcu  # type: ignore
        print("✅ torch_gcu模块导入成功")
        print("torch_gcu版本:", getattr(torch_gcu, '__version__', '未知'))
        print("可用GCU设备数:", torch_gcu.device_count())
        
        # 检查当前设备
        if torch_gcu.device_count() > 0:
            print("当前GCU设备:", torch_gcu.current_device())
            for i in range(torch_gcu.device_count()):
                print("GCU设备 {}: {}".format(i, torch_gcu.get_device_name(i)))
        
    except ImportError as e:
        print("❌ torch_gcu模块导入失败:", e)
        return False
    
    print()
    return True

def check_pip_packages():
    """检查pip包安装"""
    print("=== pip包检查 ===")
    
    packages = ["torch", "torchvision", "torch_gcu"]
    
    for package in packages:
        code, stdout, stderr = run_command("pip3 show {}".format(package))
        if code == 0:
            lines = stdout.strip().split('\n')
            version_line = [line for line in lines if line.startswith('Version:')]
            location_line = [line for line in lines if line.startswith('Location:')]
            
            if version_line:
                print("✅ {}: {}".format(package, version_line[0]))
            if location_line:
                print("   位置: {}".format(location_line[0].replace('Location: ', '')))
        else:
            print("❌ {} 未安装".format(package))
    
    print()

def check_environment_variables():
    """检查环境变量"""
    print("=== 环境变量检查 ===")
    
    env_vars = [
        "PYTHONPATH",
        "LD_LIBRARY_PATH", 
        "CUDA_VISIBLE_DEVICES",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT"
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print("{}: {}".format(var, value))
        else:
            print("{}: 未设置".format(var))
    
    print()

def check_torchrun():
    """检查torchrun命令"""
    print("=== torchrun命令检查 ===")
    
    # 检查torchrun是否可用
    code, stdout, stderr = run_command("which torchrun")
    if code == 0:
        print("✅ torchrun路径:", stdout.strip())
        
        # 检查torchrun版本
        code, stdout, stderr = run_command("torchrun --help")
        if code == 0:
            print("✅ torchrun命令可用")
        else:
            print("❌ torchrun命令执行失败:", stderr)
    else:
        print("❌ torchrun命令未找到")
        
        # 尝试python -m torch.distributed.run
        code, stdout, stderr = run_command("python3 -m torch.distributed.run --help")
        if code == 0:
            print("✅ python -m torch.distributed.run 可用")
        else:
            print("❌ python -m torch.distributed.run 不可用:", stderr)
    
    print()

def main():
    """主函数"""
    print("🔍 T20环境PyTorch诊断开始...")
    print("=" * 50)
    
    check_python_environment()
    pytorch_ok = check_pytorch_installation()
    torch_gcu_ok = check_torch_gcu()
    check_pip_packages()
    check_environment_variables()
    check_torchrun()
    
    print("=" * 50)
    print("🏁 诊断完成")
    
    if pytorch_ok and torch_gcu_ok:
        print("✅ PyTorch和torch_gcu模块正常")
    else:
        print("❌ 发现问题，需要修复PyTorch安装")
        
        # 提供修复建议
        print("\n🔧 修复建议:")
        if not pytorch_ok:
            print("1. 重新安装PyTorch: pip3 install torch torchvision")
        if not torch_gcu_ok:
            print("2. 检查torch_gcu安装: pip3 install torch_gcu")
        print("3. 检查环境变量配置")
        print("4. 重启容器或重新加载环境")

if __name__ == "__main__":
    main()