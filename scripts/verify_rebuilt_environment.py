#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T20重建环境验证脚本
用于验证通过rebuild_t20_environment.sh重建的环境是否正确配置

使用方法:
1. 在重建环境后，进入容器
2. 执行: python scripts/verify_rebuilt_environment.py
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def print_header(title):
    """打印格式化的标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """打印章节标题"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def check_command(cmd, description):
    """检查命令是否可执行"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ {description}: 成功")
            if result.stdout.strip():
                print(f"  输出: {result.stdout.strip()[:100]}")
            return True
        else:
            print(f"✗ {description}: 失败 (退出码: {result.returncode})")
            if result.stderr.strip():
                print(f"  错误: {result.stderr.strip()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description}: 超时")
        return False
    except Exception as e:
        print(f"✗ {description}: 异常 - {str(e)}")
        return False

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {description}: 存在 ({filepath})")
        return True
    else:
        print(f"✗ {description}: 缺失 ({filepath})")
        return False

def check_python_import(module_name, description):
    """检查Python模块是否可导入"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None and spec.loader is not None:
            # 尝试实际导入
            module = importlib.import_module(module_name)
            print(f"✓ {description}: 可导入")
            return True, module
        else:
            print(f"✗ {description}: 模块规范未找到")
            return False, None
    except ImportError as e:
        print(f"✗ {description}: 导入失败 - {str(e)}")
        return False, None
    except Exception as e:
        print(f"✗ {description}: 异常 - {str(e)}")
        return False, None

def main():
    print_header("T20重建环境验证报告")
    
    # 统计结果
    total_checks = 0
    passed_checks = 0
    
    # 1. 基础系统检查
    print_section("1. 基础系统环境")
    
    checks = [
        ("python --version", "Python版本"),
        ("pip3 --version", "pip3版本"),
        ("which python", "Python路径"),
        ("echo $PATH", "PATH环境变量"),
    ]
    
    for cmd, desc in checks:
        total_checks += 1
        if check_command(cmd, desc):
            passed_checks += 1
    
    # 2. TopsRider软件栈检查
    print_section("2. TopsRider软件栈")
    
    # 关键文件检查
    critical_files = [
        ("/opt/tops/bin/tops-smi", "tops-smi工具"),
        ("/opt/tops/lib/libtops.so", "TOPS核心库"),
        ("/opt/tops/lib/python3.8/site-packages", "Python包目录"),
    ]
    
    for filepath, desc in critical_files:
        total_checks += 1
        if check_file_exists(filepath, desc):
            passed_checks += 1
    
    # 环境变量检查
    env_vars = [
        ("LD_LIBRARY_PATH", "/opt/tops/lib"),
        ("PYTHONPATH", "/opt/tops/lib/python3.8/site-packages"),
        ("PATH", "/opt/tops/bin"),
    ]
    
    for var_name, expected_path in env_vars:
        total_checks += 1
        var_value = os.environ.get(var_name, "")
        if expected_path in var_value:
            print(f"✓ {var_name}环境变量: 包含{expected_path}")
            passed_checks += 1
        else:
            print(f"✗ {var_name}环境变量: 不包含{expected_path}")
            print(f"  当前值: {var_value}")
    
    # 3. PyTorch和torch-gcu检查
    print_section("3. PyTorch和torch-gcu框架")
    
    # PyTorch基础检查
    total_checks += 1
    torch_success, torch_module = check_python_import("torch", "PyTorch模块")
    if torch_success and torch_module is not None:
        passed_checks += 1
        try:
            if hasattr(torch_module, '__version__'):
                print(f"  PyTorch版本: {torch_module.__version__}")
            else:
                print("  无法获取PyTorch版本")
        except:
            print("  无法获取PyTorch版本")
    
    # torch.gcu检查
    if torch_success and torch_module is not None:
        total_checks += 1
        try:
            if hasattr(torch_module, 'gcu'):
                gcu_available = torch_module.gcu.is_available()
                if gcu_available:
                    print("✓ torch.gcu可用性: True")
                    passed_checks += 1
                    
                    # GCU设备数量
                    try:
                        if hasattr(torch_module.gcu, 'device_count'):
                            device_count = torch_module.gcu.device_count()
                            print(f"  GCU设备数量: {device_count}")
                        else:
                            print("  无法获取GCU设备数量")
                    except:
                        print("  无法获取GCU设备数量")
                else:
                    print("✗ torch.gcu可用性: False")
            else:
                print("✗ torch.gcu模块不存在")
        except Exception as e:
            print(f"✗ torch.gcu检查失败: {str(e)}")
    
    # 4. 项目特定检查
    print_section("4. 项目环境")
    
    # 检查项目目录
    project_paths = [
        ("/workspace/code/MapSage_V5", "项目代码目录"),
        ("/workspace/data", "数据目录"),
        ("/workspace/weights", "权重目录"),
        ("/workspace/outputs", "输出目录"),
    ]
    
    for path, desc in project_paths:
        total_checks += 1
        if check_file_exists(path, desc):
            passed_checks += 1
    
    # 检查项目文件
    if os.path.exists("/workspace/code/MapSage_V5"):
        os.chdir("/workspace/code/MapSage_V5")
        project_files = [
            ("requirements.txt", "依赖文件"),
            ("scripts/validate_official_installation.py", "官方验证脚本"),
            ("scripts/diagnose_torch_gcu.py", "诊断脚本"),
        ]
        
        for filepath, desc in project_files:
            total_checks += 1
            if check_file_exists(filepath, desc):
                passed_checks += 1
    
    # 5. 关键Python包检查
    print_section("5. 关键Python包")
    
    key_packages = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("mmcv", "MMCV"),
        ("mmseg", "MMSegmentation"),
    ]
    
    for package, desc in key_packages:
        total_checks += 1
        success, _ = check_python_import(package, desc)
        if success:
            passed_checks += 1
    
    # 6. 功能性测试
    print_section("6. 功能性测试")
    
    if torch_success and torch_module is not None:
        # 简单的张量操作测试
        total_checks += 1
        try:
            x = torch_module.randn(2, 3)
            y = x + 1
            print("✓ PyTorch张量操作: 成功")
            passed_checks += 1
        except Exception as e:
            print(f"✗ PyTorch张量操作: 失败 - {str(e)}")
        
        # GCU设备测试
        if hasattr(torch_module, 'gcu'):
            total_checks += 1
            try:
                if torch_module.gcu.is_available():
                    device = torch_module.device('gcu:0')
                    x = torch_module.randn(2, 3, device=device)
                    print("✓ GCU设备张量创建: 成功")
                    passed_checks += 1
                else:
                    print("✗ GCU设备张量创建: GCU不可用")
            except Exception as e:
                print(f"✗ GCU设备张量创建: 失败 - {str(e)}")
        else:
            total_checks += 1
            print("✗ GCU设备张量创建: torch.gcu模块不存在")
    
    # 7. 生成总结报告
    print_header("验证总结")
    
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"总检查项目: {total_checks}")
    print(f"通过项目: {passed_checks}")
    print(f"失败项目: {total_checks - passed_checks}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 环境验证成功！T20环境已正确配置。")
        print("\n建议后续步骤:")
        print("1. 运行完整的官方验证: python scripts/validate_official_installation.py")
        print("2. 开始训练任务")
        return 0
    elif success_rate >= 70:
        print("\n⚠️  环境基本可用，但存在一些问题。")
        print("\n建议:")
        print("1. 检查失败的项目并尝试修复")
        print("2. 运行诊断脚本: python scripts/diagnose_torch_gcu.py")
        return 1
    else:
        print("\n❌ 环境验证失败，需要重新配置。")
        print("\n建议:")
        print("1. 重新运行重建脚本: bash scripts/rebuild_t20_environment.sh")
        print("2. 检查TopsRider安装包是否正确")
        print("3. 联系技术支持")
        return 2

if __name__ == "__main__":
    sys.exit(main())