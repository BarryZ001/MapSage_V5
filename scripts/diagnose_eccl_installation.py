#!/usr/bin/env python3
"""
eccl安装诊断脚本
用于诊断eccl模块安装失败的原因并提供解决方案
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

def run_command(cmd, capture_output=True):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_eccl_installation_status():
    """检查eccl安装状态"""
    print("=" * 60)
    print("🔍 eccl安装状态诊断")
    print("=" * 60)
    
    # 1. 检查Python模块
    try:
        # 动态导入eccl模块以避免静态分析错误
        eccl = __import__('eccl')
        print("✅ eccl Python模块已安装")
        if hasattr(eccl, '__version__'):
            print(f"   版本: {eccl.__version__}")
        return True
    except ImportError as e:
        print(f"❌ eccl Python模块未找到: {e}")
    
    # 2. 检查系统库文件
    eccl_lib_paths = [
        "/usr/local/lib/libeccl.so*",
        "/opt/tops/lib/libeccl.so*", 
        "/usr/lib/libeccl.so*",
        "/usr/local/topsrider/*/lib/libeccl.so*"
    ]
    
    found_libs = []
    for pattern in eccl_lib_paths:
        libs = glob.glob(pattern)
        found_libs.extend(libs)
    
    if found_libs:
        print("✅ 找到eccl库文件:")
        for lib in found_libs:
            print(f"   📁 {lib}")
    else:
        print("❌ 未找到eccl库文件")
    
    # 3. 检查Python site-packages
    python_paths = [
        "/usr/local/lib/python3.8/dist-packages/eccl*",
        "/opt/tops/lib/python3.8/site-packages/eccl*",
        "/usr/lib/python3/dist-packages/eccl*"
    ]
    
    found_python_packages = []
    for pattern in python_paths:
        packages = glob.glob(pattern)
        found_python_packages.extend(packages)
    
    if found_python_packages:
        print("✅ 找到eccl Python包:")
        for pkg in found_python_packages:
            print(f"   📦 {pkg}")
    else:
        print("❌ 未找到eccl Python包")
    
    return len(found_libs) > 0 or len(found_python_packages) > 0

def check_installer_components():
    """检查安装包中的eccl组件"""
    print("\n" + "=" * 60)
    print("🔍 检查TopsRider安装包组件")
    print("=" * 60)
    
    installer_paths = [
        "/installer/TopsRider_t2x_2.5.136_deb_amd64.run",
        "/tmp/TopsRider_t2x_2.5.136_deb_amd64.run",
        "/opt/TopsRider_t2x_2.5.136_deb_amd64.run"
    ]
    
    installer_found = None
    for path in installer_paths:
        if os.path.exists(path):
            installer_found = path
            print(f"✅ 找到安装包: {path}")
            break
    
    if not installer_found:
        print("❌ 未找到TopsRider安装包")
        print("💡 请确认安装包路径是否正确")
        return False
    
    # 列出安装包中的组件
    print("\n📋 查看安装包组件:")
    cmd = f"{installer_found} -l"
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print("✅ 安装包组件列表:")
        print(stdout)
        
        # 检查是否包含eccl相关组件
        if "eccl" in stdout.lower() or "tops-eccl" in stdout.lower():
            print("✅ 安装包包含eccl组件")
            return True
        else:
            print("❌ 安装包中未找到eccl组件")
            return False
    else:
        print(f"❌ 无法列出安装包组件: {stderr}")
        return False

def try_eccl_installation_methods():
    """尝试不同的eccl安装方法"""
    print("\n" + "=" * 60)
    print("🔧 尝试eccl安装方法")
    print("=" * 60)
    
    methods = [
        {
            "name": "方法1: 使用tops-eccl组件安装",
            "command": "/installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl"
        },
        {
            "name": "方法2: 使用eccl组件安装", 
            "command": "/installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C eccl"
        },
        {
            "name": "方法3: 安装完整TopsRider后再安装eccl",
            "command": "/installer/TopsRider_t2x_2.5.136_deb_amd64.run -y && /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl"
        }
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n{i}. {method['name']}")
        print(f"   命令: {method['command']}")
        
        # 询问是否执行（在实际使用时）
        print("   💡 建议手动执行此命令")

def check_system_dependencies():
    """检查系统依赖"""
    print("\n" + "=" * 60)
    print("🔍 检查系统依赖")
    print("=" * 60)
    
    # 检查必要的系统库
    dependencies = [
        "libmpi-dev",
        "libopenmpi-dev", 
        "build-essential",
        "python3-dev"
    ]
    
    for dep in dependencies:
        cmd = f"dpkg -l | grep {dep}"
        returncode, stdout, stderr = run_command(cmd)
        
        if returncode == 0 and stdout.strip():
            print(f"✅ {dep}: 已安装")
        else:
            print(f"❌ {dep}: 未安装")
            print(f"   安装命令: sudo apt-get install {dep}")

def provide_troubleshooting_guide():
    """提供故障排除指南"""
    print("\n" + "=" * 60)
    print("📋 eccl安装故障排除指南")
    print("=" * 60)
    
    print("🚀 推荐解决步骤:")
    print()
    print("1. 检查安装包完整性:")
    print("   ls -la /installer/TopsRider_t2x_2.5.136_deb_amd64.run")
    print("   /installer/TopsRider_t2x_2.5.136_deb_amd64.run -l | grep -i eccl")
    print()
    print("2. 尝试重新安装eccl组件:")
    print("   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl")
    print()
    print("3. 如果上述失败，尝试完整安装:")
    print("   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y")
    print()
    print("4. 检查安装日志:")
    print("   tail -f /var/log/topsrider_install.log")
    print()
    print("5. 手动设置环境变量:")
    print("   export LD_LIBRARY_PATH=/opt/tops/lib:$LD_LIBRARY_PATH")
    print("   export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:$PYTHONPATH")
    print()
    print("6. 验证安装:")
    print("   python3 -c 'import eccl; print(eccl.__version__)'")

def main():
    """主函数"""
    print("🔍 eccl安装诊断工具")
    print("=" * 60)
    
    # 检查当前eccl状态
    eccl_installed = check_eccl_installation_status()
    
    if not eccl_installed:
        # 检查安装包组件
        installer_ok = check_installer_components()
        
        # 检查系统依赖
        check_system_dependencies()
        
        # 提供安装方法
        try_eccl_installation_methods()
        
        # 提供故障排除指南
        provide_troubleshooting_guide()
    else:
        print("\n✅ eccl已正确安装！")

if __name__ == "__main__":
    main()