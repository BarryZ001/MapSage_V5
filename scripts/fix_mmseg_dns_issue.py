#!/usr/bin/env python3
"""
专门修复MMSegmentation中dns.rdtypes.ANY错误的脚本
针对T20环境的特殊情况进行修复
"""

import subprocess
import sys
import os
import importlib.util

def run_command(cmd, description=""):
    """执行命令并返回结果"""
    print(f"🔧 {description}")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ 成功: {description}")
            if result.stdout.strip():
                print(f"输出: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print(f"❌ 失败: {description}")
            if result.stderr.strip():
                print(f"错误: {result.stderr.strip()}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"❌ 超时: {description}")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ 异常: {description} - {e}")
        return False, str(e)

def check_dns_installation():
    """检查DNS包的安装情况"""
    print("🔍 检查DNS包安装情况...")
    
    # 检查dnspython是否安装
    success, output = run_command("pip3 list | grep -i dns", "查找DNS相关包")
    if success:
        print(f"已安装的DNS包: {output}")
    
    # 检查Python路径中的dns模块
    success, output = run_command("python3 -c \"import sys; print('\\n'.join(sys.path))\"", "查看Python路径")
    
    # 查找dns模块位置
    success, output = run_command("python3 -c \"import dns; print(dns.__file__)\"", "查找dns模块位置")
    if success:
        print(f"dns模块位置: {output}")

def fix_dns_rdtypes_issue():
    """修复dns.rdtypes.ANY问题"""
    print("\n🔧 修复dns.rdtypes.ANY问题...")
    
    # 方法1: 重新安装dnspython
    print("方法1: 重新安装dnspython...")
    success, _ = run_command("pip3 uninstall -y dnspython", "卸载现有dnspython")
    success, _ = run_command("pip3 install dnspython==2.3.0", "安装dnspython 2.3.0")
    
    if test_dns_import():
        return True
    
    # 方法2: 尝试其他版本
    versions = ["2.2.1", "2.1.0", "2.4.2"]
    for version in versions:
        print(f"方法2: 尝试dnspython {version}...")
        success, _ = run_command(f"pip3 install dnspython=={version}", f"安装dnspython {version}")
        if test_dns_import():
            return True
    
    # 方法3: 从源码安装
    print("方法3: 从源码安装最新版本...")
    success, _ = run_command("pip3 install --upgrade --force-reinstall dnspython", "强制重新安装dnspython")
    if test_dns_import():
        return True
    
    return False

def test_dns_import():
    """测试dns.rdtypes.ANY导入"""
    print("🧪 测试dns.rdtypes.ANY导入...")
    
    test_code = """
try:
    import dns
    print(f"DNS version: {dns.__version__}")
    
    from dns.rdtypes import ANY
    print("SUCCESS: dns.rdtypes.ANY imported successfully")
    
    # 测试具体的ANY类
    print(f"ANY module: {ANY}")
    print("Available attributes:", [attr for attr in dir(ANY) if not attr.startswith('_')])
    
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except AttributeError as e:
    print(f"ATTRIBUTE_ERROR: {e}")
except Exception as e:
    print(f"OTHER_ERROR: {e}")
"""
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if "SUCCESS" in result.stdout:
        print("✅ dns.rdtypes.ANY导入成功")
        print(result.stdout)
        return True
    else:
        print(f"❌ dns.rdtypes.ANY导入失败:")
        print(result.stdout)
        if result.stderr:
            print(f"错误信息: {result.stderr}")
        return False

def test_mmsegmentation_import():
    """测试MMSegmentation导入"""
    print("\n🧪 测试MMSegmentation导入...")
    
    test_code = """
try:
    import mmseg
    print("SUCCESS: MMSegmentation imported successfully")
    print(f"MMSeg version: {mmseg.__version__}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
"""
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if "SUCCESS" in result.stdout:
        print("✅ MMSegmentation导入成功")
        print(result.stdout)
        return True
    else:
        print(f"❌ MMSegmentation导入失败:")
        print(result.stdout)
        if result.stderr:
            print(f"错误信息: {result.stderr}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 MMSegmentation DNS rdtypes错误修复工具")
    print("=" * 60)
    
    # 检查当前状态
    check_dns_installation()
    
    # 测试当前DNS导入状态
    if test_dns_import():
        print("✅ DNS rdtypes问题已经解决")
        if test_mmsegmentation_import():
            print("✅ MMSegmentation工作正常")
            return True
        else:
            print("⚠️ DNS正常但MMSegmentation仍有问题")
    
    # 尝试修复
    if fix_dns_rdtypes_issue():
        print("✅ DNS问题修复成功")
        if test_mmsegmentation_import():
            print("✅ MMSegmentation现在工作正常")
            return True
        else:
            print("⚠️ DNS修复成功但MMSegmentation仍有问题")
            return False
    else:
        print("❌ DNS问题修复失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)