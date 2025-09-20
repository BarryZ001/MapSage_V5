#!/usr/bin/env python3
"""
T20环境DNS rdtypes错误修复脚本
修复MMSegmentation中的 'module dns.rdtypes has no attribute ANY' 错误
"""

import subprocess
import sys
import os

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
            return True
        else:
            print(f"❌ 失败: {description}")
            if result.stderr.strip():
                print(f"错误: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ 超时: {description}")
        return False
    except Exception as e:
        print(f"❌ 异常: {description} - {e}")
        return False

def check_dns_issue():
    """检查DNS相关问题"""
    print("🔍 检查DNS相关包...")
    
    # 检查dnspython版本
    try:
        # 使用subprocess检查而不是直接导入
        result = subprocess.run([sys.executable, "-c", "import dns; print(dns.__version__)"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ dnspython版本: {version}")
            
            # 检查rdtypes.ANY问题
            test_code = """
try:
    from dns.rdtypes import ANY
    print("SUCCESS: dns.rdtypes.ANY imported")
except Exception as e:
    print(f"ERROR: {e}")
"""
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True)
            if "SUCCESS" in result.stdout:
                print("✅ dns.rdtypes.ANY 可以正常导入")
                return True
            else:
                print(f"❌ dns.rdtypes.ANY 导入失败: {result.stdout.strip()}")
                return False
        else:
            print("❌ dnspython 未安装或无法访问")
            return False
            
    except Exception as e:
        print(f"❌ DNS检查失败: {e}")
        return False

def fix_dns_issue():
    """修复DNS问题"""
    print("\n🔧 开始修复DNS rdtypes问题...")
    
    # 方法1: 升级dnspython到最新版本
    if run_command("pip3 install --upgrade dnspython", "升级dnspython到最新版本"):
        if check_dns_issue():
            return True
    
    # 方法2: 安装特定版本的dnspython (2.2.1是比较稳定的版本)
    if run_command("pip3 install dnspython==2.2.1", "安装dnspython 2.2.1版本"):
        if check_dns_issue():
            return True
    
    # 方法3: 重新安装dnspython
    if run_command("pip3 uninstall -y dnspython && pip3 install dnspython", "重新安装dnspython"):
        if check_dns_issue():
            return True
    
    # 方法4: 尝试安装更老的稳定版本
    if run_command("pip3 install dnspython==2.1.0", "安装dnspython 2.1.0版本"):
        if check_dns_issue():
            return True
    
    print("❌ 所有修复方法都失败了")
    return False

def test_mmsegmentation():
    """测试MMSegmentation导入"""
    print("\n🧪 测试MMSegmentation导入...")
    
    test_code = """
try:
    import mmseg
    print("SUCCESS: MMSegmentation imported successfully")
except Exception as e:
    print(f"ERROR: {e}")
"""
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if "SUCCESS" in result.stdout:
        print("✅ MMSegmentation导入成功")
        return True
    else:
        print(f"❌ MMSegmentation导入失败: {result.stdout.strip()}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 T20环境DNS rdtypes错误修复工具")
    print("=" * 60)
    
    # 检查当前DNS状态
    if check_dns_issue():
        print("\n✅ DNS rdtypes问题已经解决，无需修复")
        if test_mmsegmentation():
            print("✅ MMSegmentation工作正常")
            return True
        else:
            print("⚠️  DNS正常但MMSegmentation仍有问题，可能是其他原因")
    else:
        print("\n❌ 检测到DNS rdtypes问题，开始修复...")
        if fix_dns_issue():
            print("\n✅ DNS问题修复成功！")
            if test_mmsegmentation():
                print("✅ MMSegmentation现在可以正常工作")
                return True
            else:
                print("⚠️  DNS已修复但MMSegmentation仍有问题")
        else:
            print("\n❌ DNS问题修复失败")
    
    print("\n💡 如果问题仍然存在，请尝试:")
    print("   1. 重启Python环境")
    print("   2. 清理pip缓存: pip3 cache purge")
    print("   3. 检查是否有多个Python环境冲突")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)