#!/usr/bin/env python3
"""
修复MMSegmentation中的dns.rdtypes错误
这个错误通常是由于dnspython版本不兼容导致的
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令并返回结果"""
    print(f"🔧 {description}")
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            if result.stdout:
                print(f"输出: {result.stdout}")
            return True
        else:
            print(f"❌ {description} 失败")
            if result.stderr:
                print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False

def check_dns_issue():
    """检查DNS相关问题"""
    print("🔍 检查DNS相关包...")
    
    # 检查dnspython版本
    try:
        import dns
        print(f"✅ dnspython版本: {dns.__version__}")
        
        # 尝试导入有问题的模块
        try:
            from dns.rdtypes import ANY
            print("✅ dns.rdtypes.ANY 可以正常导入")
            return True
        except AttributeError as e:
            print(f"❌ dns.rdtypes.ANY 导入失败: {e}")
            return False
        except ImportError as e:
            print(f"❌ dns.rdtypes 导入失败: {e}")
            return False
            
    except ImportError:
        print("❌ dnspython 未安装")
        return False

def fix_dns_issue():
    """修复DNS问题"""
    print("\n🔧 开始修复DNS问题...")
    
    # 方法1: 升级dnspython到兼容版本
    print("\n方法1: 升级dnspython...")
    if run_command("pip3 install --upgrade dnspython", "升级dnspython"):
        if check_dns_issue():
            return True
    
    # 方法2: 安装特定版本的dnspython
    print("\n方法2: 安装dnspython 2.1.0...")
    if run_command("pip3 install dnspython==2.1.0", "安装dnspython 2.1.0"):
        if check_dns_issue():
            return True
    
    # 方法3: 重新安装dnspython
    print("\n方法3: 重新安装dnspython...")
    run_command("pip3 uninstall -y dnspython", "卸载dnspython")
    if run_command("pip3 install dnspython", "重新安装dnspython"):
        if check_dns_issue():
            return True
    
    # 方法4: 尝试降级到稳定版本
    print("\n方法4: 降级到dnspython 1.16.0...")
    if run_command("pip3 install dnspython==1.16.0", "安装dnspython 1.16.0"):
        if check_dns_issue():
            return True
    
    return False

def test_mmseg_import():
    """测试MMSegmentation导入"""
    print("\n🧪 测试MMSegmentation导入...")
    
    try:
        import mmseg
        print(f"✅ MMSegmentation版本: {mmseg.__version__}")
        
        # 测试关键组件
        from mmseg.apis import init_segmentor
        print("✅ mmseg.apis 导入成功")
        
        from mmseg.datasets import build_dataset
        print("✅ mmseg.datasets 导入成功")
        
        from mmseg.models import build_segmentor
        print("✅ mmseg.models 导入成功")
        
        print("✅ MMSegmentation 所有关键组件导入成功")
        return True
        
    except Exception as e:
        print(f"❌ MMSegmentation 导入失败: {e}")
        return False

def main():
    print("🔧 DNS rdtypes 错误修复脚本")
    print("=" * 50)
    
    # 检查当前状态
    if check_dns_issue():
        print("✅ DNS 模块正常，无需修复")
        if test_mmseg_import():
            print("✅ 所有组件正常工作")
            return
    
    # 尝试修复
    if fix_dns_issue():
        print("\n✅ DNS 问题修复成功")
        
        # 验证修复结果
        if test_mmseg_import():
            print("✅ MMSegmentation 现在可以正常工作")
        else:
            print("❌ MMSegmentation 仍有问题，可能需要重新安装")
    else:
        print("\n❌ DNS 问题修复失败")
        print("\n💡 建议手动操作:")
        print("1. pip3 uninstall -y dnspython")
        print("2. pip3 install dnspython==2.1.0")
        print("3. 或者尝试: pip3 install dnspython==1.16.0")
        print("4. 重启Python环境后重新测试")

if __name__ == "__main__":
    main()