#!/usr/bin/env python3
"""
T20环境训练启动DNS问题修复工具
专门解决eventlet、wandb、sentry-sdk导致的DNS rdtypes错误

错误特征:
- AttributeError: module 'dns.rdtypes' has no attribute 'ANY'
- 发生在训练启动时，通过eventlet -> wandb -> sentry-sdk链条触发
"""

import subprocess
import sys
import os
import importlib

def run_command(cmd, description=""):
    """执行命令并返回结果"""
    try:
        print(f"🔧 {description}")
        print(f"   执行: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"   ✅ 成功")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print(f"   ❌ 失败 (退出码: {result.returncode})")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"   ⏰ 命令超时")
        return False, "Command timeout"
    except Exception as e:
        print(f"   💥 异常: {str(e)}")
        return False, str(e)

def check_problematic_packages():
    """检查可能导致DNS问题的包"""
    print("🔍 检查可能导致DNS问题的包...")
    
    problematic_packages = {
        'eventlet': '检查eventlet版本和DNS补丁',
        'wandb': '检查wandb是否导入DNS问题',
        'sentry-sdk': '检查sentry-sdk DNS依赖',
        'dnspython': '检查dnspython版本兼容性'
    }
    
    issues = []
    
    for package, description in problematic_packages.items():
        try:
            if package == 'sentry-sdk':
                try:
                    sentry_module = importlib.import_module('sentry_sdk')
                    version = getattr(sentry_module, '__version__', 'unknown')
                except ImportError:
                    print(f"   ⚠️  {package}: 未安装")
                    continue
            else:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
            
            print(f"   ✅ {package}: {version}")
            
            # 特殊检查
            if package == 'dnspython':
                try:
                    dns_rdtypes = importlib.import_module('dns.rdtypes')
                    # 尝试访问ANY属性
                    if hasattr(dns_rdtypes, 'ANY'):
                        print(f"      ✅ dns.rdtypes.ANY 可用")
                    else:
                        print(f"      ❌ dns.rdtypes.ANY 属性不存在")
                        issues.append(f"{package}: dns.rdtypes.ANY属性缺失")
                except ImportError as e:
                    print(f"      ❌ dns.rdtypes 不可用: {e}")
                    issues.append(f"{package}: dns.rdtypes导入失败")
                    
        except ImportError:
            print(f"   ⚠️  {package}: 未安装")
        except Exception as e:
            print(f"   ❌ {package}: 检查失败 - {e}")
            issues.append(f"{package}: {e}")
    
    return issues

def fix_dns_comprehensive():
    """全面修复DNS问题"""
    print("\n🔧 开始全面DNS修复...")
    
    fixes = [
        # 1. 强制重新安装dnspython
        ("pip3 uninstall -y dnspython", "卸载当前dnspython"),
        ("pip3 install dnspython==2.3.0", "安装兼容版本dnspython"),
        
        # 2. 修复eventlet相关问题
        ("pip3 install --upgrade eventlet", "升级eventlet"),
        
        # 3. 处理wandb和sentry-sdk
        ("pip3 install --upgrade wandb", "升级wandb"),
        ("pip3 install --upgrade sentry-sdk", "升级sentry-sdk"),
        
        # 4. 清理缓存
        ("pip3 cache purge", "清理pip缓存"),
        
        # 5. 强制重新安装所有相关包
        ("pip3 install --force-reinstall --no-cache-dir dnspython==2.3.0 eventlet wandb sentry-sdk", "强制重新安装关键包"),
    ]
    
    success_count = 0
    for cmd, desc in fixes:
        success, output = run_command(cmd, desc)
        if success:
            success_count += 1
        else:
            print(f"   ⚠️  修复步骤失败，但继续执行...")
    
    print(f"\n📊 修复完成: {success_count}/{len(fixes)} 步骤成功")
    return success_count > len(fixes) // 2  # 超过一半成功就认为修复有效

def test_training_imports():
    """测试训练相关的导入"""
    print("\n🧪 测试训练相关导入...")
    
    test_imports = [
        "import dns.rdtypes.ANY",
        "import eventlet",
        "import wandb",
        "import sentry_sdk",
        "import torch",
        "import mmseg",
        "from timm.utils.misc import natural_key",  # 这是触发问题的具体导入
    ]
    
    success_count = 0
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"   ✅ {import_stmt}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {import_stmt} - {e}")
    
    print(f"\n📊 导入测试: {success_count}/{len(test_imports)} 成功")
    return success_count == len(test_imports)

def create_dns_workaround():
    """创建DNS问题的临时解决方案"""
    print("\n🔧 创建DNS问题临时解决方案...")
    
    workaround_script = '''#!/usr/bin/env python3
"""
DNS问题临时解决方案
在训练脚本开始前运行此代码
"""

import os
import sys

# 设置环境变量禁用problematic功能
os.environ['WANDB_DISABLED'] = 'true'  # 禁用wandb
os.environ['SENTRY_DSN'] = ''  # 禁用sentry

# 尝试修复DNS导入问题
try:
    import dns.rdtypes
    if not hasattr(dns.rdtypes, 'ANY'):
        # 手动创建ANY属性
        import dns.rdtypes.ANY as ANY_module
        dns.rdtypes.ANY = ANY_module
        print("✅ DNS rdtypes.ANY 手动修复成功")
except Exception as e:
    print(f"⚠️  DNS手动修复失败: {e}")

print("🔧 DNS临时解决方案已应用")
'''
    
    workaround_path = "/Users/barryzhang/myDev3/MapSage_V5/scripts/dns_workaround.py"
    try:
        with open(workaround_path, 'w', encoding='utf-8') as f:
            f.write(workaround_script)
        print(f"   ✅ 临时解决方案已保存到: {workaround_path}")
        return True
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("🔧 T20环境训练启动DNS问题修复工具")
    print("=" * 70)
    
    # 1. 检查问题包
    issues = check_problematic_packages()
    
    if not issues:
        print("\n✅ 未发现明显的包问题")
    else:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"   - {issue}")
    
    # 2. 执行全面修复
    if fix_dns_comprehensive():
        print("\n✅ DNS全面修复完成")
    else:
        print("\n⚠️  DNS修复部分成功，可能仍有问题")
    
    # 3. 测试导入
    if test_training_imports():
        print("\n✅ 所有训练相关导入测试通过")
        print("\n🎉 DNS问题已解决，可以开始训练！")
        return True
    else:
        print("\n❌ 仍有导入问题，创建临时解决方案...")
        
        # 4. 创建临时解决方案
        if create_dns_workaround():
            print("\n💡 使用临时解决方案:")
            print("   在训练脚本开始前添加:")
            print("   exec(open('scripts/dns_workaround.py').read())")
            print("\n   或者设置环境变量:")
            print("   export WANDB_DISABLED=true")
            print("   export SENTRY_DSN=''")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)