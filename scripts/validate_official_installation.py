#!/usr/bin/env python3
"""
T20环境TopsRider软件栈官方标准验证脚本
基于官方手册V2.1和成功经验整合

此脚本按照官方手册的标准流程验证TopsRider软件栈安装
包括torch-gcu框架和ptex模块的完整性检查
"""

import os
import sys
import subprocess
import importlib.util

def print_header():
    """打印验证脚本头部信息"""
    print("="*60)
    print("🔍 T20环境TopsRider软件栈官方标准验证")
    print("📋 基于官方手册V2.1和成功经验")
    print("="*60)

def check_container_environment():
    """检查是否在正确的容器环境中"""
    print("\n🏠 环境检查:")
    
    # 检查是否在容器内
    if os.path.exists('/usr/local/topsrider'):
        print("  ✅ 检测到TopsRider安装目录")
    else:
        print("  ❌ 未检测到TopsRider安装目录")
        return False
    
    # 检查关键目录结构
    key_dirs = [
        '/usr/local/topsrider',
        '/usr/local/topsrider/ai_development_toolkit',
        '/usr/local/topsrider/ai_development_toolkit/pytorch-gcu',
        '/opt/tops',
        '/opt/tops/bin',
        '/opt/tops/lib'
    ]
    
    for dir_path in key_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path} 存在")
        else:
            print(f"  ❌ {dir_path} 不存在")
    
    return True

def check_topsrider_installation():
    """按照官方手册检查TopsRider基础安装"""
    print("\n📦 TopsRider基础软件栈检查:")
    
    # 检查安装文件
    installer_found = False
    for root, dirs, files in os.walk('/root'):
        for file in files:
            if file.startswith('TopsRider') and file.endswith('.run'):
                print(f"  ✅ 找到安装程序: {os.path.join(root, file)}")
                installer_found = True
                break
        if installer_found:
            break
    
    if not installer_found:
        print("  ⚠️  未找到TopsRider安装程序")
    
    # 检查tops-smi命令（官方手册提到的关键工具）
    try:
        result = subprocess.run(['tops-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ tops-smi 命令可用")
            print(f"    输出预览: {result.stdout.strip()[:100]}...")
        else:
            print(f"  ❌ tops-smi 命令失败: {result.stderr}")
    except Exception as e:
        print(f"  ❌ tops-smi 命令异常: {e}")

def check_torch_gcu_framework():
    """按照官方手册检查torch-gcu框架"""
    print("\n🔥 torch-gcu框架检查（官方手册第二步安装内容）:")
    
    try:
        # 动态导入torch避免静态导入错误
        torch_spec = importlib.util.find_spec('torch')
        if torch_spec is None or torch_spec.loader is None:
            print("  ❌ PyTorch未安装")
            return False
        
        torch = importlib.util.module_from_spec(torch_spec)
        torch_spec.loader.exec_module(torch)
        
        print(f"  ✅ PyTorch版本: {torch.__version__}")
        
        # 检查torch.gcu属性
        if hasattr(torch, 'gcu'):
            print("  ✅ torch.gcu 属性存在")
            
            # 检查GCU可用性
            if hasattr(torch.gcu, 'is_available') and torch.gcu.is_available():
                print("  ✅ torch.gcu.is_available() = True")
                
                # 检查设备数量
                if hasattr(torch.gcu, 'device_count'):
                    device_count = torch.gcu.device_count()
                    print(f"  ✅ GCU设备数量: {device_count}")
                
                return True
            else:
                print("  ❌ torch.gcu.is_available() = False")
                return False
        else:
            print("  ❌ torch.gcu 属性不存在")
            return False
            
    except Exception as e:
        print(f"  ❌ torch-gcu检查异常: {e}")
        return False

def check_ptex_module():
    """按照成功经验检查ptex模块"""
    print("\n🎯 ptex模块检查（torch-gcu框架核心组件）:")
    
    try:
        # 动态导入ptex避免静态导入错误
        ptex_spec = importlib.util.find_spec('ptex')
        if ptex_spec is None or ptex_spec.loader is None:
            print("  ❌ ptex模块未安装")
            return False
        
        ptex = importlib.util.module_from_spec(ptex_spec)
        ptex_spec.loader.exec_module(ptex)
        
        print(f"  ✅ ptex版本: {ptex.__version__}")
        
        # 检查XLA设备
        device_count = ptex.device_count()
        print(f"  ✅ XLA设备数量: {device_count}")
        
        # 测试设备创建
        device = ptex.device('xla')
        print(f"  ✅ XLA设备创建成功: {device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ptex模块检查异常: {e}")
        return False

def check_shared_libraries():
    """按照官方手册检查共享库配置"""
    print("\n📚 共享库配置检查（官方手册ldconfig指导）:")
    
    # 检查关键共享库文件
    key_libraries = [
        '/opt/tops/lib/libtops.so',
        '/usr/local/topsrider/ai_development_toolkit/pytorch-gcu/lib/libtorch_gcu.so'
    ]
    
    for lib in key_libraries:
        if os.path.exists(lib):
            print(f"  ✅ {lib} 存在")
            
            # 使用ldd检查依赖（官方手册建议）
            try:
                result = subprocess.run(['ldd', lib], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    if 'not found' in result.stdout:
                        print(f"    ❌ 存在未找到的依赖项")
                        for line in result.stdout.split('\n'):
                            if 'not found' in line:
                                print(f"      {line.strip()}")
                    else:
                        print(f"    ✅ 依赖项完整")
                else:
                    print(f"    ❌ ldd检查失败: {result.stderr}")
            except Exception as e:
                print(f"    ❌ 依赖检查异常: {e}")
        else:
            print(f"  ❌ {lib} 不存在")
    
    # 检查ldconfig缓存
    try:
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=10)
        if 'libtops' in result.stdout:
            print("  ✅ libtops已在动态链接器缓存中")
        else:
            print("  ❌ libtops未在动态链接器缓存中")
            print("  💡 建议运行: ldconfig")
    except Exception as e:
        print(f"  ❌ ldconfig检查异常: {e}")

def perform_integration_test():
    """执行集成测试（基于成功经验）"""
    print("\n🧪 集成功能测试:")
    
    try:
        # 导入必要模块
        torch_spec = importlib.util.find_spec('torch')
        ptex_spec = importlib.util.find_spec('ptex')
        
        if not torch_spec or not ptex_spec or not torch_spec.loader or not ptex_spec.loader:
            print("  ❌ 缺少必要模块，跳过集成测试")
            return False
        
        torch = importlib.util.module_from_spec(torch_spec)
        torch_spec.loader.exec_module(torch)
        
        ptex = importlib.util.module_from_spec(ptex_spec)
        ptex_spec.loader.exec_module(ptex)
        
        # 创建XLA设备
        device = ptex.device('xla')
        print(f"  ✅ XLA设备创建: {device}")
        
        # 创建测试张量
        x = torch.randn(2, 3).to(device)
        y = torch.randn(2, 3).to(device)
        z = x + y
        
        print(f"  ✅ 张量运算成功: {z.shape}")
        print(f"  ✅ 结果设备: {z.device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 集成测试失败: {e}")
        return False

def generate_summary_report(results):
    """生成验证总结报告"""
    print("\n" + "="*60)
    print("📊 验证总结报告")
    print("="*60)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"总检查项: {total_checks}")
    print(f"通过检查: {passed_checks}")
    print(f"失败检查: {total_checks - passed_checks}")
    print(f"通过率: {passed_checks/total_checks*100:.1f}%")
    
    print("\n📋 详细结果:")
    for check_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {check_name}: {status}")
    
    if passed_checks == total_checks:
        print("\n🎉 所有检查通过！T20环境配置完整")
        print("💡 可以开始进行模型训练")
    else:
        print("\n⚠️  存在配置问题，建议：")
        print("1. 运行修复脚本: bash scripts/fix_t20_environment.sh")
        print("2. 检查TopsRider安装完整性")
        print("3. 重启容器后重新验证")

def main():
    """主验证流程"""
    print_header()
    
    # 执行各项检查
    results = {}
    
    results['环境检查'] = check_container_environment()
    results['TopsRider基础安装'] = check_topsrider_installation()
    results['torch-gcu框架'] = check_torch_gcu_framework()
    results['ptex模块'] = check_ptex_module()
    results['共享库配置'] = check_shared_libraries()
    results['集成功能测试'] = perform_integration_test()
    
    # 生成总结报告
    generate_summary_report(results)
    
    # 返回整体验证结果
    return all(results.values())

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)