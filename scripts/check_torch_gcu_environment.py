#!/usr/bin/env python3
"""
torch_gcu和eccl环境检测脚本
用于诊断燧原GCU训练环境的配置状态
"""

import os
import sys
import importlib.util
import torch
import torch.distributed as dist

def check_torch_environment():
    """检查PyTorch基础环境"""
    print("=" * 60)
    print("🔍 PyTorch 基础环境检查")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"分布式训练可用: {dist.is_available()}")
    if dist.is_available():
        try:
            # 安全地获取backend列表
            backends = []
            for attr_name in dir(dist.Backend):
                if not attr_name.startswith('_'):
                    backends.append(attr_name.lower())
            print(f"支持的分布式backend: {backends}")
        except Exception as e:
            print(f"获取backend列表失败: {e}")

def check_torch_gcu():
    """检查torch_gcu模块"""
    print("\n" + "=" * 60)
    print("🔍 torch_gcu 模块检查")
    print("=" * 60)
    
    try:
        spec = importlib.util.find_spec("torch_gcu")
        if spec is None:
            print("❌ torch_gcu 模块未找到")
            return False
        
        print(f"✅ torch_gcu 模块路径: {spec.origin}")
        
        # 尝试动态导入避免静态分析错误
        try:
            torch_gcu_module = __import__('torch_gcu')
            print("✅ torch_gcu 导入成功")
            
            # 检查GCU设备
            if hasattr(torch_gcu_module, 'device_count'):
                device_count = torch_gcu_module.device_count()
                print(f"GCU设备数量: {device_count}")
        except ImportError:
            print("⚠️ torch_gcu 模块存在但导入失败（可能在非GCU环境中）")
        except Exception as e:
            print(f"⚠️ torch_gcu 导入异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ torch_gcu 检查出错: {e}")
        return False

def check_eccl():
    """检查eccl模块"""
    print("\n" + "=" * 60)
    print("🔍 eccl 模块检查")
    print("=" * 60)
    
    try:
        spec = importlib.util.find_spec("eccl")
        if spec is None:
            print("❌ eccl 模块未找到")
            return False
        
        print(f"✅ eccl 模块路径: {spec.origin}")
        
        # 尝试动态导入避免静态分析错误
        try:
            eccl_module = __import__('eccl')
            print("✅ eccl 导入成功")
        except ImportError:
            print("⚠️ eccl 模块存在但导入失败（可能在非GCU环境中）")
        except Exception as e:
            print(f"⚠️ eccl 导入异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ eccl 检查出错: {e}")
        return False

def check_topsrider_installation():
    """检查TopsRider软件栈安装状态"""
    print("\n" + "=" * 60)
    print("🔍 TopsRider 软件栈检查")
    print("=" * 60)
    
    # 检查TopsRider相关环境变量
    topsrider_vars = [
        'TOPS_INSTALL_PATH',
        'TOPS_RUNTIME_PATH', 
        'TOPSRIDER_PATH',
        'GCU_DEVICE_PATH'
    ]
    
    found_vars = []
    for var in topsrider_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
            found_vars.append(var)
        else:
            print(f"❌ {var}: 未设置")
    
    # 检查TopsRider安装目录
    possible_paths = [
        '/usr/local/topsrider',
        '/opt/topsrider',
        '/home/topsrider',
        os.path.expanduser('~/topsrider')
    ]
    
    topsrider_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ TopsRider安装目录找到: {path}")
            topsrider_found = True
            
            # 检查关键文件
            key_files = ['bin', 'lib', 'include']
            for key_file in key_files:
                file_path = os.path.join(path, key_file)
                if os.path.exists(file_path):
                    print(f"  ✅ {key_file}/ 目录存在")
                else:
                    print(f"  ❌ {key_file}/ 目录缺失")
            break
    
    if not topsrider_found:
        print("❌ 未找到TopsRider安装目录")
        print("💡 建议检查TopsRider是否正确安装")
        print("💡 参考官方文档使用 --with-vgcu 或 --python 参数重新安装")
    
    return len(found_vars) > 0 or topsrider_found

def check_installation_completeness():
    """检查TopsRider安装完整性"""
    print("\n" + "=" * 60)
    print("🔍 TopsRider 安装完整性检查")
    print("=" * 60)
    
    # 检查关键Python模块
    modules_to_check = [
        ('torch_gcu', '✅ torch_gcu (PyTorch GCU支持)', True),
        ('eccl', '✅ eccl (分布式通信库)', True),
        ('horovod', '🔶 horovod (分布式训练框架)', False),
        ('tops_models', '🔶 tops_models (模型库)', False),
    ]
    
    installed_modules = []
    missing_critical = []
    
    for module_name, description, is_critical in modules_to_check:
        try:
            module = __import__(module_name)
            print(f"✅ {description}")
            if hasattr(module, '__version__'):
                print(f"   版本: {module.__version__}")
            installed_modules.append(module_name)
        except ImportError:
            if is_critical:
                print(f"❌ {description} - 未安装")
                missing_critical.append(module_name)
            else:
                print(f"⚠️  {description} - 未安装 (可选)")
    
    # 检查GCU设备数量
    try:
        torch_gcu = __import__('torch_gcu')
        if hasattr(torch_gcu, 'device_count'):
            device_count = torch_gcu.device_count()
            print(f"🎯 GCU设备数量: {device_count}")
            if device_count > 0:
                print("✅ GCU设备可用，支持分布式训练")
            else:
                print("⚠️  未检测到GCU设备")
    except:
        pass
    
    return len(missing_critical) == 0, installed_modules, missing_critical

def provide_installation_guidance():
    """提供安装指导"""
    print("\n" + "=" * 60)
    print("📋 安装指导")
    print("=" * 60)
    
    print("🚀 推荐安装方法:")
    print("1. 使用自动安装脚本 (推荐):")
    print("   sudo bash scripts/install_topsrider_complete.sh")
    print()
    print("2. 手动安装关键组件:")
    print("   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C topsplatform")
    print("   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl")
    print("   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.8")
    print()
    print("3. 安装后设置环境变量:")
    print("   source /etc/profile.d/topsrider.sh")
    print()
    print("📖 详细安装指南:")
    print("   docs/TopsRider_Complete_Installation_Guide.md")

def check_torch_gcu_environment():
    """检查torch_gcu环境状态"""
    check_torch_environment()
    torch_gcu_ok = check_torch_gcu()
    eccl_ok = check_eccl()
    topsrider_ok = check_topsrider_installation()
    
    # 新增：检查安装完整性
    installation_complete, installed_modules, missing_critical = check_installation_completeness()
    
    # 6. 环境变量检查
    print("\n" + "-" * 40)
    print("🔍 检查环境变量")
    print("-" * 40)
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"📚 LD_LIBRARY_PATH: {ld_library_path}")
    
    python_path = os.environ.get('PYTHONPATH', '')
    print(f"🐍 PYTHONPATH: {python_path}")
    
    # 7. TopsRider安装目录检查
    print("\n" + "-" * 40)
    print("🔍 检查TopsRider安装目录")
    print("-" * 40)
    
    possible_paths = [
        '/usr/local/topsrider',
        '/opt/tops/lib',
        '/usr/local/tops',
        '/opt/topsrider'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到目录: {path}")
            try:
                files = os.listdir(path)[:10]  # 只显示前10个文件
                print(f"   📁 内容示例: {files}")
            except PermissionError:
                print(f"   ⚠️ 无权限访问目录内容")
        else:
            print(f"❌ 目录不存在: {path}")
    
    # 8. 总结建议
    print("\n" + "=" * 60)
    print("📋 诊断总结与建议")
    print("=" * 60)
    
    if not torch_gcu_ok and not eccl_ok:
        print("❌ 关键问题: torch_gcu和eccl都未安装")
        print("🔧 建议: 使用TopsRider完整安装脚本")
        print("   sudo bash scripts/install_topsrider_complete.sh")
        provide_installation_guidance()
    elif not torch_gcu_ok:
        print("❌ 关键问题: torch_gcu未安装")
        print("🔧 建议: 安装torch_gcu组件")
        print("   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.8")
    elif not eccl_ok:
        print("❌ 关键问题: eccl未安装")
        print("🔧 建议: 安装eccl分布式通信库")
        print("   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl")
    elif not installation_complete:
        print("⚠️  部分组件未安装")
        print(f"❌ 缺失关键组件: {', '.join(missing_critical)}")
        provide_installation_guidance()
    else:
        print("✅ 环境检查通过！")
        print("🎉 TopsRider软件栈已正确安装")
        print("💡 可以开始进行分布式训练")
        
        # 显示已安装组件
        print(f"\n📦 已安装组件: {', '.join(installed_modules)}")
    
    return torch_gcu_ok and eccl_ok and installation_complete

if __name__ == "__main__":
    success = check_torch_gcu_environment()
    sys.exit(0 if success else 1)