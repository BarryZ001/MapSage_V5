#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch_gcu分布式后端兼容性诊断脚本
专门诊断torch_gcu与PyTorch分布式后端的兼容性问题
"""

import os
import sys
import subprocess

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def check_torch_versions():
    """检查torch和torch_gcu版本"""
    print_section("检查PyTorch和torch_gcu版本")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        try:
            import torch_gcu  # type: ignore
            print(f"✅ torch_gcu版本: {torch_gcu.__version__}")  # type: ignore
            
            # 检查torch_gcu的分布式模块
            try:
                import torch_gcu.distributed  # type: ignore
                print("✅ torch_gcu.distributed模块可用")
                
                # 检查torch_gcu分布式后端
                if hasattr(torch_gcu.distributed, 'get_available_backends'):  # type: ignore
                    backends = torch_gcu.distributed.get_available_backends()  # type: ignore
                    print(f"✅ torch_gcu可用后端: {backends}")
                else:
                    print("⚠️ torch_gcu.distributed没有get_available_backends方法")
                    
            except ImportError as e:
                print(f"❌ torch_gcu.distributed导入失败: {e}")
                
        except ImportError as e:
            print(f"❌ torch_gcu导入失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ torch导入失败: {e}")
        return False
    
    return True

def check_distributed_backends():
    """详细检查分布式后端"""
    print_section("详细检查分布式后端")
    
    try:
        import torch.distributed as dist
        
        # 检查PyTorch原生后端
        native_backends = ['gloo', 'nccl', 'mpi']
        print("🔍 PyTorch原生后端:")
        
        for backend in native_backends:
            try:
                if hasattr(dist, 'is_backend_available'):
                    available = dist.is_backend_available(backend)  # type: ignore
                    status = "✅" if available else "❌"
                    print(f"  {status} {backend}: {'可用' if available else '不可用'}")
                else:
                    print(f"  ⚠️ {backend}: 无法检查（缺少is_backend_available方法）")
            except Exception as e:
                print(f"  ❌ {backend}: 检查失败 - {e}")
        
        # 检查ECCL后端
        print("\n🔍 ECCL后端:")
        try:
            # 方法1：直接检查
            if hasattr(dist, 'is_backend_available'):
                eccl_available = dist.is_backend_available('eccl')  # type: ignore
                print(f"  {'✅' if eccl_available else '❌'} eccl (is_backend_available): {'可用' if eccl_available else '不可用'}")
            
            # 方法2：检查注册的后端
            if hasattr(dist, 'Backend'):
                backend_enum = dist.Backend
                available_backends = [attr for attr in dir(backend_enum) if not attr.startswith('_')]
                print(f"  📋 注册的后端枚举: {available_backends}")
                
                if 'ECCL' in available_backends:
                    print("  ✅ ECCL在后端枚举中")
                else:
                    print("  ❌ ECCL不在后端枚举中")
            
            # 方法3：尝试直接初始化
            print("\n🧪 尝试初始化测试:")
            test_backends = ['gloo', 'eccl']
            
            for backend in test_backends:
                try:
                    # 设置环境变量
                    os.environ['MASTER_ADDR'] = '127.0.0.1'
                    os.environ['MASTER_PORT'] = '29503'
                    os.environ['RANK'] = '0'
                    os.environ['WORLD_SIZE'] = '1'
                    
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    
                    print(f"  🔧 测试{backend}后端初始化...")
                    dist.init_process_group(
                        backend=backend,
                        init_method='tcp://127.0.0.1:29503',
                        world_size=1,
                        rank=0
                    )
                    print(f"  ✅ {backend}后端初始化成功！")
                    
                    # 清理
                    dist.destroy_process_group()
                    
                except Exception as e:
                    print(f"  ❌ {backend}后端初始化失败: {e}")
                    print(f"     错误类型: {type(e).__name__}")
                    
        except Exception as e:
            print(f"❌ 分布式后端检查失败: {e}")
            
    except ImportError as e:
        print(f"❌ torch.distributed导入失败: {e}")

def check_torch_gcu_integration():
    """检查torch_gcu与PyTorch的集成"""
    print_section("检查torch_gcu与PyTorch集成")
    
    try:
        import torch
        import torch_gcu  # type: ignore
        
        # 检查GCU设备
        if hasattr(torch_gcu, 'device_count'):  # type: ignore
            device_count = torch_gcu.device_count()  # type: ignore
            print(f"✅ GCU设备数量: {device_count}")
        
        # 检查当前设备
        if hasattr(torch_gcu, 'current_device'):  # type: ignore
            current_device = torch_gcu.current_device()  # type: ignore
            print(f"✅ 当前GCU设备: {current_device}")
        
        # 检查torch_gcu是否修改了torch.distributed
        print("\n🔍 检查torch.distributed模块:")
        import torch.distributed as dist
        
        # 检查模块来源
        print(f"  📍 torch.distributed模块路径: {dist.__file__}")
        
        # 检查是否有torch_gcu的修改
        if hasattr(dist, '_torch_gcu_patched'):
            print("  ✅ 检测到torch_gcu对distributed的补丁")
        else:
            print("  ⚠️ 未检测到torch_gcu对distributed的补丁")
        
        # 检查后端注册
        if hasattr(dist, '_backend_registry'):
            registry = getattr(dist, '_backend_registry', {})
            print(f"  📋 后端注册表: {list(registry.keys()) if registry else '空'}")
        
    except Exception as e:
        print(f"❌ torch_gcu集成检查失败: {e}")

def check_environment_variables():
    """检查相关环境变量"""
    print_section("检查环境变量")
    
    important_vars = [
        'ECCL_ROOT',
        'TOPS_ECCL_ROOT', 
        'LD_LIBRARY_PATH',
        'PYTHONPATH',
        'TORCH_GCU_BACKEND',
        'GLOO_SOCKET_IFNAME',
        'NCCL_SOCKET_IFNAME'
    ]
    
    for var in important_vars:
        value = os.environ.get(var, '')
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 未设置")

def suggest_solutions():
    """提供解决方案建议"""
    print_section("解决方案建议")
    
    print("基于诊断结果，建议尝试以下解决方案：")
    print()
    print("1. 🔧 **检查torch_gcu版本兼容性**")
    print("   - 确认torch_gcu版本是否支持ECCL后端")
    print("   - 可能需要升级或降级torch_gcu版本")
    print()
    print("2. 🔧 **使用gloo后端作为替代**")
    print("   - 如果gloo后端可用，可以用它进行分布式训练")
    print("   - 修改训练脚本使用gloo而不是eccl")
    print()
    print("3. 🔧 **重新安装torch_gcu**")
    print("   - 卸载当前torch_gcu: pip uninstall torch-gcu")
    print("   - 重新安装支持ECCL的版本")
    print()
    print("4. 🔧 **检查TopsRider软件栈**")
    print("   - 确认TopsRider版本与torch_gcu兼容")
    print("   - 可能需要更新整个软件栈")
    print()
    print("5. 🔧 **联系燧原技术支持**")
    print("   - 提供torch版本、torch_gcu版本信息")
    print("   - 询问ECCL后端的正确配置方法")

def main():
    """主函数"""
    print("🔍 torch_gcu分布式后端兼容性诊断")
    print("=" * 60)
    
    try:
        # 检查版本
        if not check_torch_versions():
            print("❌ 基础导入失败，无法继续诊断")
            return False
        
        # 检查分布式后端
        check_distributed_backends()
        
        # 检查torch_gcu集成
        check_torch_gcu_integration()
        
        # 检查环境变量
        check_environment_variables()
        
        # 提供建议
        suggest_solutions()
        
        return True
        
    except Exception as e:
        print(f"❌ 诊断过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)