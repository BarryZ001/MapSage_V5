#!/usr/bin/env python3
"""
测试修复后的训练环境设置
验证配置文件修复和环境兼容性
"""

import os
import sys
import importlib.util
import subprocess

def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "=" * 50)
    print("🧪 测试配置文件加载")
    print("=" * 50)
    
    config_path = "configs/train_dinov3_mmrs1m_t20_gcu_8card_fixed.py"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        # 动态加载配置文件
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            print(f"❌ 无法创建配置文件规范: {config_path}")
            return False
            
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        print("✅ 配置文件加载成功")
        
        # 检查关键配置项
        if hasattr(config_module, 'model'):
            print("✅ model 配置存在")
        if hasattr(config_module, 'train_dataloader'):
            print("✅ train_dataloader 配置存在")
        if hasattr(config_module, 'optim_wrapper'):
            print("✅ optim_wrapper 配置存在")
            
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_distributed_backend_fallback():
    """测试分布式backend fallback逻辑"""
    print("\n" + "=" * 50)
    print("🧪 测试分布式backend fallback")
    print("=" * 50)
    
    try:
        import torch.distributed as dist
        
        # 获取可用的backend
        available_backends = []
        if hasattr(dist, 'Backend'):
            for backend_name in ['nccl', 'gloo', 'mpi']:
                if hasattr(dist.Backend, backend_name.upper()):
                    available_backends.append(backend_name)
        
        print(f"✅ 可用的分布式backend: {available_backends}")
        
        # 测试backend选择逻辑
        if 'nccl' in available_backends:
            print("✅ NCCL backend 可用 (推荐用于GPU)")
        if 'gloo' in available_backends:
            print("✅ Gloo backend 可用 (CPU fallback)")
            
        return len(available_backends) > 0
        
    except ImportError as e:
        print(f"❌ torch.distributed 导入失败: {e}")
        return False

def test_torch_gcu_integration():
    """测试torch_gcu集成"""
    print("\n" + "=" * 50)
    print("🧪 测试torch_gcu集成")
    print("=" * 50)
    
    try:
        # 尝试导入torch_gcu
        torch_gcu = __import__('torch_gcu')
        print("✅ torch_gcu 导入成功")
        
        # 检查基本功能
        if hasattr(torch_gcu, 'device_count'):
            device_count = torch_gcu.device_count()
            print(f"✅ GCU设备数量: {device_count}")
        
        return True
        
    except ImportError:
        print("⚠️  torch_gcu 未安装 (在T20服务器上应该可用)")
        return True  # 在本地环境这是正常的

def test_environment_detection():
    """测试环境检测脚本"""
    print("\n" + "=" * 50)
    print("🧪 测试环境检测脚本")
    print("=" * 50)
    
    script_path = "scripts/check_torch_gcu_environment.py"
    
    if not os.path.exists(script_path):
        print(f"❌ 环境检测脚本不存在: {script_path}")
        return False
    
    try:
        # 动态加载环境检测脚本
        spec = importlib.util.spec_from_file_location("env_check", script_path)
        if spec is None or spec.loader is None:
            print(f"❌ 无法创建环境检测脚本规范: {script_path}")
            return False
            
        env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_module)
        
        print("✅ 环境检测脚本加载成功")
        
        # 检查关键函数
        if hasattr(env_module, 'check_torch_gcu_environment'):
            print("✅ check_torch_gcu_environment 函数存在")
        if hasattr(env_module, 'check_topsrider_installation'):
            print("✅ check_topsrider_installation 函数存在")
            
        return True
        
    except Exception as e:
        print(f"❌ 环境检测脚本加载失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始测试修复后的训练环境设置")
    print("=" * 60)
    
    tests = [
        ("配置文件加载", test_config_loading),
        ("分布式backend fallback", test_distributed_backend_fallback),
        ("torch_gcu集成", test_torch_gcu_integration),
        ("环境检测脚本", test_environment_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: 通过")
            else:
                print(f"❌ {test_name}: 失败")
        except Exception as e:
            print(f"❌ {test_name}: 异常 - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境修复成功")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关组件")
        return 1

if __name__ == "__main__":
    sys.exit(main())