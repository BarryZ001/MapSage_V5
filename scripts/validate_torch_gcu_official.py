#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于燧原官方文档的torch_gcu环境验证脚本
参考: https://support.enflame-tech.com/onlinedoc_dev_3.5/3-model/infer/torch_gcu/torch_gcu2.5/content/source/torch_gcu_user_guide.html
"""

import os
import sys
import subprocess

def check_basic_environment():
    """检查基础环境"""
    print("🔍 检查基础环境...")
    print(f"  - Python版本: {sys.version}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"  - PyTorch版本: {torch.__version__}")
        
        # 根据官方文档，torch_gcu支持PyTorch v2.5.1
        if torch.__version__.startswith('2.5'):
            print("  ✅ PyTorch版本兼容 (支持v2.5.1)")
        else:
            print(f"  ⚠️ PyTorch版本可能不兼容，推荐v2.5.1，当前: {torch.__version__}")
            
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False
    
    return True

def check_torch_gcu_official():
    """根据官方文档检查torch_gcu"""
    print("\n🔍 检查torch_gcu (官方方法)...")
    
    try:
        # 官方文档推荐的检查方法
        import torch
        import torch_gcu
        
        # 检查是否可用
        is_available = torch_gcu.is_available()
        print(f"  - torch_gcu.is_available(): {is_available}")
        
        if is_available:
            print("  ✅ torch_gcu安装成功，且设备可用")
            
            # 获取设备数量
            device_count = torch_gcu.device_count()
            print(f"  - 可用GCU设备数: {device_count}")
            
            # 测试基本操作
            try:
                # 创建tensor并移动到GCU
                a = torch.tensor([1, 2, 3]).gcu()
                b = torch.tensor([1, 2, 3]).to("gcu")
                c = torch.tensor([1, 2, 3], device="gcu")
                
                print("  ✅ GCU tensor创建成功")
                print(f"    - a.device: {a.device}")
                print(f"    - b.device: {b.device}")
                print(f"    - c.device: {c.device}")
                
                # 测试基本运算
                result = a + b
                print(f"  ✅ GCU运算测试成功: {result}")
                
            except Exception as e:
                print(f"  ❌ GCU操作测试失败: {e}")
                return False
                
        else:
            print("  ❌ torch_gcu不可用")
            return False
            
    except ImportError as e:
        print(f"  ❌ torch_gcu导入失败: {e}")
        print("  💡 请检查torch_gcu是否正确安装")
        return False
    except Exception as e:
        print(f"  ❌ torch_gcu检查失败: {e}")
        return False
    
    return True

def check_distributed_support():
    """检查分布式训练支持"""
    print("\n🔍 检查分布式训练支持...")
    
    try:
        import torch.distributed as dist
        
        # 检查ECCL后端支持
        print("  - 检查ECCL后端支持...")
        
        # 根据官方文档，需要将backend从nccl改为eccl
        print("  💡 根据官方文档:")
        print("    torch.distributed.init_process_group(backend='eccl', ...)")
        
        # 检查相关环境变量
        env_vars = [
            'TORCH_ECCL_AVOID_RECORD_STREAMS',
            'TORCH_ECCL_ASYNC_ERROR_HANDLING',
            'PYTORCH_GCU_ALLOC_CONF'
        ]
        
        print("  - 检查ECCL相关环境变量:")
        for var in env_vars:
            value = os.environ.get(var, '未设置')
            print(f"    - {var}: {value}")
        
        print("  ✅ 分布式配置检查完成")
        
    except Exception as e:
        print(f"  ❌ 分布式检查失败: {e}")
        return False
    
    return True

def check_amp_support():
    """检查AMP支持"""
    print("\n🔍 检查AMP支持...")
    
    try:
        import torch
        import torch_gcu
        
        # 根据官方文档，需要使用torch.gcu.amp而不是torch.cuda.amp
        print("  💡 根据官方文档:")
        print("    使用 torch.gcu.amp.autocast() 替代 torch.cuda.amp.autocast()")
        print("    使用 torch.gcu.amp.GradScaler() 替代 torch.cuda.amp.GradScaler()")
        
        # 测试AMP功能
        if torch_gcu.is_available():
            try:
                # 测试autocast
                with torch.gcu.amp.autocast():
                    a = torch.randn(10, 10).gcu()
                    b = torch.randn(10, 10).gcu()
                    c = torch.mm(a, b)
                
                print("  ✅ torch.gcu.amp.autocast() 测试成功")
                
                # 测试GradScaler
                scaler = torch.gcu.amp.GradScaler()
                print("  ✅ torch.gcu.amp.GradScaler() 创建成功")
                
            except Exception as e:
                print(f"  ❌ AMP功能测试失败: {e}")
                return False
        else:
            print("  ⚠️ torch_gcu不可用，跳过AMP测试")
            
    except Exception as e:
        print(f"  ❌ AMP检查失败: {e}")
        return False
    
    return True

def check_profiler_support():
    """检查Profiler支持"""
    print("\n🔍 检查Profiler支持...")
    
    try:
        import torch
        import torch_gcu
        
        if torch_gcu.is_available():
            # 根据官方文档的profiler示例
            size = (100, 100, 100)
            
            with torch.autograd.profiler.profile() as prof:
                a = torch.randn(size).gcu()
                b = torch.randn(size).gcu()
                for i in range(3):
                    c = a + b
            
            # 获取性能统计
            table = prof.table()
            print("  ✅ Profiler测试成功")
            print("  📊 性能统计表格 (前5行):")
            lines = table.split('\n')[:6]
            for line in lines:
                print(f"    {line}")
                
        else:
            print("  ⚠️ torch_gcu不可用，跳过Profiler测试")
            
    except Exception as e:
        print(f"  ❌ Profiler测试失败: {e}")
        return False
    
    return True

def check_dependencies():
    """检查依赖项"""
    print("\n🔍 检查燧原软件依赖...")
    
    # 根据官方文档的依赖项
    dependencies = [
        'topsruntime',
        'eccl', 
        'topsaten',
        'sdk'
    ]
    
    print("  💡 根据官方文档，需要以下依赖:")
    for dep in dependencies:
        print(f"    - {dep}")
    
    # 检查环境变量
    tops_vars = [
        'TOPS_VISIBLE_DEVICES',
        'TOPS_HOME',
        'LD_LIBRARY_PATH'
    ]
    
    print("  - 检查TOPS相关环境变量:")
    for var in tops_vars:
        value = os.environ.get(var, '未设置')
        if value != '未设置':
            print(f"    ✅ {var}: {value}")
        else:
            print(f"    ⚠️ {var}: 未设置")

def print_recommendations():
    """打印使用建议"""
    print("\n💡 使用建议 (基于官方文档):")
    print("=" * 50)
    
    print("1. 代码迁移:")
    print("   - 将 .cuda() 改为 .gcu()")
    print("   - 将 .to('cuda') 改为 .to('gcu')")
    print("   - 将 torch.cuda.xxx 改为 torch.gcu.xxx")
    
    print("\n2. 分布式训练:")
    print("   - 将 backend='nccl' 改为 backend='eccl'")
    print("   - 其他torch.distributed.xxx接口保持不变")
    
    print("\n3. AMP使用:")
    print("   - 将 torch.cuda.amp 改为 torch.gcu.amp")
    
    print("\n4. 环境变量设置:")
    print("   - PYTORCH_GCU_ALLOC_CONF='backend:topsMallocAsync'")
    print("   - TORCH_ECCL_AVOID_RECORD_STREAMS=false")
    print("   - TORCH_ECCL_ASYNC_ERROR_HANDLING=3")
    
    print("\n5. 调试信息:")
    print("   - ENFLAME_LOG_DEBUG_LEVEL='DEBUG'")
    print("   - ENFLAME_LOG_DEBUG_MOD='TORCH_GCU/OP,TORCH_GCU/FALLBACK'")

def main():
    """主函数"""
    print("🚀 torch_gcu环境验证 (基于燧原官方文档)")
    print("=" * 60)
    
    success = True
    
    # 检查各项功能
    success &= check_basic_environment()
    success &= check_torch_gcu_official()
    success &= check_distributed_support()
    success &= check_amp_support()
    success &= check_profiler_support()
    check_dependencies()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 torch_gcu环境验证通过！")
        print("✅ 可以开始使用燧原T20进行训练")
    else:
        print("❌ torch_gcu环境验证失败")
        print("💡 请参考上述错误信息进行修复")
    
    print_recommendations()
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)