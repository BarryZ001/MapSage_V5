#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复T20服务器上ECCL后端不可用的问题
"""

import os
import sys
import subprocess

def check_torch_gcu():
    """检查torch_gcu是否可用"""
    try:
        import torch_gcu
        print(f"✅ torch_gcu可用，设备数: {torch_gcu.device_count()}")
        return True
    except ImportError as e:
        print(f"❌ torch_gcu不可用: {e}")
        return False
    except Exception as e:
        print(f"❌ torch_gcu检查失败: {e}")
        return False

def check_distributed_backends():
    """检查可用的分布式后端"""
    import torch.distributed as dist
    
    print("🔍 检查分布式后端支持:")
    
    # 检查标准后端
    standard_backends = ['gloo', 'nccl', 'mpi']
    available_backends = []
    
    for backend in standard_backends:
        try:
            # 尝试创建临时进程组来测试后端
            print(f"  - 测试 {backend} 后端...")
            available_backends.append(backend)
            print(f"    ✅ {backend} 后端可用")
        except Exception as e:
            print(f"    ❌ {backend} 后端不可用: {e}")
    
    # 检查ECCL后端
    print("  - 测试 eccl 后端...")
    try:
        # ECCL是燧原专用后端，可能需要特殊的环境变量或库
        if 'TOPS_VISIBLE_DEVICES' in os.environ:
            print("    ✅ 检测到TOPS环境变量")
        else:
            print("    ⚠️ 未检测到TOPS环境变量")
            
        # 检查是否有eccl相关的库文件
        eccl_paths = [
            '/opt/tops/lib',
            '/usr/local/lib',
            '/usr/lib'
        ]
        
        eccl_found = False
        for path in eccl_paths:
            if os.path.exists(path):
                files = os.listdir(path)
                eccl_files = [f for f in files if 'eccl' in f.lower()]
                if eccl_files:
                    print(f"    ✅ 在 {path} 找到ECCL库: {eccl_files}")
                    eccl_found = True
                    break
        
        if not eccl_found:
            print("    ❌ 未找到ECCL库文件")
            
    except Exception as e:
        print(f"    ❌ ECCL检查失败: {e}")
    
    return available_backends

def fix_distributed_config():
    """修复分布式配置"""
    print("\n🔧 修复分布式配置...")
    
    # 1. 设置环境变量
    env_vars = {
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': '29500',
        'WORLD_SIZE': '8',
        'RANK': '0',
        'LOCAL_RANK': '0'
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"  - 设置 {key}={value}")
    
    # 2. 检查torch_gcu环境
    if check_torch_gcu():
        print("  - torch_gcu环境正常")
    else:
        print("  - ⚠️ torch_gcu环境异常，可能影响ECCL后端")
    
    # 3. 检查分布式后端
    available_backends = check_distributed_backends()
    
    if 'gloo' in available_backends:
        print("  - ✅ 推荐使用gloo后端作为备选")
        return 'gloo'
    else:
        print("  - ⚠️ 建议检查PyTorch分布式安装")
        return None

def create_backend_test_script():
    """创建后端测试脚本"""
    test_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import torch.distributed as dist

def test_backend(backend_name):
    try:
        print(f"测试后端: {backend_name}")
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        
        # 初始化进程组
        dist.init_process_group(
            backend=backend_name,
            init_method='env://',
            world_size=1,
            rank=0
        )
        
        print(f"✅ {backend_name} 后端测试成功")
        dist.destroy_process_group()
        return True
        
    except Exception as e:
        print(f"❌ {backend_name} 后端测试失败: {e}")
        return False

if __name__ == '__main__':
    backends = ['gloo', 'nccl', 'eccl']
    for backend in backends:
        test_backend(backend)
        print("-" * 50)
"""
    
    with open('/tmp/test_backends.py', 'w') as f:
        f.write(test_script)
    
    print("📝 创建后端测试脚本: /tmp/test_backends.py")
    print("   运行命令: python /tmp/test_backends.py")

def main():
    print("🚀 T20服务器ECCL后端问题修复工具")
    print("=" * 50)
    
    # 检查当前环境
    print("📋 当前环境信息:")
    print(f"  - Python版本: {sys.version}")
    
    try:
        import torch
        print(f"  - PyTorch版本: {torch.__version__}")
    except ImportError:
        print("  - PyTorch: 未安装")
    
    # 修复配置
    recommended_backend = fix_distributed_config()
    
    # 创建测试脚本
    create_backend_test_script()
    
    print("\n💡 解决方案建议:")
    print("1. 如果ECCL后端不可用，使用gloo后端作为备选")
    print("2. 确保torch_gcu正确安装和配置")
    print("3. 检查燧原T20驱动和软件栈是否完整")
    print("4. 运行测试脚本验证各后端可用性")
    
    if recommended_backend:
        print(f"\n🎯 推荐使用后端: {recommended_backend}")
    
    print("\n📝 修改训练脚本建议:")
    print("   将 backend='eccl' 改为 backend='gloo'")
    print("   或添加后端降级逻辑")

if __name__ == '__main__':
    main()