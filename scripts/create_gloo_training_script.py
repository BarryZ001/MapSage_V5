#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建使用gloo后端的分布式训练脚本
由于ECCL后端不可用，使用gloo后端作为替代方案
"""

import os

def create_gloo_distributed_script():
    """创建使用gloo后端的分布式训练脚本"""
    
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用gloo后端的GCU分布式训练脚本
适用于torch_gcu环境中ECCL后端不可用的情况
"""

import os
import sys
import torch
import torch.distributed as dist
import torch_gcu  # type: ignore

def setup_distributed_gloo():
    """设置gloo后端的分布式环境"""
    print("🔧 设置gloo后端分布式环境...")
    
    # 设置环境变量
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    print(f"📊 分布式参数:")
    print(f"  - RANK: {rank}")
    print(f"  - WORLD_SIZE: {world_size}")
    print(f"  - MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"  - MASTER_PORT: {os.environ['MASTER_PORT']}")
    
    try:
        # 初始化分布式进程组（使用gloo后端）
        print("🚀 初始化gloo后端...")
        dist.init_process_group(
            backend='gloo',
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world_size,
            rank=rank
        )
        
        print("✅ gloo后端初始化成功！")
        
        # 设置GCU设备
        if torch_gcu.is_available():  # type: ignore
            device_count = torch_gcu.device_count()  # type: ignore
            local_rank = rank % device_count
            torch_gcu.set_device(local_rank)  # type: ignore
            print(f"✅ 设置GCU设备: {local_rank}")
        
        return True
        
    except Exception as e:
        print(f"❌ gloo后端初始化失败: {e}")
        return False

def test_distributed_operations():
    """测试分布式操作"""
    print("\\n🧪 测试分布式操作...")
    
    try:
        # 创建测试张量
        if torch_gcu.is_available():  # type: ignore
            device = torch_gcu.current_device()  # type: ignore
            tensor = torch.tensor([float(dist.get_rank())], device=f'gcu:{device}')
        else:
            tensor = torch.tensor([float(dist.get_rank())])
        
        print(f"📊 进程 {dist.get_rank()} 的原始张量: {tensor.item()}")
        
        # 执行all_reduce操作
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"✅ 进程 {dist.get_rank()} 的all_reduce结果: {tensor.item()}")
        
        # 执行broadcast操作
        if dist.get_rank() == 0:
            broadcast_tensor = torch.tensor([42.0])
        else:
            broadcast_tensor = torch.tensor([0.0])
            
        if torch_gcu.is_available():  # type: ignore
            device = torch_gcu.current_device()  # type: ignore
            broadcast_tensor = broadcast_tensor.to(f'gcu:{device}')
        
        dist.broadcast(broadcast_tensor, src=0)
        print(f"✅ 进程 {dist.get_rank()} 的broadcast结果: {broadcast_tensor.item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分布式操作测试失败: {e}")
        return False

def cleanup_distributed():
    """清理分布式环境"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ 分布式进程组已清理")
    except Exception as e:
        print(f"⚠️ 清理分布式环境时出错: {e}")

def main():
    """主函数"""
    print("🚀 gloo后端分布式训练测试")
    print("=" * 50)
    
    try:
        # 设置分布式环境
        if not setup_distributed_gloo():
            return False
        
        # 测试分布式操作
        success = test_distributed_operations()
        
        # 清理环境
        cleanup_distributed()
        
        if success:
            print("\\n🎉 gloo后端分布式测试成功！")
            print("💡 可以使用gloo后端进行分布式训练")
        else:
            print("\\n❌ gloo后端分布式测试失败")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        cleanup_distributed()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
'''
    
    # 写入脚本文件
    script_path = "scripts/test_gloo_distributed.py"
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 设置执行权限
        os.chmod(script_path, 0o755)
        print(f"✅ gloo分布式测试脚本已生成: {script_path}")
        return script_path
        
    except Exception as e:
        print(f"❌ 生成gloo分布式测试脚本失败: {e}")
        return None

def create_training_launcher():
    """创建分布式训练启动脚本"""
    
    launcher_content = '''#!/bin/bash
# gloo后端分布式训练启动脚本

echo "🚀 启动gloo后端分布式训练..."

# 设置环境变量
source scripts/setup_eccl_env.sh

# 检查参数
SCRIPT_PATH=${1:-"scripts/train_distributed_gcu_fixed.py"}
NUM_GPUS=${2:-8}

echo "📋 训练参数:"
echo "  - 训练脚本: $SCRIPT_PATH"
echo "  - GPU数量: $NUM_GPUS"

# 使用torchrun启动分布式训练（强制使用gloo后端）
export TORCH_DISTRIBUTED_BACKEND=gloo

torchrun \\
    --nproc_per_node=$NUM_GPUS \\
    --master_addr=127.0.0.1 \\
    --master_port=29500 \\
    $SCRIPT_PATH \\
    --backend=gloo \\
    --launcher=pytorch

echo "🎯 分布式训练完成！"
'''
    
    launcher_path = "scripts/start_gloo_training.sh"
    try:
        with open(launcher_path, 'w', encoding='utf-8') as f:
            f.write(launcher_content)
        
        os.chmod(launcher_path, 0o755)
        print(f"✅ gloo训练启动脚本已生成: {launcher_path}")
        return launcher_path
        
    except Exception as e:
        print(f"❌ 生成gloo训练启动脚本失败: {e}")
        return None

def main():
    """主函数"""
    print("🔧 创建gloo后端分布式训练脚本")
    print("=" * 50)
    
    # 创建测试脚本
    test_script = create_gloo_distributed_script()
    
    # 创建启动脚本
    launcher_script = create_training_launcher()
    
    print("\\n📋 使用说明:")
    print("1. 运行诊断脚本: python3 scripts/diagnose_torch_gcu_backends.py")
    print("2. 测试gloo后端: python3 scripts/test_gloo_distributed.py")
    print("3. 启动分布式训练: bash scripts/start_gloo_training.sh")
    print("\\n💡 由于ECCL后端不可用，建议使用gloo后端进行分布式训练")

if __name__ == '__main__':
    main()