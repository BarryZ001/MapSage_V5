#!/usr/bin/env python3
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
    print("\n🧪 测试分布式操作...")
    
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
            print("\n🎉 gloo后端分布式测试成功！")
            print("💡 可以使用gloo后端进行分布式训练")
        else:
            print("\n❌ gloo后端分布式测试失败")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        cleanup_distributed()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
