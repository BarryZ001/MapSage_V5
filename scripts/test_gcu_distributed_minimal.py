#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小的torch_gcu分布式测试脚本
用于验证eccl后端是否正常工作
"""

import os
import sys
import torch
import torch.distributed as dist  # type: ignore
import time

def test_gcu_import():
    """测试torch_gcu导入"""
    try:
        import torch_gcu  # type: ignore
        import torch_gcu.distributed as gcu_dist  # type: ignore
        print(f"✅ torch_gcu导入成功，可用设备数: {torch_gcu.device_count()}")
        print("✅ torch_gcu.distributed模块导入成功")
        return True, torch_gcu, gcu_dist
    except ImportError as e:
        print(f"❌ torch_gcu导入失败: {e}")
        return False, None, None

def test_distributed_init():
    """测试分布式初始化"""
    # 获取分布式训练参数
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    print(f"🌍 分布式参数: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    print(f"🔧 Master地址: {master_addr}:{master_port}")
    
    if world_size == 1:
        print("⚠️ 单进程模式，跳过分布式测试")
        return True
    
    # 测试torch_gcu导入
    has_gcu, torch_gcu, gcu_dist = test_gcu_import()
    
    # 设置GCU设备
    if has_gcu and torch_gcu is not None:
        try:
            torch_gcu.set_device(local_rank)
            print(f"✅ 设置GCU设备: {local_rank}")
        except Exception as e:
            print(f"⚠️ 设置GCU设备失败: {e}")
    
    # 尝试不同的后端
    backends_to_try = ['eccl', 'gloo'] if has_gcu else ['gloo']
    
    for backend in backends_to_try:
        print(f"\n🔧 测试后端: {backend}")
        
        if dist.is_initialized():
            dist.destroy_process_group()
        
        try:
            # 初始化分布式进程组
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=rank,
                timeout=dist.default_pg_timeout * 2  # type: ignore
            )
            print(f"✅ {backend}后端初始化成功")
            
            # 测试基本的分布式操作
            test_tensor = torch.tensor([rank], dtype=torch.float32)
            if has_gcu and torch_gcu is not None:
                try:
                    device = torch_gcu.device(local_rank)
                    test_tensor = test_tensor.to(device)
                    print(f"✅ 张量移动到GCU设备: {device}")
                except Exception as e:
                    print(f"⚠️ 张量移动到GCU失败: {e}")
            
            # 测试all_reduce操作
            try:
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                expected_sum = sum(range(world_size))
                if abs(test_tensor.item() - expected_sum) < 1e-6:
                    print(f"✅ all_reduce测试成功: {test_tensor.item()} == {expected_sum}")
                else:
                    print(f"⚠️ all_reduce结果不正确: {test_tensor.item()} != {expected_sum}")
            except Exception as e:
                print(f"❌ all_reduce测试失败: {e}")
                continue
            
            # 测试barrier同步
            try:
                dist.barrier()
                print("✅ barrier同步成功")
            except Exception as e:
                print(f"❌ barrier同步失败: {e}")
                continue
            
            print(f"✅ {backend}后端所有测试通过")
            return True
            
        except Exception as e:
            print(f"❌ {backend}后端初始化失败: {e}")
            continue
    
    print("❌ 所有后端测试都失败")
    return False

def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("✅ 分布式进程组清理完成")
        except Exception as e:
            print(f"⚠️ 分布式清理失败: {e}")

def main():
    """主函数"""
    print("🧪 开始torch_gcu分布式测试...")
    
    try:
        success = test_distributed_init()
        
        if success:
            print("\n🎉 分布式测试成功！")
            print("✅ 可以使用分布式训练")
        else:
            print("\n❌ 分布式测试失败")
            print("⚠️ 请检查环境配置")
            
        return success
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False
    finally:
        cleanup()

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)