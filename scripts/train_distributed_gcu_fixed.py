#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目根目录到Python路径
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.runner import Runner

# 尝试导入torch_gcu和ptex
try:
    import torch_gcu
    import torch_gcu.distributed as gcu_dist  # 导入torch_gcu分布式模块
    print(f"✅ torch_gcu导入成功，可用设备数: {torch_gcu.device_count()}")
    print("✅ torch_gcu.distributed模块导入成功")
    USE_GCU_DISTRIBUTED = True
except ImportError as e:
    print(f"⚠️ torch_gcu导入失败: {e}")
    torch_gcu = None
    gcu_dist = None
    USE_GCU_DISTRIBUTED = False

try:
    import ptex
    print("✅ ptex导入成功")
except ImportError as e:
    print(f"⚠️ ptex导入失败: {e}")
    ptex = None

# 导入MMSeg相关模块
try:
    import mmseg  # type: ignore
    from mmseg.models import *  # type: ignore
    from mmseg.datasets import *  # type: ignore
except ImportError as e:
    print(f"⚠️ 模块导入失败: {e}")

# 导入自定义模块
try:
    import mmseg_custom.models
    import mmseg_custom.datasets
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"⚠️ 自定义模块导入失败: {e}")

def setup_distributed():
    """设置分布式训练环境"""
    # 获取分布式训练参数
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"🌍 分布式训练设置: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    # 如果是多进程分布式训练，初始化进程组
    if world_size > 1:
        # 设置分布式后端 - 统一使用gloo后端
        os.environ['MMENGINE_DDP_BACKEND'] = 'gloo'
        print("🔧 设置MMEngine DDP后端为gloo")
        
        # 初始化分布式进程组 - 统一使用gloo后端
        if not dist.is_initialized():
            backend = 'gloo'
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}",
                world_size=world_size,
                rank=rank
            )
            print(f"✅ 分布式进程组初始化完成 - Backend: {backend}")
    
    # 设置GCU设备
    if torch_gcu is not None:
        try:
            torch_gcu.set_device(local_rank)
            print(f"🔧 设置GCU设备: {local_rank}")
        except Exception as e:
            print(f"⚠️ 设置GCU设备失败: {e}")
    
    # 设置环境变量
    os.environ['TOPS_VISIBLE_DEVICES'] = str(local_rank)
    print(f"🔧 设置TOPS_VISIBLE_DEVICES: {local_rank}")
    
    # 禁用CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("🔧 禁用CUDA_VISIBLE_DEVICES")
    
    print(f"✅ 分布式环境初始化完成 - Rank {rank}/{world_size}")
    return rank, local_rank, world_size

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        if USE_GCU_DISTRIBUTED and gcu_dist is not None:
            # 使用torch_gcu.distributed.destroy_process_group
            try:
                gcu_dist.destroy_process_group()
                print("✅ 使用torch_gcu.distributed.destroy_process_group清理完成")
            except Exception as e:
                print(f"⚠️ torch_gcu分布式清理失败: {e}")
                # 回退到标准方法
                dist.destroy_process_group()
        else:
            dist.destroy_process_group()
            print("✅ 分布式进程组清理完成")

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation distributed training script for GCU')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch', help='job launcher')
    args = parser.parse_args()

    print("📦 正在初始化分布式MMSegmentation训练...")
    
    # 设置分布式环境
    rank, local_rank, world_size = setup_distributed()
    
    try:
        # 从文件加载配置
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"配置文件不存在: {args.config}")
        
        print(f"📝 加载配置文件: {args.config}")
        cfg = Config.fromfile(args.config)
        
        # 设置工作目录
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = './work_dirs'
        
        # 设置GCU设备
        device_str = None
        if ptex is not None:
            try:
                device_str = "xla"
                print(f"🔧 设置GCU设备为: {device_str}")
            except Exception as e:
                print(f"⚠️ 设置ptex设备失败: {e}")
        elif torch_gcu is not None:
            try:
                device_str = f"gcu:{local_rank}"
                print(f"🔧 设置GCU设备为: {device_str}")
            except Exception as e:
                print(f"⚠️ 设置torch_gcu设备失败: {e}")
        
        # 更新配置以支持分布式训练
        if world_size > 1:
            cfg.launcher = args.launcher
        else:
            cfg.launcher = 'none'
            print("🔧 单进程模式，禁用分布式")
        
        # 调整batch size（每个进程的batch size）
        if hasattr(cfg, 'train_dataloader') and 'batch_size' in cfg.train_dataloader:
            original_batch_size = cfg.train_dataloader.batch_size
            # 保持总batch size不变，分配到各个进程
            cfg.train_dataloader.batch_size = max(1, original_batch_size // world_size)
            print(f"📊 调整batch size: {original_batch_size} -> {cfg.train_dataloader.batch_size} (per process)")
        
        print(f"📁 工作目录: {cfg.work_dir}")
        print(f"🚀 启动训练 - Rank {rank}/{world_size}")
        
        # 创建Runner
        runner = Runner.from_cfg(cfg)
        
        # 在Runner创建后，手动将模型移动到GCU设备
        if device_str is not None and hasattr(runner, 'model'):
            print(f"🔧 将模型移动到设备: {device_str}")
            try:
                if ptex is not None:
                    # 使用ptex设备
                    device = ptex.device("xla")
                    runner.model = runner.model.to(device)
                    print(f"✅ 模型已移动到ptex设备: {device}")
                elif torch_gcu is not None:
                    # 使用torch_gcu设备
                    device = torch_gcu.device(local_rank)
                    runner.model = runner.model.to(device)
                    print(f"✅ 模型已移动到GCU设备: {device}")
            except Exception as e:
                print(f"⚠️ 移动模型到设备失败: {e}")
                print("🔄 尝试使用CPU训练")
        
        runner.train()
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        raise
    finally:
        # 清理分布式环境
        cleanup_distributed()

if __name__ == '__main__':
    main()