#!/usr/bin/env python3
"""
8卡分布式训练脚本 - 燧原T20 GCU版本
支持DINOv3 + MMRS-1M数据集的8卡分布式训练
"""

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

# 尝试导入GCU相关库
try:
    import torch_gcu
    print(f"✅ torch_gcu导入成功，可用设备数: {torch_gcu.device_count()}")
except ImportError as e:
    print(f"⚠️ torch_gcu导入失败: {e}")
    torch_gcu = None

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
    """初始化分布式训练环境"""
    # 从环境变量获取分布式训练参数
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"🌍 分布式训练参数:")
    print(f"  - WORLD_SIZE: {world_size}")
    print(f"  - RANK: {rank}")
    print(f"  - LOCAL_RANK: {local_rank}")
    
    if world_size > 1:
        # 初始化分布式进程组
        if not dist.is_initialized():
            # 设置分布式后端
            backend = 'gloo'  # GCU环境使用gloo后端
            init_method = 'env://'
            
            print(f"🔧 初始化分布式进程组:")
            print(f"  - Backend: {backend}")
            print(f"  - Init method: {init_method}")
            
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank
            )
            
            print(f"✅ 分布式进程组初始化成功")
        else:
            print("✅ 分布式进程组已初始化")
    
    return world_size, rank, local_rank

def main():
    parser = argparse.ArgumentParser(description='8卡分布式训练脚本')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                       default='pytorch', help='分布式启动器')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程rank')
    args = parser.parse_args()
    
    print(f"🚀 启动8卡分布式训练")
    print(f"📄 配置文件: {args.config}")
    print(f"🔧 启动器: {args.launcher}")
    
    # 设置分布式环境
    world_size, rank, local_rank = setup_distributed()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 设置工作目录
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 设置日志目录
    log_dir = os.path.join(cfg.work_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 更新配置以支持分布式训练
    if world_size > 1:
        cfg.launcher = args.launcher
        print(f"🔧 启用分布式训练，launcher: {args.launcher}")
        # 配置GCU设备，让MMEngine自动处理分布式
        cfg.device = 'gcu'
        print(f"🔧 配置GCU设备，world_size: {world_size}")
    else:
        cfg.launcher = 'none'
        print("🔧 单进程模式，禁用分布式")
        # 单卡训练配置
        cfg.device = 'gcu'
        print(f"🔧 配置单卡GCU设备")
    
    # 调整batch size（每个进程的batch size）
    if hasattr(cfg, 'train_dataloader') and 'batch_size' in cfg.train_dataloader:
        original_batch_size = cfg.train_dataloader.batch_size
        # 8卡分布式训练，每卡保持配置的batch_size
        print(f"📊 每卡batch size: {original_batch_size}")
        print(f"📊 总batch size: {original_batch_size * world_size}")
    
    print(f"📁 工作目录: {cfg.work_dir}")
    print(f"🚀 启动训练 - Rank {rank}/{world_size}")
    
    # 设置GCU设备
    if torch_gcu is not None:
        torch_gcu.set_device(local_rank)
        print(f"🔧 设置当前进程GCU设备: {local_rank}")
        
        # 设置默认设备类型为GCU，确保新创建的tensor都在GCU上
        try:
            torch.set_default_device(f'gcu:{local_rank}')
            print(f"🔧 设置默认tensor设备为: gcu:{local_rank}")
        except AttributeError:
            # 如果torch版本不支持set_default_device，跳过
            print(f"⚠️ torch版本不支持set_default_device，跳过设置")
    
    # 创建Runner并开始训练
    print("🚀 创建Runner并开始训练...")
    runner = Runner.from_cfg(cfg)
    
    # 确保模型移动到正确的GCU设备
    if torch_gcu is not None and hasattr(runner, 'model'):
        device = f'gcu:{local_rank}'
        print(f"🔧 手动移动模型到设备: {device}")
        runner.model = runner.model.to(device)
        
        # 验证模型参数设备
        for name, param in runner.model.named_parameters():
            if param.device.type != 'gcu':
                print(f"⚠️ 参数 {name} 仍在 {param.device}，手动移动到 {device}")
                param.data = param.data.to(device)
            break  # 只检查第一个参数作为示例
    
    print("✅ Runner创建完成，模型已配置到GCU设备")
    
    runner.train()
    
    # 清理分布式环境
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 分布式进程组已清理")

if __name__ == '__main__':
    main()