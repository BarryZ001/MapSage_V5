# scripts/train_distributed.py - 分布式训练脚本

import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加当前目录到Python路径
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dist import init_dist

# 导入GCU支持
try:
    import torch_gcu
    print(f"✅ torch_gcu导入成功，可用设备数: {torch_gcu.device_count()}")
except ImportError as e:
    print(f"⚠️ torch_gcu导入失败: {e}")
    torch_gcu = None

# 导入mmseg来触发所有注册
try:
    import mmseg  # type: ignore
    from mmseg.models import *  # type: ignore
    from mmseg.datasets import *  # type: ignore
except ImportError as e:
    print(f"⚠️ 模块导入失败: {e}")

# 导入我们的自定义模块
try:
    import mmseg_custom.models
    import mmseg_custom.datasets
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"⚠️ 自定义模块导入失败: {e}")

def setup_distributed():
    """设置分布式训练环境"""
    # 获取环境变量
    world_size = int(os.environ.get('WORLD_SIZE', 8))  # 默认8个GCU
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"🌍 分布式训练设置: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    # 设置GCU设备
    if torch_gcu is not None:
        # 设置当前进程使用的GCU设备
        torch_gcu.set_device(local_rank)
        print(f"🔧 设置GCU设备: {local_rank}")
        
        # 设置环境变量确保使用GCU
        os.environ['TOPS_VISIBLE_DEVICES'] = str(local_rank)
        print(f"🔧 设置TOPS_VISIBLE_DEVICES: {local_rank}")
        
        # 禁用CUDA相关的环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("🔧 禁用CUDA_VISIBLE_DEVICES")
    else:
        print("⚠️ torch_gcu不可用，可能会使用CPU训练")
    
    # 初始化分布式环境 - 使用自定义方式避免CUDA调用
    if not dist.is_initialized():
        # 设置分布式后端为gloo，避免NCCL
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        # 直接使用torch.distributed.init_process_group而不是MMEngine的init_dist
        dist.init_process_group(
            backend='eccl',  # 使用ECCL后端，支持GCU
            rank=rank,
            world_size=world_size
        )
        print(f"✅ 分布式环境初始化完成 - Rank {rank}/{world_size}")
    
    return rank, local_rank, world_size

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation distributed training script')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch', help='job launcher')
    args = parser.parse_args()

    print("📦 正在初始化分布式MMSegmentation训练...")
    
    # 设置分布式环境
    rank, local_rank, world_size = setup_distributed()
    
    # 从文件加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    print(f"📝 加载配置文件: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 设置工作目录
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs'
    
    # 更新配置以支持分布式训练
    cfg.launcher = args.launcher
    
    # 调整batch size（每个进程的batch size）
    if hasattr(cfg, 'train_dataloader') and 'batch_size' in cfg.train_dataloader:
        original_batch_size = cfg.train_dataloader.batch_size
        # 保持总batch size不变，分配到各个进程
        cfg.train_dataloader.batch_size = max(1, original_batch_size // world_size)
        print(f"📊 调整batch size: {original_batch_size} -> {cfg.train_dataloader.batch_size} (per process)")
    
    print(f"📁 工作目录: {cfg.work_dir}")
    print(f"🚀 启动分布式训练 - Rank {rank}/{world_size}")
    
    # 创建Runner并开始训练
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()