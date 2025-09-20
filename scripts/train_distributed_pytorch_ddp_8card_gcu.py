#!/usr/bin/env python3
"""
PyTorch DDP分布式训练脚本 - T20 GCU 8卡版本
不依赖Horovod，使用原生PyTorch分布式训练
适用于燧原T20 GCU环境
"""

import os
import sys
import argparse
import warnings
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    sys.exit(1)

# 尝试导入GCU支持
try:
    import torch_gcu  # type: ignore
    # 安全地检查torch_gcu版本
    if hasattr(torch_gcu, '__version__'):
        print(f"torch_gcu version: {torch_gcu.__version__}")
    else:
        print("torch_gcu imported successfully (version info not available)")
    GCU_AVAILABLE = True
except ImportError:
    print("Warning: torch_gcu not available. GCU device support disabled.")
    GCU_AVAILABLE = False
except AttributeError as e:
    print(f"Warning: torch_gcu version check failed: {e}")
    GCU_AVAILABLE = True  # 仍然可以使用GCU功能

# 导入MMSegmentation相关模块
try:
    from mmengine import Config
    from mmengine.runner import Runner
    from mmengine.logging import print_log
    print("MMSegmentation modules imported successfully")
except ImportError as e:
    print(f"Error importing MMSegmentation: {e}")
    sys.exit(1)


def setup_distributed_environment():
    """设置分布式训练环境"""
    # 从环境变量获取分布式参数
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Distributed setup: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # 设置设备
    if GCU_AVAILABLE and hasattr(torch, 'gcu') and torch.gcu.is_available():  # type: ignore
        device = f'gcu:{local_rank}'
        torch.gcu.set_device(local_rank)  # type: ignore
        print(f"Using GCU device: {device}")
    else:
        device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"Using device: {device}")
    
    # 初始化进程组
    if world_size > 1:
        # 设置分布式后端
        if GCU_AVAILABLE:
            backend = 'nccl'  # GCU通常使用NCCL后端
        elif torch.cuda.is_available():
            backend = 'nccl'
        else:
            backend = 'gloo'
        
        print(f"Initializing distributed training with backend: {backend}")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        # 验证分布式初始化
        if dist.is_initialized():
            print(f"Distributed training initialized successfully")
            print(f"Process group size: {dist.get_world_size()}")
            print(f"Current rank: {dist.get_rank()}")
        else:
            raise RuntimeError("Failed to initialize distributed training")
    
    return rank, local_rank, world_size, device


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch DDP分布式训练 - T20 GCU 8卡')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录，用于保存日志和模型')
    parser.add_argument('--resume', help='恢复训练的检查点路径')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度训练')
    parser.add_argument('--auto-scale-lr', action='store_true', help='根据GPU数量自动缩放学习率')
    
    # 添加分布式训练相关参数
    parser.add_argument('--local_rank', type=int, default=0, 
                       help='Local rank for distributed training (automatically set by torch.distributed.launch)')
    parser.add_argument('--local-rank', type=int, default=0, dest='local_rank',
                       help='Local rank for distributed training (alternative format)')
    
    return parser.parse_args()


def modify_config_for_distributed(cfg, rank, world_size, work_dir, amp=False, auto_scale_lr=False):
    """修改配置以适应分布式训练"""
    
    # 设置工作目录
    if work_dir:
        cfg.work_dir = work_dir
    
    # 设置分布式训练参数
    if hasattr(cfg, 'env_cfg'):
        cfg.env_cfg.dist_cfg = dict(backend='nccl')
    else:
        cfg.env_cfg = dict(dist_cfg=dict(backend='nccl'))
    
    # 调整batch size
    if hasattr(cfg, 'train_dataloader') and hasattr(cfg.train_dataloader, 'batch_size'):
        original_batch_size = cfg.train_dataloader.batch_size
        # 在分布式训练中，每个进程的batch size保持不变
        print(f"Batch size per GPU: {original_batch_size}")
        print(f"Total effective batch size: {original_batch_size * world_size}")
    
    # 自动缩放学习率
    if auto_scale_lr and hasattr(cfg, 'optim_wrapper') and hasattr(cfg.optim_wrapper, 'optimizer'):
        if hasattr(cfg.optim_wrapper.optimizer, 'lr'):
            original_lr = cfg.optim_wrapper.optimizer.lr
            scaled_lr = original_lr * world_size
            cfg.optim_wrapper.optimizer.lr = scaled_lr
            print(f"Learning rate scaled from {original_lr} to {scaled_lr}")
    
    # 启用AMP
    if amp:
        if not hasattr(cfg, 'optim_wrapper'):
            cfg.optim_wrapper = dict()
        cfg.optim_wrapper['type'] = 'AmpOptimWrapper'
        cfg.optim_wrapper['loss_scale'] = 'dynamic'
        print("Automatic Mixed Precision (AMP) enabled")
    
    # 设置随机种子
    if not hasattr(cfg, 'randomness'):
        cfg.randomness = dict()
    cfg.randomness['seed'] = 42
    cfg.randomness['deterministic'] = False
    
    # 设置日志配置
    if hasattr(cfg, 'default_hooks') and hasattr(cfg.default_hooks, 'logger'):
        cfg.default_hooks.logger.interval = 50
    
    return cfg


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        print("=== PyTorch DDP分布式训练启动 ===")
        print(f"配置文件: {args.config}")
        print(f"工作目录: {args.work_dir}")
        print(f"恢复训练: {args.resume}")
        print(f"自动混合精度: {args.amp}")
        print(f"自动缩放学习率: {args.auto_scale_lr}")
        print(f"Local rank: {args.local_rank}")
        print("=" * 40)
        
        # 如果通过命令行传入了local_rank，设置到环境变量中
        # 这样setup_distributed_environment函数就能正确获取到
        if hasattr(args, 'local_rank') and args.local_rank is not None:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        
        # 设置分布式环境
        rank, local_rank, world_size, device = setup_distributed_environment()
        
        # 加载配置
        cfg = Config.fromfile(args.config)
        
        # 修改配置以适应分布式训练
        cfg = modify_config_for_distributed(
            cfg, rank, world_size, args.work_dir, 
            amp=args.amp, auto_scale_lr=args.auto_scale_lr
        )
        
        # 设置恢复训练
        if args.resume:
            cfg.resume = True
            cfg.load_from = args.resume
            print(f"Resume training from: {args.resume}")
        
        # 创建Runner并开始训练
        if rank == 0:
            print("Starting distributed training...")
        
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        if rank == 0:
            print("Training completed successfully!")
            
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return 1
    
    finally:
        # 清理分布式环境
        cleanup_distributed()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())