#!/usr/bin/env python3
"""
基于OpenMPI+Horovod的分布式训练脚本
符合TopsDL官方推荐架构
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Union

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
try:
    import horovod.torch as hvd  # type: ignore
    HOROVOD_AVAILABLE = True
except ImportError:
    print("Warning: Horovod not available. Please install with: pip install horovod[pytorch]")
    HOROVOD_AVAILABLE = False
    hvd = None  # type: ignore

try:
    import torch_xla.core.xla_model as xm  # type: ignore
    XLA_AVAILABLE = True
except ImportError:
    print("Warning: torch_xla not available. XLA device support disabled.")
    XLA_AVAILABLE = False
    xm = None  # type: ignore

from mmengine import Config
from mmengine.runner import Runner
from mmengine.logging import print_log

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [Rank %(rank)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def setup_horovod_environment() -> Tuple[int, int, int, str]:
    """初始化Horovod分布式环境"""
    if not HOROVOD_AVAILABLE or hvd is None:
        raise RuntimeError("Horovod is not available. Please install with: pip install horovod[pytorch]")
    
    # 初始化Horovod
    hvd.init()  # type: ignore
    
    # 获取分布式信息
    rank = hvd.rank()  # type: ignore
    local_rank = hvd.local_rank()  # type: ignore
    size = hvd.size()  # type: ignore
    
    print_log(f"Horovod initialized: rank={rank}, local_rank={local_rank}, size={size}", 
              logger='current')
    
    # 设置GCU设备
    try:
        import torch_gcu
        if torch_gcu.is_available():
            torch_gcu.set_device(local_rank)
            device = f'gcu:{local_rank}'
            print_log(f"Using GCU device: {device}", logger='current')
        else:
            device = 'cpu'
            print_log("GCU not available, using CPU", logger='current')
    except ImportError:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f'cuda:{local_rank}'
            print_log(f"Using CUDA device: {device}", logger='current')
        else:
            device = 'cpu'
            print_log("Using CPU device", logger='current')
    
    return rank, local_rank, size, device

def modify_config_for_horovod(cfg, rank, size, device):
    """修改配置以适配Horovod分布式训练"""
    
    # 设置设备
    cfg.device = device
    
    # 调整学习率（Horovod通常需要根据进程数调整学习率）
    if hasattr(cfg, 'optim') and hasattr(cfg.optim, 'lr'):
        original_lr = cfg.optim.lr
        cfg.optim.lr = original_lr * size
        print_log(f"Adjusted learning rate: {original_lr} -> {cfg.optim.lr} (scaled by {size})", 
                  logger='current')
    
    # 设置数据加载器
    if hasattr(cfg, 'train_dataloader'):
        # 确保每个进程处理不同的数据子集
        cfg.train_dataloader.sampler = dict(
            type='DistributedSampler',
            shuffle=True,
            seed=42
        )
        
        # 调整batch size（可选，根据需要）
        if hasattr(cfg.train_dataloader, 'batch_size'):
            # 保持总batch size不变，每个进程处理更小的batch
            cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size // size
            print_log(f"Adjusted batch size per process: {cfg.train_dataloader.batch_size}", 
                      logger='current')
    
    # 设置验证数据加载器
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader.sampler = dict(
            type='DistributedSampler',
            shuffle=False,
            seed=42
        )
    
    # 禁用MMEngine的DDP包装（Horovod有自己的分布式机制）
    if hasattr(cfg, 'model_wrapper_cfg'):
        cfg.model_wrapper_cfg = None
    
    # 设置日志和检查点保存（只在rank 0保存）
    if rank == 0:
        print_log("Rank 0: Enabled logging and checkpointing", logger='current')
    else:
        # 非主进程禁用某些输出
        if hasattr(cfg, 'vis_backends'):
            cfg.vis_backends = []
        if hasattr(cfg, 'default_hooks') and hasattr(cfg.default_hooks, 'logger'):
            cfg.default_hooks.logger.interval = 1000  # 减少日志频率
    
    return cfg

def wrap_model_with_horovod(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """使用Horovod包装模型和优化器"""
    if not HOROVOD_AVAILABLE or hvd is None:
        raise RuntimeError("Horovod is not available")
    
    # 广播模型参数（确保所有进程的模型参数一致）
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)  # type: ignore
    
    # 广播优化器状态
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)  # type: ignore
    
    # 使用Horovod的DistributedOptimizer包装优化器
    optimizer = hvd.DistributedOptimizer(  # type: ignore
        optimizer, 
        named_parameters=model.named_parameters(),
        compression=hvd.Compression.none,  # type: ignore # 可以选择压缩算法
        op=hvd.Average  # type: ignore # 梯度平均
    )
    
    print_log("Model and optimizer wrapped with Horovod", logger='current')
    
    return model, optimizer

def main():
    parser = argparse.ArgumentParser(description='Horovod分布式训练脚本')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--resume', help='恢复训练的检查点路径')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度')
    parser.add_argument('--auto-scale-lr', action='store_true', help='自动缩放学习率')
    
    args = parser.parse_args()
    
    # 初始化Horovod
    rank, local_rank, size, device = setup_horovod_environment()
    
    # 设置日志（包含rank信息）
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [Rank {rank}] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print_log(f"Starting Horovod distributed training on {size} processes", logger='current')
    print_log(f"Current process: rank={rank}, local_rank={local_rank}, device={device}", 
              logger='current')
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    print_log(f"Loaded config from {args.config}", logger='current')
    
    # 修改配置以适配Horovod
    cfg = modify_config_for_horovod(cfg, rank, size, device)
    
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not hasattr(cfg, 'work_dir'):
        cfg.work_dir = f'./work_dirs/{Path(args.config).stem}_horovod'
    
    # 确保工作目录存在（只在rank 0创建）
    if rank == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)
        print_log(f"Work directory: {cfg.work_dir}", logger='current')
    
    # 同步所有进程
    if HOROVOD_AVAILABLE and hvd is not None:
        hvd.allreduce(torch.tensor(0), name='sync')  # type: ignore
    
    # 设置恢复训练
    if args.resume:
        cfg.resume = args.resume
        print_log(f"Resume training from {args.resume}", logger='current')
    
    # 设置自动混合精度
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
        print_log("Enabled automatic mixed precision", logger='current')
    
    try:
        # 强制CPU初始化（避免设备不匹配）
        original_device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        print_log("Creating runner with CPU initialization...", logger='current')
        
        # 创建Runner
        runner = Runner.from_cfg(cfg)
        
        # 恢复设备环境
        if original_device:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_device
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
        print_log("Runner created successfully", logger='current')
        
        # 手动将模型移动到指定设备
        print_log(f"Moving model to device: {device}", logger='current')
        
        if device.startswith('gcu:'):
            # 对于GCU设备，直接移动到指定设备
            runner.model = runner.model.to(device)
        elif device.startswith('xla:') and XLA_AVAILABLE and xm is not None:
            # 对于XLA设备，需要特殊处理
            runner.model = runner.model.to(xm.xla_device())  # type: ignore
        else:
            runner.model = runner.model.to(device)
        
        # 验证模型参数设备
        model_device = next(runner.model.parameters()).device
        print_log(f"Model parameters are on device: {model_device}", logger='current')
        
        # 获取优化器 - 安全访问optim_wrapper
        if hasattr(runner, 'optim_wrapper') and runner.optim_wrapper is not None:
            optim_wrapper = runner.optim_wrapper
            # 使用getattr安全访问optimizer属性
            optimizer = getattr(optim_wrapper, 'optimizer', None)
            if optimizer is not None:
                # 使用Horovod包装模型和优化器
                if HOROVOD_AVAILABLE and hvd is not None:
                    runner.model, wrapped_optimizer = wrap_model_with_horovod(runner.model, optimizer)
                    # 更新runner中的优化器
                    setattr(optim_wrapper, 'optimizer', wrapped_optimizer)
                    print_log("Model and optimizer wrapped with Horovod successfully", logger='current')
                else:
                    print_log("Warning: Horovod not available, skipping distributed wrapper", logger='current')
            else:
                print_log("Warning: optim_wrapper has no optimizer attribute or optimizer is None", logger='current')
        else:
            print_log("Warning: Runner has no optim_wrapper or optim_wrapper is None", logger='current')
        
        # 开始训练
        print_log("Starting training...", logger='current')
        runner.train()
        
        print_log("Training completed successfully!", logger='current')
        
    except Exception as e:
        print_log(f"Training failed with error: {str(e)}", logger='current', level=logging.ERROR)
        import traceback
        print_log(f"Traceback: {traceback.format_exc()}", logger='current', level=logging.ERROR)
        raise
        
    finally:
        print_log(f"Rank {rank} finished", logger='current')

if __name__ == '__main__':
    main()