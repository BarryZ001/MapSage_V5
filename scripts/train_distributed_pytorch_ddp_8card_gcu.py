#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import importlib
from datetime import datetime, timedelta

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
    GCU_AVAILABLE = True  # 仍然可以使用GCU功能 if available

# 导入MMSegmentation相关模块
try:
    from mmengine import Config
    from mmengine.runner import Runner
    from mmengine.logging import print_log
    print("MMSegmentation modules imported successfully")
except ImportError as e:
    print(f"Error importing MMSegmentation: {e}")
    sys.exit(1)

# 导入自定义模块以确保模型注册
try:
    import mmseg_custom  # 导入自定义模块包
    print("Custom modules imported successfully")
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    # 尝试单独导入关键模块
    try:
        from mmseg_custom.models import EncoderDecoder
        print("EncoderDecoder model imported successfully")
    except ImportError as e2:
        print(f"Error: Could not import EncoderDecoder model: {e2}")
        print("Please ensure mmseg_custom package is properly installed")


def find_python_module(name: str) -> bool:
    """helper: check if python module exists"""
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def init_distributed_with_fallback(init_method='env://', verbose=True):
    """
    尝试初始化分布式训练，支持多种后端的回退策略
    优先级: eccl (GCU) > nccl (CUDA) > gloo (CPU)
    """
    # 获取环境信息
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # 如果只有一个进程，跳过分布式初始化
    if world_size == 1:
        if verbose:
            print(f"[{datetime.now()}] Single process mode (world_size=1), skipping distributed initialization")
        return None
    
    # 定义后端候选列表（按优先级排序）
    candidates = []
    
    # 检查GCU环境，优先使用eccl后端
    if GCU_AVAILABLE:
        try:
            import torch_gcu
            if getattr(torch_gcu, 'is_available', lambda: False)() and getattr(torch_gcu, 'device_count', lambda: 0)() > 0:
                candidates.append('eccl')
                if verbose:
                    print(f"[{datetime.now()}] GCU available, adding 'eccl' backend to candidates")
        except Exception as e:
            if verbose:
                print(f"[{datetime.now()}] GCU check failed: {e}")
    
    # 检查CUDA环境
    if torch.cuda.is_available():
        candidates.append('nccl')
        if verbose:
            print(f"[{datetime.now()}] CUDA available, adding 'nccl' backend to candidates")
    
    # 总是添加gloo作为CPU后备
    candidates.append('gloo')
    
    if verbose:
        print(f"[{datetime.now()}] Trying backends in order: {candidates}")

    errs = {}
    for backend in candidates:
        try:
            if verbose:
                print(f"[{datetime.now()}] Attempting to initialize distributed with backend: {backend}")
            
            # 设置超时时间
            timeout = timedelta(seconds=30)
            
            # 初始化分布式进程组
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                timeout=timeout
            )
            
            if verbose:
                print(f"[{datetime.now()}] Successfully initialized distributed with backend: {backend}")
            return backend
            
        except Exception as e:
            errs[backend] = str(e)
            if verbose:
                print(f"[{datetime.now()}] Backend {backend} failed: {e}")
            
            # 清理失败的初始化
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except:
                pass

    # 所有后端都失败了，收集诊断信息并抛出异常
    diag = [
        "Failed to initialize any distributed backend. Tried: " + ", ".join(candidates),
        "",
        "Errors by backend:"
    ]
    for b, e in errs.items():
        diag.append(f"  - {b}: {e}")
    diag += [
        "",
        f"torch.__version__: {getattr(torch, '__version__', 'unknown')}",
        f"torch.cuda.is_available(): {torch.cuda.is_available()}",
        f"torch.distributed.is_available(): {dist.is_available()}",
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}",
        f"PYTHONPATH={os.environ.get('PYTHONPATH')}",
        f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}",
    ]
    raise RuntimeError("\n".join(diag))


def setup_distributed_environment():
    """设置分布式训练环境，返回 (rank, local_rank, world_size, device, backend_used)"""
    # 从环境变量获取分布式参数（torchrun 提供）
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    print(f"[{datetime.now()}] Distributed setup: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # 设置设备（默认为 cpu）
    device = 'cpu'

    # 优先检查GCU设备
    if GCU_AVAILABLE:
        try:
            import torch_gcu  # type: ignore
            # torch_gcu provides is_available() and device_count()
            if getattr(torch_gcu, 'is_available', lambda: False)() and getattr(torch_gcu, 'device_count', lambda: 0)() > 0:
                # 使用torch_gcu.device()创建正确的GCU设备对象
                try:
                    torch_gcu.set_device(local_rank)
                    device = torch_gcu.device(local_rank)  # 使用torch_gcu.device()而不是字符串
                    print(f"[{datetime.now()}] Successfully set torch_gcu device: {local_rank}")
                    print(f"[{datetime.now()}] GCU device object: {device}")
                except Exception as e:
                    # 如果torch_gcu.device()不可用，回退到CPU
                    print(f"[{datetime.now()}] Warning: torch_gcu.device() failed: {e}")
                    device = 'cpu'
                # 设置环境变量便于其它库发现设备
                os.environ['TOPS_VISIBLE_DEVICES'] = str(local_rank)
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA 可见性
                print(f"[{datetime.now()}] Using GCU device: {device}")
            else:
                print(f"[{datetime.now()}] torch_gcu present but reports not available or no devices")
        except Exception as e:
            print(f"[{datetime.now()}] GCU setup exception: {e}")

    # 如果GCU不可用，检查 CUDA
    if device == 'cpu':
        if torch.cuda.is_available():
            device = f'cuda:{local_rank}'
            try:
                torch.cuda.set_device(local_rank)
            except Exception as e:
                print(f"[{datetime.now()}] Warning: torch.cuda.set_device failed: {e}")
            print(f"[{datetime.now()}] Using CUDA device: {device}")
        else:
            print(f"[{datetime.now()}] Using device: {device}")

    backend_used = None
    if world_size > 1:
        # Initialize distributed process group with fallback strategy
        # Use env:// by default (torchrun sets MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE)
        init_method = os.environ.get('INIT_METHOD', 'env://')
        try:
            # Try prioritized backends
            backend_used = init_distributed_with_fallback(init_method=init_method, verbose=True)
        except Exception as e:
            # If initialization fails, print diagnostics and fallback to single-process
            print(f"[{datetime.now()}] Error initializing distributed: {e}")
            print(f"[{datetime.now()}] Falling back to single-process training mode")
            # 重置为单进程模式
            rank = 0
            local_rank = 0
            world_size = 1
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            backend_used = None
    else:
        print(f"[{datetime.now()}] Single-process mode: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    return rank, local_rank, world_size, device, backend_used


def cleanup_distributed():
    """清理分布式环境"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[{datetime.now()}] Distributed process group destroyed")
    except Exception as e:
        print(f"[{datetime.now()}] Warning while destroying process group: {e}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch DDP分布式训练 - T20 GCU 8卡')
    parser.add_argument('--config', required=True, help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录，用于保存日志和模型')
    parser.add_argument('--resume', help='恢复训练的检查点路径')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度训练')
    parser.add_argument('--auto-scale-lr', action='store_true', help='根据GPU数量自动缩放学习率')

    # 添加分布式训练相关参数
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch', help='分布式启动器')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--deterministic', action='store_true', help='启用确定性训练')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training (automatically set by torch.distributed.launch)')
    parser.add_argument('--local-rank', type=int, default=0, dest='local_rank',
                        help='Local rank for distributed training (alternative format)')

    return parser.parse_args()


def modify_config_for_distributed(cfg, rank, world_size, work_dir, backend_used='nccl', amp=False, auto_scale_lr=False):
    """修改配置以适应分布式训练"""
    # 确保backend_used是字符串类型
    if backend_used is None:
        backend_used = 'none'
    
    print(f"[{datetime.now()}] Modifying config for distributed training:")
    print(f"  - rank: {rank}")
    print(f"  - world_size: {world_size}")
    print(f"  - work_dir: {work_dir}")
    print(f"  - backend: {backend_used}")

    # 设置工作目录
    if work_dir:
        cfg.work_dir = work_dir

    # 设置分布式训练参数：将实际使用的 backend 写入 cfg（便于 mmengine/模型一致）
    if not hasattr(cfg, 'env_cfg') or cfg.env_cfg is None:
        cfg.env_cfg = dict(dist_cfg=dict(backend=backend_used))
    else:
        # safe set
        if not isinstance(cfg.env_cfg, dict):
            cfg.env_cfg = dict()
        cfg.env_cfg['dist_cfg'] = dict(backend=backend_used)

    # 设置设备配置 - 确保模型和数据都移动到正确的GCU设备
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if GCU_AVAILABLE:
        device = f'dtu:{local_rank}'
        cfg.device = device
        print(f"[{datetime.now()}] Config device set to: {device}")
        
        # 设置模型包装器配置，确保正确的设备分配
        cfg.model_wrapper_cfg = dict(
            type='MMDistributedDataParallel',
            find_unused_parameters=False,
            broadcast_buffers=False,
            device_ids=None,  # 让MMEngine自动处理设备分配
            output_device=None
        )
        print(f"[{datetime.now()}] Model wrapper configured for GCU distributed training")

    # 调整batch size（仅打印）
    try:
        if hasattr(cfg, 'train_dataloader') and hasattr(cfg.train_dataloader, 'batch_size'):
            original_batch_size = cfg.train_dataloader.batch_size
            print(f"Batch size per process: {original_batch_size}")
            print(f"Total effective batch size: {original_batch_size * world_size}")
    except Exception:
        pass

    # 自动缩放学习率（按 world_size 缩放）
    if auto_scale_lr:
        try:
            # try to find lr
            if hasattr(cfg, 'optim_wrapper') and isinstance(cfg.optim_wrapper, dict):
                opt = cfg.optim_wrapper.get('optimizer', {})
                if isinstance(opt, dict) and 'lr' in opt:
                    original_lr = opt['lr']
                    scaled_lr = original_lr * world_size
                    opt['lr'] = scaled_lr
                    cfg.optim_wrapper['optimizer'] = opt
                    print(f"Learning rate scaled from {original_lr} to {scaled_lr}")
        except Exception:
            pass

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

        # set local rank env so setup reads it (torchrun will set this automatically)
        if hasattr(args, 'local_rank') and args.local_rank is not None:
            os.environ['LOCAL_RANK'] = str(args.local_rank)

        # 设置分布式环境 (返回 backend_used)
        rank, local_rank, world_size, device, backend_used = setup_distributed_environment()

        # 加载配置
        cfg = Config.fromfile(args.config)

        # 修改配置以适应分布式训练
        cfg = modify_config_for_distributed(
            cfg, rank, world_size, args.work_dir, backend_used=backend_used or 'none',
            amp=args.amp, auto_scale_lr=args.auto_scale_lr
        )

        # 设置恢复训练
        if args.resume:
            cfg.resume = True
            cfg.load_from = args.resume
            print(f"Resume training from: {args.resume}")

        # 创建Runner并开始训练
        if rank == 0:
            print("Starting distributed training... (rank 0)")

        runner = Runner.from_cfg(cfg)
        
        # 显式验证和移动模型到正确的设备
        if GCU_AVAILABLE and hasattr(runner, 'model'):
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            try:
                # 使用torch_gcu.device()获取正确的设备对象
                import torch_gcu
                target_device = torch_gcu.device(local_rank)
                
                # 检查模型当前设备
                model_device = next(runner.model.parameters()).device
                print(f"[{datetime.now()}] Model current device: {model_device}")
                
                # 如果模型不在正确的GCU设备上，显式移动
                if str(model_device) != str(target_device):
                    print(f"[{datetime.now()}] Moving model from {model_device} to {target_device}")
                    runner.model = runner.model.to(target_device)
                    
                    # 验证移动是否成功
                    new_device = next(runner.model.parameters()).device
                    print(f"[{datetime.now()}] Model moved to device: {new_device}")
                else:
                    print(f"[{datetime.now()}] Model already on correct device: {model_device}")
                    
            except Exception as e:
                print(f"[{datetime.now()}] Warning: Could not verify/move model device: {e}")
        
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