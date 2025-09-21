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
    Try to initialize torch.distributed process group with a prioritized list:
    - eccl (if python module available and properly registered)
    - nccl (if torch reports nccl available)
    - gloo (always fallback)
    Returns selected backend string.
    Raises RuntimeError with diagnostics if all fail.
    """
    candidates = []
    errs = {}

    # Try to register ECCL backend if available
    eccl_available = False
    if find_python_module("eccl"):
        try:
            # Import eccl with try-except to handle import errors gracefully
            try:
                import eccl  # type: ignore
            except ImportError:
                if verbose:
                    print(f"[{datetime.now()}] ECCL module found but import failed")
                eccl = None
            
            if eccl is not None:
                # Try to register ECCL backend with PyTorch
                if hasattr(eccl, 'init') and hasattr(dist, 'Backend'):
                    # Check if ECCL backend is already registered
                    if hasattr(dist.Backend, 'ECCL') or 'eccl' in getattr(dist, '_backend_registry', {}):
                        eccl_available = True
                        if verbose:
                            print(f"[{datetime.now()}] ECCL backend already registered")
                    else:
                        # Try to register ECCL backend (this may not work in all PyTorch versions)
                        try:
                            # Check if register_backend exists (newer PyTorch versions)
                            if hasattr(dist, 'register_backend'):
                                dist.register_backend('eccl', eccl)  # type: ignore
                                eccl_available = True
                                if verbose:
                                    print(f"[{datetime.now()}] ECCL backend registered successfully")
                            else:
                                # For older PyTorch versions, just try to use eccl directly
                                eccl_available = True
                                if verbose:
                                    print(f"[{datetime.now()}] ECCL backend available (direct usage)")
                        except Exception as reg_e:
                            if verbose:
                                print(f"[{datetime.now()}] Failed to register ECCL backend: {reg_e}")
        except Exception as e:
            if verbose:
                print(f"[{datetime.now()}] ECCL module import/setup failed: {e}")

    # Build candidate list
    if eccl_available:
        candidates.append("eccl")
    
    # Add nccl if available in this PyTorch build
    try:
        if getattr(dist, "is_nccl_available", lambda: False)():
            candidates.append("nccl")
    except Exception:
        pass
    
    # Always allow gloo as fallback
    candidates.append("gloo")

    # Unique preserving order
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    for backend in candidates:
        try:
            if verbose:
                print(f"[{datetime.now()}] Trying init_process_group backend='{backend}', init_method='{init_method}'")
            
            # Special handling for GLOO to improve network configuration
            if backend == "gloo":
                # Set network interface for GLOO if not already set
                if not os.environ.get('GLOO_SOCKET_IFNAME'):
                    # Try to find a suitable network interface
                    try:
                        import socket
                        hostname = socket.gethostname()
                        # Common network interfaces in containers/servers
                        for iface in ['eth0', 'ens3', 'enp0s3', 'lo']:
                            try:
                                import subprocess
                                result = subprocess.run(['ip', 'addr', 'show', iface], 
                                                      capture_output=True, text=True, timeout=5)
                                if result.returncode == 0 and 'inet ' in result.stdout:
                                    os.environ['GLOO_SOCKET_IFNAME'] = iface
                                    if verbose:
                                        print(f"[{datetime.now()}] Set GLOO_SOCKET_IFNAME={iface}")
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass
                
                # Set timeout for GLOO
                if not os.environ.get('GLOO_TIMEOUT_SECONDS'):
                    os.environ['GLOO_TIMEOUT_SECONDS'] = '300'  # 5 minutes
            
            # call init (torchrun sets envs like RANK/WORLD_SIZE/LOCAL_RANK)
            dist.init_process_group(backend=backend, init_method=init_method, timeout=timedelta(seconds=300))
            if verbose:
                print(f"[{datetime.now()}] Successfully initialized distributed backend='{backend}'")
            return backend
        except Exception as e:
            errs[backend] = f"{type(e).__name__}: {e}"
            # try to cleanup any partial init
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass

    # All failed -> collect diagnostics and raise
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
                device = f'gcu:{local_rank}'
                try:
                    torch_gcu.set_device(local_rank)
                except Exception as e:
                    # best-effort, not fatal
                    print(f"[{datetime.now()}] Warning: torch_gcu.set_device failed: {e}")
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
            traceback.print_exc()
            
            # Force single-process mode
            world_size = 1
            rank = 0
            local_rank = 0
            backend_used = 'none'
            print(f"[{datetime.now()}] Single-process fallback: rank={rank}, local_rank={local_rank}, world_size={world_size}")

        # After init, verify (only if we didn't fallback)
        if world_size > 1:
            if dist.is_initialized():
                print(f"[{datetime.now()}] Distributed training initialized successfully with backend='{backend_used}'")
                try:
                    print(f"[{datetime.now()}] Process group size: {dist.get_world_size()}")
                    print(f"[{datetime.now()}] Current rank: {dist.get_rank()}")
                except Exception:
                    pass
            else:
                print(f"[{datetime.now()}] Warning: Distributed init reported success but not initialized, falling back to single-process")
                world_size = 1
                rank = 0
                local_rank = 0
                backend_used = 'none'

    else:
        # single process / single device
        backend_used = 'none'
        print(f"[{datetime.now()}] Single-process mode: distributed disabled")

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

    # 启用AMP（保守地设置为 dynamic loss-scale）
    if amp:
        if not hasattr(cfg, 'optim_wrapper') or cfg.optim_wrapper is None:
            cfg.optim_wrapper = dict()
        cfg.optim_wrapper['type'] = 'AmpOptimWrapper'
        cfg.optim_wrapper['loss_scale'] = 'dynamic'
        print("Automatic Mixed Precision (AMP) enabled")

    # 设置随机种子与日志间隔
    if not hasattr(cfg, 'randomness') or cfg.randomness is None:
        cfg.randomness = dict()
    cfg.randomness['seed'] = 42
    cfg.randomness['deterministic'] = False

    try:
        if hasattr(cfg, 'default_hooks') and isinstance(cfg.default_hooks, dict) and 'logger' in cfg.default_hooks:
            cfg.default_hooks['logger']['interval'] = 50
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
            cfg, rank, world_size, args.work_dir, backend_used=backend_used,
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