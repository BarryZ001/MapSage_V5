#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3 + MMRS-1M 8卡分布式训练脚本 - 燧原T20 GCU版本
基于成功的demo_deepspeed_xla.py经验，适配MMSegmentation框架

使用方法:
torchrun --nproc_per_node=8 --master_port=29500 scripts/train_dinov3_distributed_8card_gcu.py configs/train_dinov3_mmrs1m_t20_gcu_8card.py
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量 - 基于成功的demo_deepspeed_xla.py经验
os.environ.setdefault('PYTORCH_GCU_ALLOC_CONF', 'backend:topsMallocAsync')
os.environ.setdefault('TORCH_ECCL_AVOID_RECORD_STREAMS', 'false')
os.environ.setdefault('TORCH_ECCL_ASYNC_ERROR_HANDLING', '3')

# 导入必要的库
try:
    import torch
    import torch_gcu  # 燧原GCU支持
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"✅ torch_gcu可用: {torch_gcu.is_available()}")
    if torch_gcu.is_available():
        print(f"✅ GCU设备数: {torch_gcu.device_count()}")
    else:
        raise RuntimeError("torch_gcu不可用，请检查安装")
except ImportError as e:
    print(f"❌ 导入torch_gcu失败: {e}")
    sys.exit(1)

# 导入MMEngine和MMSegmentation
try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.dist import init_dist, get_rank, get_world_size
    print("✅ MMEngine导入成功")
except ImportError as e:
    print(f"❌ MMEngine导入失败: {e}")
    sys.exit(1)

try:
    import mmseg
    from mmseg.models import *
    from mmseg.datasets import *
    print("✅ MMSegmentation导入成功")
except ImportError as e:
    print(f"❌ MMSegmentation导入失败: {e}")
    sys.exit(1)

# 导入自定义模块
try:
    import mmseg_custom.models
    import mmseg_custom.datasets
    import mmseg_custom.transforms
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"⚠️ 自定义模块导入失败: {e}")

def setup_gcu_environment():
    """设置GCU环境 - 基于demo_deepspeed_xla.py的成功经验"""
    
    # 获取分布式训练参数
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    print(f"🚀 [PID {os.getpid()}] 分布式参数:")
    print(f"   - local_rank: {local_rank}")
    print(f"   - world_size: {world_size}")
    print(f"   - rank: {rank}")
    
    # 设置设备 - 使用xla设备名称（基于demo_deepspeed_xla.py）
    device_name = f"xla:{local_rank}"
    torch_gcu.set_device(local_rank)  # 设置当前GCU设备
    
    print(f"🔧 设备配置: {device_name}")
    
    # 验证设备可用性
    try:
        test_tensor = torch.randn(2, 2).to(device_name)
        print(f"✅ 设备 {device_name} 验证成功")
        del test_tensor
    except Exception as e:
        print(f"❌ 设备 {device_name} 验证失败: {e}")
        raise
    
    return local_rank, world_size, rank, device_name

def init_distributed():
    """初始化分布式训练"""
    try:
        # 直接使用torch.distributed初始化，避免MMEngine的init_dist可能的CUDA依赖
        import torch.distributed as dist
        
        # 获取环境变量
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        
        print(f"🔧 分布式环境变量:")
        print(f"   - RANK: {rank}")
        print(f"   - LOCAL_RANK: {local_rank}")
        print(f"   - WORLD_SIZE: {world_size}")
        print(f"   - MASTER_ADDR: {master_addr}")
        print(f"   - MASTER_PORT: {master_port}")
        
        # 检查可用的后端
        print(f"🔍 检查分布式后端可用性:")
        print(f"   - NCCL: {dist.is_nccl_available()}")
        print(f"   - MPI: {dist.is_mpi_available()}")
        print(f"   - Gloo: {dist.is_gloo_available()}")
        
        # 初始化进程组，优先使用gloo后端（在GCU环境下更稳定）
        if not dist.is_initialized():
            backend = 'gloo'  # 使用gloo后端，因为eccl不被支持
            dist.init_process_group(
                backend=backend,
                init_method=f'tcp://{master_addr}:{master_port}',
                rank=rank,
                world_size=world_size
            )
            print(f"✅ 分布式初始化成功 - 后端: {backend}")
            print(f"   - rank: {dist.get_rank()}")
            print(f"   - world_size: {dist.get_world_size()}")
        else:
            print("✅ 分布式已初始化")
        
        return True
    except Exception as e:
        print(f"❌ 分布式初始化失败: {e}")
        return False

def load_and_validate_config(config_path, work_dir=None):
    """加载和验证配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    print(f"📝 加载配置文件: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 设置工作目录
    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs/dinov3_mmrs1m_8card_gcu'
    
    print(f"📁 工作目录: {cfg.work_dir}")
    
    # 确保工作目录存在
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 验证关键配置
    if not hasattr(cfg, 'model'):
        raise ValueError("配置文件缺少model配置")
    
    if not hasattr(cfg, 'train_dataloader'):
        raise ValueError("配置文件缺少train_dataloader配置")
    
    print("✅ 配置文件验证通过")
    return cfg

def main():
    parser = argparse.ArgumentParser(description='DINOv3 8卡分布式训练脚本')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--launcher', 
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch',
                        help='分布式启动器')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='本地rank（由torchrun自动设置）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--deterministic', action='store_true',
                        help='是否使用确定性训练')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    args = parser.parse_args()

    print("🚀 DINOv3 + MMRS-1M 8卡分布式训练启动")
    print("=" * 60)
    
    # 1. 设置GCU环境
    local_rank, world_size, rank, device_name = setup_gcu_environment()
    
    # 2. 初始化分布式训练
    if args.launcher != 'none' and world_size > 1:
        if not init_distributed():
            print("❌ 分布式初始化失败，退出训练")
            sys.exit(1)
    
    # 3. 加载配置
    cfg = load_and_validate_config(args.config, args.work_dir)
    
    # 4. 设置随机种子
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)
        print(f"🎲 随机种子: {args.seed}, 确定性: {args.deterministic}")
    
    # 5. 设置启动器
    if args.launcher != 'none':
        cfg.launcher = args.launcher
        print(f"🚀 启动器: {args.launcher}")
    
    # 6. 设置恢复训练
    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume
        print(f"🔄 恢复训练: {args.resume}")
    
    # 7. 显示训练信息
    print(f"📊 训练信息:")
    print(f"   - 配置文件: {args.config}")
    print(f"   - 工作目录: {cfg.work_dir}")
    print(f"   - 设备: {device_name}")
    print(f"   - 世界大小: {world_size}")
    print(f"   - 本地rank: {local_rank}")
    
    # 8. 创建Runner并开始训练
    try:
        print("🏗️ 创建训练Runner...")
        runner = Runner.from_cfg(cfg)
        
        print("🚀 开始训练...")
        print("=" * 60)
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 开始训练
        runner.train()
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        
        print("=" * 60)
        print("✅ 训练完成!")
        print(f"⏱️ 总训练时间: {training_time:.2f}秒 ({training_time/3600:.2f}小时)")
        print(f"📁 模型保存在: {cfg.work_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()