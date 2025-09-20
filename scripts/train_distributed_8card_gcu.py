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

# 导入必要的MMSeg组件
try:
    import mmseg
    import mmseg.models
    from mmseg.models.backbones import MixVisionTransformer
    from mmseg.models.decode_heads import SegformerHead
    from mmseg.models.segmentors import EncoderDecoder
    
    # 确保模型注册到MMEngine
    from mmengine.registry import MODELS
    if 'MixVisionTransformer' not in MODELS.module_dict:
        MODELS.register_module(name='MixVisionTransformer', module=MixVisionTransformer)
        print("✅ MixVisionTransformer已注册到MMEngine")
    
    if 'SegformerHead' not in MODELS.module_dict:
        MODELS.register_module(name='SegformerHead', module=SegformerHead)
        print("✅ SegformerHead已注册到MMEngine")
        
    if 'EncoderDecoder' not in MODELS.module_dict:
        MODELS.register_module(name='EncoderDecoder', module=EncoderDecoder)
        print("✅ EncoderDecoder已注册到MMEngine")
        
    print("✅ MMSeg模型组件导入和注册成功")
except ImportError as e:
    print(f"⚠️ MMSeg导入失败: {e}")
    print("⚠️ 将使用自定义模型组件")

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
    
    # 检查并创建工作目录
    if hasattr(cfg, 'work_dir') and cfg.work_dir:
        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir, exist_ok=True)
            print(f"📁 创建工作目录: {cfg.work_dir}")
    else:
        # 如果配置文件没有work_dir，设置默认值
        cfg.work_dir = './work_dirs/train_distributed_8card_gcu'
        os.makedirs(cfg.work_dir, exist_ok=True)
        print(f"📁 设置默认工作目录: {cfg.work_dir}")
    
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
    
    # 修改配置以避免MMEngine的设备不匹配问题
    print("🔧 修改配置以适配GCU设备...")
    
    # 关键修复：配置MMEngine使用正确的设备
    if torch_gcu is not None:
        device = f'gcu:{local_rank}'
        
        # 1. 设置模型初始化设备
        if hasattr(cfg, 'model') and isinstance(cfg.model, dict):
            cfg.model['init_cfg'] = {'type': 'Normal', 'std': 0.01}
        
        # 2. 配置分布式训练设备
        cfg.device = device
        
        # 3. 禁用CUDA相关设置，避免设备冲突
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # 4. 配置MMEngine的设备设置
        if not hasattr(cfg, 'env_cfg'):
            cfg.env_cfg = {}
        cfg.env_cfg['dist_cfg'] = {'backend': 'gloo'}
        
        # 5. 确保模型包装器使用正确设备
        if hasattr(cfg, 'model_wrapper_cfg'):
            if cfg.model_wrapper_cfg is None:
                cfg.model_wrapper_cfg = {}
            # 不设置device_ids，让MMEngine自动检测
            cfg.model_wrapper_cfg.pop('device_ids', None)
            cfg.model_wrapper_cfg.pop('output_device', None)
        
        print(f"🔧 配置设备为: {device}")
        print(f"🔧 配置分布式后端为: gloo")
    
    # 创建Runner并开始训练
    print("🚀 创建Runner...")
    runner = Runner.from_cfg(cfg)
    
    # 验证模型设备设置
    if torch_gcu is not None and hasattr(runner, 'model'):
        print("🔍 验证模型设备设置...")
        
        # 检查模型参数设备
        device_types = set()
        for name, param in runner.model.named_parameters():
            device_types.add(param.device.type)
        
        print(f"📊 模型参数设备类型: {device_types}")
        
        if 'cpu' in device_types and len(device_types) > 1:
            print("⚠️ 检测到混合设备，正在修复...")
            device = f'gcu:{local_rank}'
            runner.model = runner.model.to(device)
            print(f"✅ 模型已移动到: {device}")
        elif 'gcu' in device_types:
            print("✅ 模型已正确配置在GCU设备上")
        else:
            print(f"⚠️ 模型在意外设备上: {device_types}")
    
    print("✅ Runner创建完成，设备配置验证通过")
    
    runner.train()
    
    # 清理分布式环境
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 分布式进程组已清理")

if __name__ == '__main__':
    main()