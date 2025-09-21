#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    print("✅ torch_gcu导入成功，可用设备数: {}".format(torch_gcu.device_count()))
except ImportError as e:
    print("⚠️ torch_gcu导入失败: {}".format(e))
    torch_gcu = None

try:
    import ptex
    print("✅ ptex导入成功")
except ImportError as e:
    print("⚠️ ptex导入失败: {}".format(e))
    ptex = None

# 尝试导入MMSeg相关模块
try:
    import mmseg  # type: ignore
    from mmseg.models import *  # type: ignore
    from mmseg.datasets import *  # type: ignore
except ImportError as e:
    print("⚠️ 模块导入失败: {}".format(e))

# 尝试导入自定义模块
try:
    from mmseg_custom.models import *  # type: ignore
    from mmseg_custom.datasets import *  # type: ignore
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print("⚠️ 自定义模块导入失败: {}".format(e))

# 尝试导入MMSeg模型组件并注册
try:
    import mmseg
    import mmseg.models
    from mmseg.models.backbones import MixVisionTransformer
    from mmseg.models.decode_heads import SegformerHead
    from mmseg.models.segmentors import EncoderDecoder
    
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
    print("⚠️ MMSeg导入失败: {}".format(e))
    print("⚠️ 将使用自定义模型组件")

def main():
    parser = argparse.ArgumentParser(description='8卡分布式训练脚本')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录路径')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                       default='pytorch', help='分布式启动器')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程rank')
    args = parser.parse_args()
    
    print("🚀 启动8卡分布式训练")
    print("📄 配置文件: {}".format(args.config))
    print("🔧 启动器: {}".format(args.launcher))
    
    # 1. 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 检查并创建工作目录
    if args.work_dir:
        # 使用命令行指定的工作目录
        cfg.work_dir = args.work_dir
        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir, exist_ok=True)
            print("📁 创建工作目录: {}".format(cfg.work_dir))
    elif hasattr(cfg, 'work_dir') and cfg.work_dir:
        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir, exist_ok=True)
            print("📁 创建工作目录: {}".format(cfg.work_dir))
    else:
        # 如果配置文件没有work_dir，设置默认值
        cfg.work_dir = './work_dirs/train_distributed_8card_gcu'
        os.makedirs(cfg.work_dir, exist_ok=True)
        print("📁 设置默认工作目录: {}".format(cfg.work_dir))
    
    # 设置日志目录
    log_dir = os.path.join(cfg.work_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 获取分布式参数
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print("🌍 分布式训练参数:")
    print("  - WORLD_SIZE: {}".format(world_size))
    print("  - RANK: {}".format(rank))
    print("  - LOCAL_RANK: {}".format(local_rank))
    
    # 配置分布式训练
    if world_size > 1:
        cfg.launcher = args.launcher
        print("🔧 启用分布式训练，launcher: {}".format(args.launcher))
        
        # 配置环境变量
        if not hasattr(cfg, 'env_cfg'):
            cfg.env_cfg = {}
        if not hasattr(cfg.env_cfg, 'dist_cfg'):
            cfg.env_cfg.dist_cfg = {}
        
        # 设置ECCL后端配置
        cfg.env_cfg.dist_cfg['backend'] = 'eccl'
        cfg.env_cfg.dist_cfg['init_method'] = 'env://'
        print("✅ 配置MMEngine使用ECCL后端")
        
        # 配置GCU设备
        cfg.device = 'gcu'
        print("🔧 配置GCU设备，world_size: {}".format(world_size))
    else:
        cfg.launcher = 'none'
        print("🔧 单进程模式，禁用分布式")
        cfg.device = 'gcu'
        print("🔧 配置单卡GCU设备")
    
    # 调整batch size
    if hasattr(cfg, 'train_dataloader') and 'batch_size' in cfg.train_dataloader:
        original_batch_size = cfg.train_dataloader.batch_size
        print("📊 每卡batch size: {}".format(original_batch_size))
        print("📊 总batch size: {}".format(original_batch_size * world_size))
    
    print("📁 工作目录: {}".format(cfg.work_dir))
    print("🚀 启动训练 - Rank {}/{}".format(rank, world_size))
    
    # 设置GCU设备
    if torch_gcu is not None:
        torch_gcu.set_device(local_rank)
        print("🔧 设置当前进程GCU设备: {}".format(local_rank))
        
        device = f"xla:{local_rank}"
        cfg.device = device
        
        # 禁用CUDA相关设置
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("🔧 配置设备为: {}".format(device))
    
    # 禁用SyncBatchNorm
    def disable_sync_batchnorm_in_config(config_dict):
        """递归禁用配置中的SyncBatchNorm"""
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if key == 'norm_cfg' and isinstance(value, dict):
                    if value.get('type') == 'SyncBN':
                        print(f"🔧 发现SyncBN配置，替换为BN: {value}")
                        value['type'] = 'BN'
                        print(f"✅ 已替换为: {value}")
                elif isinstance(value, (dict, list)):
                    disable_sync_batchnorm_in_config(value)
        elif isinstance(config_dict, list):
            for item in config_dict:
                disable_sync_batchnorm_in_config(item)
    
    if hasattr(cfg, 'model') and cfg.model is not None:
        disable_sync_batchnorm_in_config(cfg.model)
        print("✅ 已禁用模型配置中的SyncBatchNorm")
    
    disable_sync_batchnorm_in_config(cfg._cfg_dict)
    print("✅ SyncBatchNorm禁用完成，现在使用普通BatchNorm兼容GCU")
    
    # 2. 初始化分布式环境 (让MMEngine按标准方式初始化)
    if cfg.get('launcher', 'none') == 'pytorch':
        from mmengine.dist import init_dist
        init_dist(launcher='pytorch', backend=cfg.env_cfg.dist_cfg.get('backend', 'eccl'))
        print("🔧 MMEngine分布式环境初始化完成")
    
    # 3. 创建 Runner 实例
    print("🚀 创建Runner...")
    runner = Runner.from_cfg(cfg)
    print("✅ Runner创建完成")
    
    # ===== START: 最终修复逻辑 (在Runner创建后，训练开始前) =====
    print("🔧 开始执行最终修复逻辑...")
    
    # 3.1 强制修正分布式后端为 ECCL
    if dist.is_initialized() and dist.get_backend() != 'eccl':
        print(f"⚠️ 检测到错误后端: {dist.get_backend()}，强制切换到 ECCL...")
        current_rank = dist.get_rank()
        current_world_size = dist.get_world_size()
        
        # 销毁当前进程组
        dist.destroy_process_group()
        print("🧹 已销毁当前进程组")
        
        # 重新初始化ECCL后端
        try:
            # 设置ECCL环境变量
            os.environ['ECCL_BACKEND'] = 'eccl'
            os.environ['ECCL_DEVICE_TYPE'] = 'gcu'
            
            dist.init_process_group(
                backend='eccl', 
                init_method='env://', 
                world_size=current_world_size, 
                rank=current_rank
            )
            print(f"✅ 成功切换到 ECCL 后端")
        except Exception as e:
            print(f"❌ ECCL后端初始化失败: {e}")
            print("🔄 回退到gloo后端")
            dist.init_process_group(
                backend='gloo', 
                init_method='env://', 
                world_size=current_world_size, 
                rank=current_rank
            )
    elif dist.is_initialized():
        print(f"✅ 当前后端已是正确的: {dist.get_backend()}")
    
    # 3.2 关键修复：强制将模型移动到正确的GCU设备
    if torch_gcu is not None and hasattr(runner, 'model') and runner.model is not None:
        # 设置GCU设备
        torch_gcu.set_device(local_rank)
        device = f'xla:{local_rank}'
        
        # 强制将模型移动到GCU设备
        runner.model = runner.model.to(device)
        print(f"🔧 模型已强制移动到设备: {device}")
        
        # 验证模型设备
        model_device = next(runner.model.parameters()).device
        print(f"🔍 验证模型设备: {model_device}")
    
    # 3.3 转换SyncBatchNorm层以兼容DDP
    if hasattr(runner, 'model') and runner.model is not None and world_size > 1:
        try:
            from mmengine.model import convert_sync_batchnorm
            runner.model = convert_sync_batchnorm(runner.model)
            print("🔧 SyncBatchNorm层已转换为DDP兼容")
        except Exception as e:
            print(f"⚠️ SyncBatchNorm转换失败: {e}")
    
    # 3.4 关键修复：重新用DDP包装模型（使用正确的参数）
    if world_size > 1 and hasattr(runner, 'model') and runner.model is not None:
        try:
            from mmengine.model import MMDistributedDataParallel
            
            # 检查模型是否已经被DDP包装
            if not isinstance(runner.model, MMDistributedDataParallel):
                # 关键：设置device_ids=None和output_device=None以避免设备不匹配错误
                runner.model = MMDistributedDataParallel(
                    runner.model,
                    device_ids=None,  # 关键：设为None让DDP使用模型当前设备
                    output_device=None  # 关键：设为None避免设备冲突
                )
                print("✅ 模型已在正确的GCU设备上重新包装为DDP")
                
                # 验证DDP包装后的模型设备
                model_device = next(runner.model.parameters()).device
                print(f"🔍 DDP包装后模型设备: {model_device}")
            else:
                print("✅ 模型已经是DDP包装")
        except Exception as e:
            print(f"⚠️ DDP包装失败: {e}")
            print(f"⚠️ 错误详情: {str(e)}")
    
    # ===== END: 最终修复逻辑 =====
    
    # 验证最终状态
    if dist.is_initialized():
        print(f"🔍 最终验证 - 后端: {dist.get_backend()}, Rank: {dist.get_rank()}/{dist.get_world_size()}")
    
    if hasattr(runner, 'model') and runner.model is not None:
        model_device = next(runner.model.parameters()).device
        print(f"🔍 最终验证 - 模型设备: {model_device}")
    
    # 4. 开始训练
    print("🚀 开始训练...")
    runner.train()
    
    # 清理分布式环境
    if dist.is_initialized():
        print("🧹 清理分布式环境...")
        dist.destroy_process_group()
        print("✅ 分布式环境清理完成")

if __name__ == '__main__':
    main()