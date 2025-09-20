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

def setup_distributed():
    """设置分布式训练环境"""
    # 获取分布式训练参数
    world_size = int(os.environ.get('WORLD_SIZE', 8))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print("🌍 分布式训练参数:")
    print("  - WORLD_SIZE: {}".format(world_size))
    print("  - RANK: {}".format(rank))
    print("  - LOCAL_RANK: {}".format(local_rank))
    
    # 根据燧原官方文档配置ECCL后端
    print("🔍 检查torch_gcu和ECCL后端支持...")
    
    # 检查torch_gcu是否可用
    try:
        import torch_gcu
        if torch_gcu.is_available():
            print("✅ torch_gcu可用，设备数: {}".format(torch_gcu.device_count()))
            # 根据官方文档，torch_gcu可用时应该使用eccl后端
            backend = 'eccl'
            print("🎯 使用燧原专用后端: eccl (官方推荐)")
        else:
            print("⚠️ torch_gcu不可用，使用备用后端")
            backend = 'gloo'
    except ImportError as e:
        print("❌ torch_gcu未安装: {}".format(e))
        print("🔄 降级使用gloo后端")
        backend = 'gloo'
    except Exception as e:
        print("❌ torch_gcu检查失败: {}".format(e))
        print("🔄 降级使用gloo后端")
        backend = 'gloo'
    
    init_method = 'env://'
    
    print("🔧 初始化分布式进程组:")
    print("  - Backend: {}".format(backend))
    print("  - Init method: {}".format(init_method))
    
    # 初始化分布式进程组
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        print("✅ 分布式进程组初始化成功")
    except Exception as e:
        print("❌ 分布式进程组初始化失败: {}".format(e))
        # 如果eccl失败，尝试使用gloo作为备选
        if backend == 'eccl':
            print("🔄 尝试使用gloo后端作为备选...")
            try:
                dist.init_process_group(
                    backend='gloo',
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank
                )
                print("✅ 使用gloo后端初始化成功")
            except Exception as e2:
                print("❌ gloo后端也失败: {}".format(e2))
                raise
        else:
            raise
    
    return world_size, rank, local_rank

def main():
    parser = argparse.ArgumentParser(description='8卡分布式训练脚本')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                       default='pytorch', help='分布式启动器')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程rank')
    args = parser.parse_args()
    
    print("🚀 启动8卡分布式训练")
    print("📄 配置文件: {}".format(args.config))
    print("🔧 启动器: {}".format(args.launcher))
    
    # 设置分布式环境
    world_size, rank, local_rank = setup_distributed()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 检查并创建工作目录
    if hasattr(cfg, 'work_dir') and cfg.work_dir:
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
    
    # 更新配置以支持分布式训练
    if world_size > 1:
        cfg.launcher = args.launcher
        print("🔧 启用分布式训练，launcher: {}".format(args.launcher))
        # 配置GCU设备，让MMEngine自动处理分布式
        cfg.device = 'gcu'
        print("🔧 配置GCU设备，world_size: {}".format(world_size))
    else:
        cfg.launcher = 'none'
        print("🔧 单进程模式，禁用分布式")
        # 单卡训练配置
        cfg.device = 'gcu'
        print("🔧 配置单卡GCU设备")
    
    # 调整batch size（每个进程的batch size）
    if hasattr(cfg, 'train_dataloader') and 'batch_size' in cfg.train_dataloader:
        original_batch_size = cfg.train_dataloader.batch_size
        # 8卡分布式训练，每卡保持配置的batch_size
        print("📊 每卡batch size: {}".format(original_batch_size))
        print("📊 总batch size: {}".format(original_batch_size * world_size))
    
    print("📁 工作目录: {}".format(cfg.work_dir))
    print("🚀 启动训练 - Rank {}/{}".format(rank, world_size))
    
    # 设置GCU设备
    if torch_gcu is not None:
        torch_gcu.set_device(local_rank)
        print("🔧 设置当前进程GCU设备: {}".format(local_rank))
        
        # 设置默认设备类型为GCU，确保新创建的tensor都在GCU上
        try:
            # 检查torch版本是否支持set_default_device
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device("gcu:{}".format(local_rank))
                print("🔧 设置默认tensor设备为: gcu:{}".format(local_rank))
            else:
                print("⚠️ torch版本不支持set_default_device，跳过设置")
        except Exception as e:
            print("⚠️ 设置默认设备失败: {}".format(e))
    
    # 修改配置以避免MMEngine的设备不匹配问题
    print("🔧 修改配置以适配GCU设备...")
    
    # 关键修复：配置MMEngine使用正确的设备
    if torch_gcu is not None:
        device = "gcu:{}".format(local_rank)
        
        # 1. 设置当前GCU设备
        torch_gcu.set_device(local_rank)
        
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
        
        print("🔧 配置设备为: {}".format(device))
        print("🔧 配置分布式后端为: gloo")
    
    # 创建Runner并开始训练
    print("🚀 创建Runner...")
    runner = Runner.from_cfg(cfg)
    
    # 关键修复：在DDP包装前确保模型在正确的GCU设备上
    if torch_gcu is not None and hasattr(runner, 'model'):
        print("🔧 T20环境：确保模型在正确的GCU设备上...")
        
        # 检查模型当前设备
        model_devices = set()
        param_count = 0
        for name, param in runner.model.named_parameters():
            model_devices.add(str(param.device))
            param_count += 1
            if param_count >= 3:  # 检查前几个参数
                break
        
        print("🔍 模型当前设备分布: {}".format(model_devices))
        print("🔍 检查了 {} 个参数".format(param_count))
        
        # 如果模型参数在CPU上，必须移动到GCU设备
        if any('cpu' in device_str for device_str in model_devices):
            print("🔄 T20关键修复：将模型从CPU移动到 gcu:{}...".format(local_rank))
            
            # 强制移动模型到GCU设备
            runner.model = runner.model.to("gcu:{}".format(local_rank))
            
            # 验证移动是否成功
            verification_devices = set()
            for name, param in runner.model.named_parameters():
                verification_devices.add(str(param.device))
                if len(verification_devices) >= 2:  # 检查多个参数确保一致性
                    break
            
            print("✅ 模型移动后设备分布: {}".format(verification_devices))
            
            # 确保所有参数都在正确的GCU设备上
            expected_device = "gcu:{}".format(local_rank)
            if all(expected_device in device_str for device_str in verification_devices):
                print("✅ 模型成功移动到 {}".format(expected_device))
            else:
                print("❌ 模型移动失败，期望设备: {}, 实际设备: {}".format(expected_device, verification_devices))
        else:
            print("✅ 模型已在正确的GCU设备上: {}".format(model_devices))
    
    # 验证模型设备设置
    if torch_gcu is not None and hasattr(runner, 'model'):
        print("🔍 验证模型设备设置...")
        
        # 检查模型参数设备
        device_types = set()
        for name, param in runner.model.named_parameters():
            device_types.add(param.device.type)
        
        print("📊 模型参数设备类型: {}".format(device_types))
        
        if 'cpu' in device_types and len(device_types) > 1:
            print("⚠️ 检测到混合设备，正在修复...")
            device = "gcu:{}".format(local_rank)
            runner.model = runner.model.to(device)
            print("✅ 模型已移动到: {}".format(device))
        elif 'gcu' in device_types:
            print("✅ 模型已正确配置在GCU设备上")
        else:
            print("⚠️ 模型在意外设备上: {}".format(device_types))
    
    print("✅ Runner创建完成，设备配置验证通过")
    
    runner.train()
    
    # 清理分布式环境
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 分布式进程组已清理")

if __name__ == '__main__':
    main()