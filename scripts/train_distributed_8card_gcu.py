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
            
            # 关键修复：强制使用ECCL后端
            # ECCL是燧原官方为GCU设备专门优化的分布式通信后端
            backend = 'eccl'
            print("🎯 使用燧原官方ECCL后端 (专为GCU设备优化)")
            
            # 检查ECCL后端是否可用
            try:
                # 尝试导入ECCL相关模块
                import torch_gcu.distributed
                print("✅ ECCL分布式模块导入成功")
            except ImportError as e:
                print("⚠️ ECCL模块导入失败: {}".format(e))
                print("🔄 回退到gloo后端")
                backend = 'gloo'
        else:
            print("⚠️ torch_gcu不可用，使用备用后端")
            backend = 'gloo'
    except ImportError as e:
        print("❌ torch_gcu未安装: {}".format(e))
        print("🔄 使用gloo后端")
        backend = 'gloo'
    except Exception as e:
        print("❌ torch_gcu检查失败: {}".format(e))
        print("🔄 使用gloo后端")
        backend = 'gloo'
    
    init_method = 'env://'
    
    print("🔧 初始化分布式进程组:")
    print("  - Backend: {}".format(backend))
    print("  - Init method: {}".format(init_method))
    
    # 初始化分布式进程组
    try:
        # 关键修复：对于ECCL后端，需要特殊的初始化方式
        if backend == 'eccl':
            print("🔧 使用ECCL后端特殊初始化...")
            # 设置ECCL环境变量
            os.environ['ECCL_BACKEND'] = 'eccl'
            os.environ['ECCL_DEVICE_TYPE'] = 'gcu'
            
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        print("✅ 分布式进程组初始化成功")
    except Exception as e:
        print("❌ 分布式进程组初始化失败: {}".format(e))
        # 如果ECCL后端失败，尝试使用gloo作为备选
        if backend == 'eccl':
            print("🔄 ECCL后端失败，尝试使用gloo后端作为备选...")
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
    parser.add_argument('--work-dir', help='工作目录路径')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                       default='pytorch', help='分布式启动器')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程rank')
    args = parser.parse_args()
    
    print("🚀 启动8卡分布式训练")
    print("📄 配置文件: {}".format(args.config))
    print("🔧 启动器: {}".format(args.launcher))
    
    # ===== START: FORCE ECCL BACKEND =====
    # 强制使用ECCL后端，确保与GCU设备兼容
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f'🔧 强制使用ECCL后端进行分布式训练 - Rank {rank}/{world_size}')
        
        # 检查是否已经初始化
        if not dist.is_initialized():
            try:
                # 设置ECCL环境变量
                os.environ['ECCL_BACKEND'] = 'eccl'
                os.environ['ECCL_DEVICE_TYPE'] = 'gcu'
                
                # 强制初始化ECCL后端
                dist.init_process_group(
                    backend='eccl', 
                    init_method='env://', 
                    world_size=world_size, 
                    rank=rank
                )
                print("✅ ECCL后端初始化成功")
            except Exception as e:
                print(f"❌ ECCL后端初始化失败: {e}")
                print("🔄 尝试使用setup_distributed函数")
                # 如果强制初始化失败，回退到原有逻辑
                world_size, rank, local_rank = setup_distributed()
        else:
            print("✅ 分布式进程组已初始化")
    else:
        # 设置分布式环境
        world_size, rank, local_rank = setup_distributed()
    # ===== END: FORCE ECCL BACKEND =====
    
    # 加载配置文件
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
    
    # 更新配置以支持分布式训练
    if world_size > 1:
        cfg.launcher = args.launcher
        print("🔧 启用分布式训练，launcher: {}".format(args.launcher))
        
        # 关键修复：强制MMEngine使用ECCL后端
        # 检查当前使用的后端
        if dist.is_initialized():
            current_backend = dist.get_backend()
            print("🔍 当前分布式后端: {}".format(current_backend))
            
            # 如果当前后端是ECCL，配置MMEngine使用它
            if current_backend == 'eccl':
                # 确保MMEngine的分布式配置使用ECCL
                if not hasattr(cfg, 'env_cfg'):
                    cfg.env_cfg = {}
                if not hasattr(cfg.env_cfg, 'dist_cfg'):
                    cfg.env_cfg.dist_cfg = {}
                
                # 设置后端配置
                cfg.env_cfg.dist_cfg['backend'] = 'eccl'
                print("✅ 强制MMEngine使用ECCL后端")
            else:
                print("⚠️ 当前后端不是ECCL: {}，可能导致XLA设备兼容性问题".format(current_backend))
        
        # 配置GCU设备
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
        
        # 注释掉set_default_device调用，因为它可能与分布式通信冲突
        # 让MMEngine自动处理设备配置
        print("🔧 跳过设置默认设备，让MMEngine自动处理设备配置")
    
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
        
        # 4. 配置MMEngine的设备设置 - T20 XLA设备兼容性修复
        if not hasattr(cfg, 'env_cfg'):
            cfg.env_cfg = {}
        
        # 关键修复：对于XLA设备，使用ECCL后端进行分布式通信
        if torch_gcu is not None:
            # 对于T20 GCU设备，强制使用ECCL后端
            cfg.env_cfg['dist_cfg'] = {'backend': 'eccl', 'init_method': 'env://'}
            print("🔧 T20修复：强制配置ECCL后端用于XLA设备分布式通信")
        else:
            cfg.env_cfg['dist_cfg'] = {'backend': 'gloo'}
        
        # 5. 确保模型包装器使用正确设备
        if hasattr(cfg, 'model_wrapper_cfg'):
            if cfg.model_wrapper_cfg is None:
                cfg.model_wrapper_cfg = {}
            # 不设置device_ids，让MMEngine自动检测
            cfg.model_wrapper_cfg.pop('device_ids', None)
            cfg.model_wrapper_cfg.pop('output_device', None)
        
        print("🔧 配置设备为: {}".format(device))
        print("🔧 配置分布式后端为: eccl")
    
    # 关键修复：在创建Runner前强制设置设备配置
    if torch_gcu is not None:
        print("🔧 T20关键修复：在Runner创建前配置设备...")
        
        # 强制设置当前设备
        torch_gcu.set_device(local_rank)
        
        # 关键修复：对于XLA设备，使用GCU设备进行分布式通信和模型计算
        device = f'gcu:{local_rank}'  # 统一使用GCU设备
        
        # 确保配置中的设备设置正确
        cfg.device = device  # MMEngine使用GCU设备
        
        # 关键修复：完全禁用MMEngine的DDP device_ids设置
        # 让MMEngine自动处理设备配置，避免设备不匹配错误
        if not hasattr(cfg, 'model_wrapper_cfg') or cfg.model_wrapper_cfg is None:
            cfg.model_wrapper_cfg = {}
        
        # 完全移除device_ids和output_device配置
        # 这样MMEngine会自动检测模型所在设备并正确配置DDP
        cfg.model_wrapper_cfg.pop('device_ids', None)
        
        # 不设置device_ids，让MMEngine根据模型实际设备自动配置
        print("🔧 禁用DDP device_ids自动配置，让MMEngine自动检测设备")
        print("🔧 配置模型设备: {}".format(device))
    
    # 创建Runner并开始训练
    print("🚀 创建Runner...")
    
    # 让Runner自己根据配置字典构建模型，不要提前构建
    # 这样可以避免yapf格式化错误，因为cfg.model保持为字典格式
    print("🔧 让Runner自动构建模型，保持cfg.model为配置字典格式")
    
    # ===== START: 禁用DDP的device_ids自动配置 =====
    if cfg.get('launcher') == 'pytorch':
        # 在 MMDistributedDataParallel 的配置中禁用 device_ids
        # 使用model_wrapper_cfg而不是model_wrapper，保持与MMEngine的一致性
        if not hasattr(cfg, 'model_wrapper_cfg') or cfg.model_wrapper_cfg is None:
            cfg.model_wrapper_cfg = {}
        
        # 明确设置DDP配置，禁用device_ids和output_device
        cfg.model_wrapper_cfg.update({
            'type': 'MMDistributedDataParallel',
            'find_unused_parameters': False,
            'device_ids': None,  # 关键：显式设置device_ids为None
            'output_device': None  # 关键：显式设置output_device为None
        })
        print("🔧 已更新model_wrapper_cfg配置，禁用device_ids和output_device自动配置")
    # ===== END: 禁用DDP的device_ids自动配置 =====
    
    # ===== START: 禁用SyncBatchNorm for GCU兼容性 =====
    # 关键修复：在Runner创建前禁用SyncBatchNorm，避免GCU设备兼容性问题
    print("🔧 开始禁用SyncBatchNorm以兼容GCU分布式训练...")
    
    def disable_sync_batchnorm_in_config(config_dict):
        """递归禁用配置中的SyncBatchNorm"""
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if key == 'norm_cfg' and isinstance(value, dict):
                    if value.get('type') == 'SyncBN':
                        print(f"🔧 发现SyncBN配置，替换为BN: {value}")
                        value['type'] = 'BN'  # 使用普通BatchNorm替代SyncBatchNorm
                        print(f"✅ 已替换为: {value}")
                elif isinstance(value, (dict, list)):
                    disable_sync_batchnorm_in_config(value)
        elif isinstance(config_dict, list):
            for item in config_dict:
                disable_sync_batchnorm_in_config(item)
    
    # 禁用模型配置中的SyncBatchNorm
    if hasattr(cfg, 'model') and cfg.model is not None:
        disable_sync_batchnorm_in_config(cfg.model)
        print("✅ 已禁用模型配置中的SyncBatchNorm")
    
    # 禁用其他可能的SyncBatchNorm配置
    disable_sync_batchnorm_in_config(cfg._cfg_dict)
    print("✅ SyncBatchNorm禁用完成，现在使用普通BatchNorm兼容GCU")
    # ===== END: 禁用SyncBatchNorm for GCU兼容性 =====
    
    runner = Runner.from_cfg(cfg)
    
    # 验证Runner创建后的模型设备状态
    if torch_gcu is not None and hasattr(runner, 'model'):
        print("🔍 验证Runner创建后的模型设备状态...")
        
        # 检查模型参数设备
        model_devices = set()
        param_count = 0
        for name, param in runner.model.named_parameters():
            model_devices.add(str(param.device))
            param_count += 1
            if param_count >= 5:  # 检查更多参数确保准确性
                break
        
        print("🔍 模型设备分布: {}".format(model_devices))
        print("🔍 检查了 {} 个参数".format(param_count))
        
        # 如果模型在CPU上，使用正确的GCU API移动到设备
        if any('cpu' in device_str for device_str in model_devices):
            print("🔧 模型在CPU上，移动到GCU设备...")
            
            # 设置当前GCU设备
            torch_gcu.set_device(local_rank)
            
            # 使用XLA设备接口移动模型到GCU设备（T20服务器标准方式）
            xla_device = f'xla:{local_rank}'
            runner.model = runner.model.to(xla_device)
            
            # 再次验证
            verification_devices = set()
            for name, param in runner.model.named_parameters():
                verification_devices.add(str(param.device))
                if len(verification_devices) >= 2:
                    break
            
            print("✅ 模型已移动到GCU设备: {}".format(verification_devices))
        else:
            print("✅ 模型已正确配置在设备上: {}".format(model_devices))
    
    print("✅ Runner创建完成，设备配置验证通过")
    
    runner.train()
    
    # 清理分布式环境
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
        print("🧹 分布式进程组已清理")

if __name__ == '__main__':
    main()