#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 修复distutils.version兼容性问题
try:
    from packaging import version
    import distutils
    if not hasattr(distutils, 'version'):
        distutils.version = version
        print("✅ 修复distutils.version兼容性问题")
except ImportError:
    try:
        import distutils.version
    except ImportError:
        print("⚠️ 无法导入版本处理模块，可能影响TensorBoard功能")

# 添加项目根目录到Python路径
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.runner import Runner

# 尝试导入torch_gcu和ptex
try:
    import torch_gcu  # type: ignore
    import torch_gcu.distributed as gcu_dist  # type: ignore
    print(f"✅ torch_gcu导入成功，可用设备数: {torch_gcu.device_count()}")
    print("✅ torch_gcu.distributed模块导入成功")
    USE_GCU_DISTRIBUTED = True
except ImportError as e:
    print(f"⚠️ torch_gcu导入失败: {e}")
    torch_gcu = None
    gcu_dist = None
    USE_GCU_DISTRIBUTED = False

try:
    import ptex  # type: ignore
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

# 导入自定义模块
try:
    import mmseg_custom.models
    import mmseg_custom.datasets
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"⚠️ 自定义模块导入失败: {e}")


def check_network_connectivity(master_addr, master_port, timeout=10):
    """检查网络连接性"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((master_addr, int(master_port)))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"⚠️ 网络连接检查失败: {e}")
        return False


def setup_distributed_robust(backend='gloo', max_retries=3, retry_delay=5):
    """设置稳定的分布式训练环境"""
    # 获取分布式训练参数
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"🌍 分布式训练设置: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    # 验证环境变量设置
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    print(f"🔧 Master地址: {master_addr}:{master_port}")
    
    # 设置更长的超时时间
    timeout_seconds = 300  # 5分钟超时
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # 指定网络接口
    
    # 如果是多进程分布式训练，初始化进程组
    if world_size > 1:
        # 检查网络连接性（仅对非master节点）
        if rank != 0:
            print(f"🔍 检查与Master节点的网络连接...")
            if not check_network_connectivity(master_addr, master_port):
                print(f"⚠️ 无法连接到Master节点 {master_addr}:{master_port}")
                time.sleep(retry_delay)  # 等待一段时间再重试
        
        # 使用传入的backend参数，默认为gloo
        os.environ['MMENGINE_DDP_BACKEND'] = backend
        print(f"🔧 设置MMEngine DDP后端为: {backend}")
        
        # 多次重试初始化分布式进程组
        for attempt in range(max_retries):
            if not dist.is_initialized():
                try:
                    print(f"🔄 尝试初始化分布式进程组 (第{attempt + 1}次)")
                    
                    # 设置更长的超时时间
                    import datetime
                    timeout = datetime.timedelta(seconds=timeout_seconds)
                    
                    dist.init_process_group(
                        backend=backend,
                        init_method=f"tcp://{master_addr}:{master_port}",
                        world_size=world_size,
                        rank=rank,
                        timeout=timeout
                    )
                    print(f"✅ 分布式进程组初始化完成 - Backend: {backend}")
                    break
                    
                except Exception as e:
                    print(f"❌ 第{attempt + 1}次{backend}后端初始化失败: {e}")
                    if attempt < max_retries - 1:
                        print(f"⏳ 等待{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        print("❌ 所有重试都失败，无法初始化分布式训练")
                        raise
            else:
                print("✅ 分布式进程组已经初始化")
                break
    
    # 设置GCU设备
    if torch_gcu is not None:
        try:
            torch_gcu.set_device(local_rank)
            print(f"🔧 设置GCU设备: {local_rank}")
        except Exception as e:
            print(f"⚠️ 设置GCU设备失败: {e}")
    
    # 设置环境变量
    os.environ['TOPS_VISIBLE_DEVICES'] = str(local_rank)
    print(f"🔧 设置TOPS_VISIBLE_DEVICES: {local_rank}")
    
    # 禁用CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("🔧 禁用CUDA_VISIBLE_DEVICES")
    
    print(f"✅ 分布式环境初始化完成 - Rank {rank}/{world_size}")
    return rank, local_rank, world_size


def cleanup_distributed():
    """清理分布式训练环境"""
    try:
        if dist.is_initialized():
            print("🧹 清理分布式进程组...")
            dist.destroy_process_group()
            print("✅ 分布式进程组清理完成")
    except Exception as e:
        print(f"⚠️ 分布式清理失败: {e}")
    
    # 清理torch_gcu - 修复empty_cache方法不存在的问题
    if torch_gcu is not None:
        try:
            # 检查torch_gcu是否有empty_cache方法
            if hasattr(torch_gcu, 'empty_cache'):
                torch_gcu.empty_cache()
                print("✅ torch_gcu缓存清理完成")
            elif hasattr(torch_gcu, 'synchronize'):
                torch_gcu.synchronize()
                print("✅ torch_gcu同步完成")
            else:
                print("ℹ️ torch_gcu无需清理缓存")
        except Exception as e:
            print(f"⚠️ torch_gcu清理失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='Robust MMSegmentation distributed training script for GCU')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch', help='job launcher')
    parser.add_argument('--backend', choices=['nccl', 'gloo', 'mpi'], default='gloo', help='distributed backend')
    parser.add_argument('--max-retries', type=int, default=3, help='maximum retries for distributed initialization')
    parser.add_argument('--retry-delay', type=int, default=5, help='delay between retries in seconds')
    args = parser.parse_args()

    print("📦 正在初始化稳定的分布式MMSegmentation训练...")
    
    # 设置分布式环境
    rank, local_rank, world_size = setup_distributed_robust(
        backend=args.backend,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    
    try:
        # 从文件加载配置
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"配置文件不存在: {args.config}")
        
        print(f"📝 加载配置文件: {args.config}")
        cfg = Config.fromfile(args.config)
        
        # 清理配置中可能导致pickle错误的模块对象
        def clean_config_for_pickle(config_dict):
            """递归清理配置中不能被pickle的对象"""
            if isinstance(config_dict, dict):
                cleaned = {}
                for key, value in config_dict.items():
                    # 跳过模块对象和函数对象
                    if hasattr(value, '__module__') and not isinstance(value, (str, int, float, bool, list, tuple, dict)):
                        continue
                    elif callable(value) and not isinstance(value, type):
                        continue
                    else:
                        cleaned[key] = clean_config_for_pickle(value)
                return cleaned
            elif isinstance(config_dict, (list, tuple)):
                return [clean_config_for_pickle(item) for item in config_dict]
            else:
                return config_dict
        
        # 备份原始配置中的关键信息
        original_custom_imports = getattr(cfg, 'custom_imports', None)
        
        # 临时移除可能导致pickle问题的custom_imports
        if hasattr(cfg, 'custom_imports'):
            delattr(cfg, 'custom_imports')
            print("🔧 临时移除custom_imports以避免pickle错误")
        
        # 设置工作目录
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = './work_dirs'
        
        # 设置GCU设备
        device_str = None
        if ptex is not None:
            try:
                device_str = "xla"
                print(f"🔧 设置GCU设备为: {device_str}")
            except Exception as e:
                print(f"⚠️ 设置ptex设备失败: {e}")
        elif torch_gcu is not None:
            try:
                device_str = f"gcu:{local_rank}"
                print(f"🔧 设置GCU设备为: {device_str}")
            except Exception as e:
                print(f"⚠️ 设置torch_gcu设备失败: {e}")
        
        # 更新配置以支持分布式训练
        if world_size > 1:
            cfg.launcher = args.launcher
        else:
            cfg.launcher = 'none'
            print("🔧 单进程模式，禁用分布式")
        
        # 调整batch size（每个进程的batch size）
        if hasattr(cfg, 'train_dataloader') and 'batch_size' in cfg.train_dataloader:
            original_batch_size = cfg.train_dataloader.batch_size
            cfg.train_dataloader.batch_size = max(1, original_batch_size // world_size)
            print(f"📊 调整batch size: {original_batch_size} -> {cfg.train_dataloader.batch_size} (per process)")
        
        # 配置MMEngine的分布式设置以正确处理GCU设备
        if world_size > 1 and device_str is not None:
            if not hasattr(cfg, 'model_wrapper_cfg'):
                cfg.model_wrapper_cfg = {}
            
            # 配置分布式包装器，增加稳定性设置
            cfg.model_wrapper_cfg.update({
                'type': 'MMDistributedDataParallel',
                'device_ids': None,
                'broadcast_buffers': False,  # 减少通信开销
                'find_unused_parameters': True,  # 处理未使用的参数
            })
            print("🔧 配置MMEngine分布式包装器")
        
        # 添加分布式训练的稳定性设置
        if world_size > 1:
            # 设置梯度同步频率
            if hasattr(cfg, 'optim_wrapper'):
                if not hasattr(cfg.optim_wrapper, 'accumulative_counts'):
                    cfg.optim_wrapper.accumulative_counts = 1
            
            # 设置检查点保存策略
            if hasattr(cfg, 'default_hooks'):
                if 'checkpoint' in cfg.default_hooks:
                    cfg.default_hooks.checkpoint.save_best = 'auto'
                    cfg.default_hooks.checkpoint.max_keep_ckpts = 3
        
        print("🚀 开始训练...")
        
        # 在创建Runner之前，确保配置可以被深拷贝
        try:
            import copy
            # 测试配置是否可以被深拷贝
            copy.deepcopy(cfg)
            print("✅ 配置深拷贝测试通过")
        except Exception as e:
            print(f"⚠️ 配置深拷贝测试失败: {e}")
            # 如果深拷贝失败，尝试重新构建配置
            print("🔧 尝试重新构建配置...")
            cfg_dict = cfg.to_dict()
            cfg = Config(cfg_dict)
        
        # 创建Runner并开始训练
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print("✅ 训练完成！")
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == '__main__':
    main()