#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8卡分布式训练脚本 - 燧原T20 GCU最终修正版
集成了所有必要的修复，以确保在T20硬件上稳定运行
基于用户提供的简洁版本优化
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

# --------------------------------------------------------------------
# 脚本初始化
# --------------------------------------------------------------------
# 添加项目根目录到Python路径，确保自定义模块能被找到
sys.path.insert(0, '.')

# 导入框架和GCU相关库
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.model import convert_sync_batchnorm, MMDistributedDataParallel

try:
    import torch_gcu
    print(f"✅ torch_gcu 导入成功，设备数量: {torch_gcu.device_count()}")
except ImportError as e:
    print(f"❌ 错误: torch_gcu 导入失败: {e}")
    print("⚠️ 将使用CPU模式运行")
    torch_gcu = None

# --------------------------------------------------------------------
# 主函数
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='8卡分布式训练脚本-燧原T20 GCU最终修正版')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录路径，如果指定将覆盖配置文件中的设置')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                       default='pytorch', help='分布式启动器')
    # torchrun 会自动提供 LOCAL_RANK，这里保留以兼容旧用法
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # --- 步骤 1: 加载配置文件 ---
    print("📄 加载配置文件...")
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    # 确保工作目录存在
    if hasattr(cfg, 'work_dir') and cfg.work_dir:
        os.makedirs(cfg.work_dir, exist_ok=True)
        print(f"📁 工作目录: {cfg.work_dir}")

    # --- 步骤 2: 强制初始化 ECCL 分布式后端 (这是第一个关键修复) ---
    # 必须在 MMEngine Runner 初始化之前，手动建立正确的分布式通信后端
    rank = 0
    world_size = 1
    local_rank = 0
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"🌍 分布式环境检测:")
        print(f"   - RANK: {rank}")
        print(f"   - WORLD_SIZE: {world_size}")
        print(f"   - LOCAL_RANK: {local_rank}")
        
        if not dist.is_initialized():
            print(f"🔧 [Rank {rank}] 初始化分布式进程组，使用燧原官方推荐的 'eccl' 后端...")
            try:
                # 设置ECCL环境变量
                os.environ['ECCL_BACKEND'] = 'eccl'
                os.environ['ECCL_DEVICE_TYPE'] = 'gcu'
                
                dist.init_process_group(backend='eccl', init_method='env://')
                print(f"✅ [Rank {rank}] ECCL 后端初始化成功")
                
                # 验证后端
                actual_backend = dist.get_backend()
                if actual_backend != 'eccl':
                    print(f"⚠️ 后端验证失败: 期望 'eccl'，实际 '{actual_backend}'")
                    
            except Exception as e:
                print(f"❌ ECCL 后端初始化失败: {e}")
                print("🔄 尝试使用 gloo 后端作为备选...")
                try:
                    dist.init_process_group(backend='gloo', init_method='env://')
                    print(f"✅ [Rank {rank}] gloo 后端初始化成功")
                except Exception as e2:
                    print(f"❌ gloo 后端也失败: {e2}")
                    raise
        else:
            current_backend = dist.get_backend()
            print(f"✅ [Rank {rank}] 分布式进程组已初始化，后端: {current_backend}")
    else:
        print("⚠️ 未检测到分布式环境变量，将以单卡模式运行。")

    # --- 步骤 3: 配置MMEngine使用正确的分布式设置 ---
    if world_size > 1:
        cfg.launcher = args.launcher
        
        # 配置分布式环境
        if not hasattr(cfg, 'env_cfg'):
            cfg.env_cfg = {}
        if not hasattr(cfg.env_cfg, 'dist_cfg'):
            cfg.env_cfg.dist_cfg = {}
        
        # 根据实际初始化的后端设置配置
        if dist.is_initialized():
            actual_backend = dist.get_backend()
            cfg.env_cfg.dist_cfg['backend'] = actual_backend
            cfg.env_cfg.dist_cfg['init_method'] = 'env://'
            print(f"🔧 配置MMEngine使用 {actual_backend} 后端")
    
    # 配置设备
    if torch_gcu is not None:
        cfg.device = f'gcu:{local_rank}'
        print(f"🔧 配置设备: {cfg.device}")
        
        # 关键修复：在Runner创建前设置当前GCU设备
        torch_gcu.set_device(local_rank)
        print(f"🔧 [Rank {rank}] 预设当前GCU设备: gcu:{local_rank}")
        
        # 注意：不使用torch.cuda.set_device，因为T20环境没有NVIDIA驱动
        # 只使用GCU特定的设备设置
        print(f"🔧 [Rank {rank}] 使用GCU设备设置，跳过CUDA调用")
        
    else:
        cfg.device = 'cpu'
        print("🔧 配置设备: CPU")

    # --- 步骤 4: 创建 MMEngine Runner ---
    print("🚀 创建 MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    print("✅ Runner 创建成功")

    # ===== START: FINAL FIX LOGIC (基于用户提供的完整解决方案) =====
    
    if torch_gcu is not None and hasattr(runner, 'model') and runner.model is not None:
        # 1. Get the local rank and set the current device for this process
        device = f'gcu:{local_rank}'
        torch_gcu.set_device(local_rank)
        print(f"🔧 [Rank {rank}] 设置当前设备为: {device}")

        # 2. Force the model onto the correct GCU
        print(f"🔧 [Rank {rank}] 强制将模型移动到GCU设备...")
        runner.model.to(device)
        
        # 验证模型设备
        model_device = next(runner.model.parameters()).device
        print(f"✅ [Rank {rank}] 模型现在位于设备: {model_device}")

        # 3. Convert BatchNorm layers to be DDP-compatible
        if world_size > 1:
            print(f"🔧 [Rank {rank}] 转换BatchNorm层为SyncBatchNorm...")
            runner.model = convert_sync_batchnorm(runner.model)
            print(f"✅ [Rank {rank}] BatchNorm层转换完成")

            # 4. Manually re-wrap the model with the correct settings
            print(f"🔧 [Rank {rank}] 手动重新包装模型为DDP...")
            runner.model = MMDistributedDataParallel(
                runner.model,
                device_ids=None,  # Critical: Set to None to use the current device
                output_device=None  # Critical: Also set to None
            )
            print(f"✅ [Rank {rank}] 模型已成功包装为MMDistributedDataParallel")
        else:
            print(f"✅ [Rank {rank}] 单卡训练，跳过DDP包装")
    else:
        print("⚠️ 跳过GCU设备配置（torch_gcu不可用或模型为空）")

    # ===== END: FINAL FIX LOGIC =====
    
    # --- 步骤 6: 最终验证 ---
    if dist.is_initialized():
        print(f"🔍 最终验证 - 后端: {dist.get_backend()}, Rank: {dist.get_rank()}/{dist.get_world_size()}")
    
    if hasattr(runner, 'model') and runner.model is not None:
        model_device = next(runner.model.parameters()).device
        print(f"🔍 最终验证 - 模型设备: {model_device}")
        print(f"🔍 最终验证 - 模型类型: {type(runner.model).__name__}")
    
    # --- 步骤 7: 开始训练 ---
    # 此时 runner.model 已经是完全配置正确的DDP模型，可以直接开始训练
    print(f"🎉 [Rank {rank}] 所有准备工作完成，开始训练！")
    runner.train()

    # --- 步骤 8: 清理分布式环境 ---
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"🧹 [Rank {rank}] 分布式进程组已清理")

# --------------------------------------------------------------------
# 脚本入口
# --------------------------------------------------------------------
if __name__ == '__main__':
    main()