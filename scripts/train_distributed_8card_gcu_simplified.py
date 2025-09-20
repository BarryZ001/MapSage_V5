#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8卡分布式训练脚本 - 燧原T20 GCU最终修正版 V3
通过在Runner初始化前注入正确配置，引导MMEngine使用正确的DDP参数。
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

# 添加项目根目录到Python路径
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.runner import Runner

try:
    import torch_gcu
    print(f"✅ torch_gcu 导入成功，设备数量: {torch_gcu.device_count()}")
except ImportError as e:
    print(f"❌ 错误: torch_gcu 导入失败: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='8卡分布式训练脚本-燧原T20 GCU最终修正版 V3')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录路径')
    parser.add_argument('--launcher', choices=['pytorch'], default='pytorch', help='分布式启动器')
    args = parser.parse_args()

    # --- 步骤 1: 加载配置文件 ---
    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if hasattr(cfg, 'work_dir') and cfg.work_dir:
        os.makedirs(cfg.work_dir, exist_ok=True)

    # --- 步骤 2: 初始化分布式后端 ---
    # 这是所有分布式操作的第一步，并且必须在MMEngine的任何操作之前完成。
    if 'RANK' in os.environ:
        print("🔧 初始化分布式进程组，使用燧原官方推荐的 'eccl' 后端...")
        dist.init_process_group(backend='eccl', init_method='env://')
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        print(f"✅ [Rank {rank}] ECCL 后端初始化成功。")
    else:
        local_rank = 0
        
    # --- 步骤 3: 关键修复 - 在Runner创建前，向配置中注入GCU适配信息 ---
    # 这是解决所有问题的核心所在
    
    # 3.1 强制指定设备为 'gcu'，MMEngine会使用此配置将模型移动到设备上
    cfg.device = 'gcu'
    
    # 3.2 强制指定分布式后端为 'eccl'
    if not hasattr(cfg, 'env_cfg'):
        cfg.env_cfg = {}
    cfg.env_cfg['dist_cfg'] = {'backend': 'eccl'}
    
    # 3.3 强制指定DDP包装器的参数，禁用device_ids自动分配
    # MMEngine会使用这个配置来创建MMDistributedDataParallel
    cfg.model_wrapper_cfg = dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=False,
        device_ids=None,  # 关键：设置为None，让DDP使用当前进程的设备
        output_device=None # 关键：同样设置为None
    )
    
    # 3.4 禁用SyncBN (可选，但推荐用于解决潜在的SyncBN问题)
    # 如果您仍然遇到SyncBatchNorm相关错误，请取消下面的注释
    # from mmengine.model import revert_sync_batchnorm
    # cfg.model = revert_sync_batchnorm(cfg.model)
    # print("🔧 已将模型中的SyncBatchNorm转换为普通BatchNorm")

    print(f"🔧 [Rank {local_rank}] 所有GCU适配配置已注入。")

    # --- 步骤 4: 创建并运行 Runner ---
    # 现在cfg对象已经包含了所有正确的GCU适配信息
    # Runner.from_cfg() 会读取这些信息并执行正确的初始化流程
    print("🚀 创建 MMEngine Runner...")
    runner = Runner.from_cfg(cfg)
    
    # 验证模型是否在正确的设备上
    model_device = next(runner.model.parameters()).device
    print(f"✅ [Rank {local_rank}] Runner创建成功，模型位于设备: {model_device}")
    
    if 'cpu' in str(model_device):
        print(f"❌ [Rank {local_rank}] 致命错误: 模型仍然在CPU上！请检查MMEngine版本兼容性。")
        sys.exit(1)

    print(f"🎉 [Rank {local_rank}] 所有准备工作完成，即将开始训练！")
    runner.train()

    # --- 步骤 5: 清理分布式环境 ---
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()