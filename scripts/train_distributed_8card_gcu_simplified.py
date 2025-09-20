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
from mmengine.model import convert_sync_batchnorm, MMDistributedDataParallel
from mmengine.registry import MODELS

try:
    import torch_gcu  # type: ignore
    print(f"✅ torch_gcu 导入成功，设备数量: {torch_gcu.device_count()}")
except ImportError as e:
    print(f"❌ 错误: torch_gcu 导入失败: {e}")
    print("⚠️ 请确保在燧原T20环境中运行此脚本")
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

    # --- 步骤 2: 手动初始化 ECCL 分布式后端 ---
    # 这是所有分布式操作的第一步，并且必须在MMEngine的任何操作之前完成。
    if 'RANK' in os.environ:
        dist.init_process_group(backend='eccl', init_method='env://')
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        print(f"✅ [Rank {rank}] ECCL 分布式后端初始化成功。")
    else:
        local_rank = 0
        
    # --- 步骤 3: 手动设置当前设备 ---
    # 告诉torch_gcu和PyTorch当前进程应该使用哪张卡。
    torch_gcu.set_device(local_rank)
    device = f'gcu:{local_rank}'
    print(f"🔧 [Rank {local_rank}] 当前设备已设置为: {device}")
    
    # --- 步骤 4: 手动构建模型并完成所有适配 ---
    # 我们自己构建模型，不再依赖 runner.from_cfg() 的自动构建。
    
    # 4.1 从配置字典构建模型实例 (此时模型在CPU上)
    model = MODELS.build(cfg.model)
    print(f"🔧 [Rank {local_rank}] 模型已从配置构建 (位于CPU)")

    # 4.2 将模型移动到指定的GCU设备
    model.to(device)
    print(f"🔧 [Rank {local_rank}] 模型已移动到: {device}")

    # 4.3 转换SyncBatchNorm层 (如果是多卡训练)
    if dist.is_initialized():
        model = convert_sync_batchnorm(model)
        print(f"🔧 [Rank {local_rank}] 模型中的BatchNorm层已转换为SyncBatchNorm")

    # 4.4 手动用DDP包装模型 (如果是多卡训练)
    if dist.is_initialized():
        model = MMDistributedDataParallel(
            model,
            device_ids=None,
            output_device=None
        )
        print(f"✅ [Rank {local_rank}] 模型已成功包装为DDP")

    # --- 步骤 5: 创建 Runner 并传入准备好的模型 ---
    # 注意：这里我们不再使用 Runner.from_cfg()，而是直接初始化Runner，
    # 并将我们手动准备好的、完全配置正确的模型作为参数传入。
    runner = Runner(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=cfg.train_dataloader,
        val_dataloader=cfg.val_dataloader,
        val_evaluator=cfg.val_evaluator,
        train_cfg=cfg.train_cfg,
        val_cfg=cfg.val_cfg,
        test_cfg=cfg.test_cfg,
        optim_wrapper=cfg.optim_wrapper,
        param_scheduler=cfg.param_scheduler,
        default_hooks=cfg.default_hooks,
        env_cfg=cfg.env_cfg,
        visualizer=cfg.visualizer,
        log_processor=cfg.log_processor,
        launcher=args.launcher
        # ... 其他你需要的参数 ...
    )
    
    # --- 步骤 6: 开始训练 ---
    print(f"🎉 [Rank {local_rank}] 所有准备工作完成，即将开始训练！")
    runner.train()

    # --- 步骤 7: 清理分布式环境 ---
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()