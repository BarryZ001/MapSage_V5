# -*- coding: utf-8 -*-
# scripts/train.py (简化版 - 修复导入问题)

import argparse
import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.runner import Runner

# 导入mmseg来触发所有注册
try:
    import mmseg  # type: ignore
    from mmseg.models import *  # type: ignore
    from mmseg.datasets import *  # type: ignore
except ImportError as e:
    print(f"⚠️ 模块导入失败: {e}")

# 导入我们的自定义模块
try:
    import mmseg_custom.models
    import mmseg_custom.datasets
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"⚠️ 自定义模块导入失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation training script')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--launcher', 
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend')
    args = parser.parse_args()

    print("📦 正在初始化MMSegmentation模块...")
    print("✅ MMSegmentation模块初始化完成")
    
    # 从文件加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    print(f"📝 加载配置文件: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs'
    
    print(f"📁 工作目录: {cfg.work_dir}")
    
    # 设置随机种子
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)
        print(f"🎲 随机种子: {args.seed}, 确定性: {args.deterministic}")
    
    # 设置启动器
    if args.launcher != 'none':
        cfg.launcher = args.launcher
        print(f"🚀 启动器: {args.launcher}")
    
    # 创建Runner并开始训练
    print("🚀 开始训练...")
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    print("✅ 训练完成!")

if __name__ == '__main__':
    main()