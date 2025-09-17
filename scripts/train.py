# scripts/train.py (简化版 - 修复导入问题)

import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner

# 导入mmseg来触发所有注册
try:
    import mmseg  # type: ignore
    from mmseg.models import *  # type: ignore
    from mmseg.datasets import *  # type: ignore
except ImportError as e:
    print(f"⚠️ 模块导入失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation training script')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    print("📦 正在初始化MMSegmentation模块...")
    print("✅ MMSegmentation模块初始化完成")
    
    # 从文件加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    print(f"📝 加载配置文件: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 设置工作目录
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dirs'
    
    print(f"📁 工作目录: {cfg.work_dir}")
    
    # 创建Runner并开始训练
    print("🚀 开始训练...")
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    print("✅ 训练完成!")

if __name__ == '__main__':
    main()