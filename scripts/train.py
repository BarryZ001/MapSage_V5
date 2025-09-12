# scripts/train.py (最终简化版)

import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation training script')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    # 注册所有模块 (非常重要!)
    register_all_modules(init_default_scope=False)

    # 从文件加载配置
    cfg = Config.fromfile(args.config)

    # 设置工作目录
    if 'work_dir' not in cfg:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # 从配置构建并启动Runner
    runner = Runner.from_cfg(cfg)

    # 开始训练
    runner.train()

if __name__ == '__main__':
    main()