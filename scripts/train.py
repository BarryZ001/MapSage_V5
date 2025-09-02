# scripts/train.py

import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation training script')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    # 从文件加载配置
    cfg = Config.fromfile(args.config)
    
    # 设置默认作用域 (模块注册由配置文件中的default_scope处理)
    cfg.default_scope = 'mmseg'

    # 如果有需要，可以设置工作目录
    if 'work_dir' not in cfg:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # 从配置构建并启动Runner
    runner = Runner.from_cfg(cfg)

    # 开始训练
    runner.train()

if __name__ == '__main__':
    main()