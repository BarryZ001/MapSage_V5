# scripts/train.py (Distillation Fix)

import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# === KEY CHANGE: Explicitly import the distillers module to register its components ===
from mmseg.models import distillers  # noqa

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation training script')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    # Register all modules (very important!)
    register_all_modules(init_default_scope=False)

    # Load config from file
    cfg = Config.fromfile(args.config)

    # Set work_dir if not specified in config
    if 'work_dir' not in cfg:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # Build and start the Runner from the config
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()

if __name__ == '__main__':
    main()