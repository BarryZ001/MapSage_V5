# scripts/validate.py (V13 - Simplified, No Path Argument)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
import mmcv
import os

register_all_modules()

def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation validation script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    # --data-root argument is no longer needed
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- Load Config (Now with the correct hardcoded path) ---
    cfg = Config.fromfile(args.config)

    # --- Set up Runner ---
    cfg.work_dir = '/kaggle/working/tmp_eval'
    cfg.load_from = args.checkpoint
    cfg.default_scope = 'mmseg'
    
    runner = Runner.from_cfg(cfg)

    # --- Run Validation ---
    print("\nStarting validation using the Runner...")
    metrics = runner.test()
    
    # --- Print Results ---
    print("\n\n" + "="*40)
    print("      评估完成 - 黄金基准性能")
    print("="*40)
    print(f"配置文件: {args.config}")
    print(f"权重文件: {args.checkpoint}")
    print("\n--- 指标 ---")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
            
    print("="*40 + "\n")

if __name__ == '__main__':
    main()