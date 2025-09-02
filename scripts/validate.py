# scripts/validate.py (V10 - Added default_scope)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
try:
    from mmseg.utils import register_all_modules
except ImportError:
    def register_all_modules():
        pass
import mmcv
import os

# Call the registration function once when the script starts
register_all_modules()

def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation validation script using Runner')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- Load Config ---
    cfg = Config.fromfile(args.config)
    
    # --- Set up Runner ---
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif 'work_dir' not in cfg:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
        
    cfg.default_scope = 'mmseg'
    
    runner = Runner.from_cfg(cfg)
    
    # === KEY CHANGE: Load the checkpoint BEFORE calling test() ===
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    runner.load_checkpoint(args.checkpoint)

    # --- Run Validation ---
    print("\nStarting validation using the Runner...")
    # Now, call test() without any arguments
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