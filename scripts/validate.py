# scripts/validate.py (V14 - Definitive Manual Load for Runner)

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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- Load Config ---
    cfg = Config.fromfile(args.config)

    # --- Set up Runner ---
    cfg.work_dir = '/kaggle/working/tmp_eval'
    # Set the checkpoint path in the config for the Runner to see initially
    cfg.load_from = args.checkpoint
    cfg.default_scope = 'mmseg'
    
    runner = Runner.from_cfg(cfg)

    # --- KEY CHANGE: Manually load checkpoint to bypass Runner's internal bug ---
    print(f"\nManually loading checkpoint from {args.checkpoint}...")
    # Use torch.load directly with weights_only=False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Manually load the weights into the model that the Runner built
    runner.model.load_state_dict(state_dict, strict=False)
    print("✅ Checkpoint loaded into model manually.")
    
    # --- CRITICAL STEP: Prevent Runner from trying to load it again ---
    # By setting this to None, runner.test() will skip its internal loading
    runner._load_from = None

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