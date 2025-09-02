# scripts/validate.py (V12 - Definitive Final Imports)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
# === KEY CHANGE: Import ConcatDataset from mmengine, LoveDADataset from mmseg ===
from mmengine.dataset import ConcatDataset
from mmseg.datasets import LoveDADataset
from mmseg.models import build_segmentor
import mmcv
import os

# Module registration is handled by cfg.default_scope = 'mmseg' in Runner configuration
# No explicit register_all_modules() call needed in diagnostic mode

def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation validation script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-root', help='the root path of the dataset')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- Load Config ---
    cfg = Config.fromfile(args.config)
    
    # --- Update Paths in Config ---
    # Correctly update data_root for the nested ConcatDataset structure
    if args.data_root is not None:
        if cfg.test_dataloader.dataset.type == 'ConcatDataset':
            for ds_cfg in cfg.test_dataloader.dataset.datasets:
                ds_cfg.data_root = args.data_root
        else:
            cfg.test_dataloader.dataset.data_root = args.data_root
    
    # --- Build Runner and Load Checkpoint Manually ---
    cfg.load_from = args.checkpoint
    cfg.default_scope = 'mmseg'
    runner = Runner.from_cfg(cfg)

    print(f"\nManually loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    runner.model.load_state_dict(state_dict, strict=False)
    print("✅ Checkpoint loaded into model manually.")

    # --- Run Validation ---
    print("\nStarting validation using the Runner...")
    metrics = runner.test()
    
    # --- Print Results ---
    print("\n\n" + "="*40)
    print("      评估完成 - 黄金基准性能")
    print("="*40)
    print(f"配置文件: {args.config}")
    print(f"权重文件: {args.checkpoint}")
    if args.data_root:
        print(f"数据集路径: {args.data_root}")
    print("\n--- 指标 ---")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
            
    print("="*40 + "\n")

if __name__ == '__main__':
    main()