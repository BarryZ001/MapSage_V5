# scripts/validate.py (V11 - Definitive Final Version)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.datasets import LoveDADataset, ConcatDataset
from mmseg.models import build_segmentor
import mmcv
import os

# Handle register_all_modules import with fallback
try:
    from mmseg.utils import register_all_modules
    register_all_modules()
except ImportError:
    # Fallback when mmseg.utils.register_all_modules is not available
    def register_all_modules():
        pass
    register_all_modules()

def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation validation script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    # === KEY CHANGE: Add the --data-root argument back in ===
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