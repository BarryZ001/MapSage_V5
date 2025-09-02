# scripts/validate.py (V10 - Added default_scope)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
try:
    from mmseg.utils import register_all_modules
except ImportError:
    # Fallback for environments without mmsegmentation
    def register_all_modules():
        """Dummy function for compatibility when mmseg is not available"""
        print("Warning: mmsegmentation not found, using fallback registration")
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

    # --- Set up Runner (but don't load checkpoint with it yet) ---
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif 'work_dir' not in cfg:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    cfg.default_scope = 'mmseg'
    
    # The Runner will build the model structure
    runner = Runner.from_cfg(cfg)

    # === KEY CHANGE: Manually load checkpoint with the correct arguments ===
    print(f"\nManually loading checkpoint from {args.checkpoint}...")
    # Use torch.load directly with weights_only=False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Extract the state_dict (weights)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Manually load the weights into the model that the Runner built
    runner.model.load_state_dict(state_dict, strict=False)
    print("✅ Checkpoint loaded into model manually.")

    # --- Run Validation ---
    print("\nStarting validation using the Runner...")
    # The runner.test() method will now use the model with the weights we just loaded
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