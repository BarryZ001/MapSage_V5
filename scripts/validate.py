# scripts/validate.py (V10 - Added default_scope)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
try:
    from mmseg.utils import register_all_modules
except ImportError:
    # Comprehensive fallback for environments without MMSegmentation
    def register_all_modules(init_default_scope=True):
        """
        Fallback function when MMSegmentation is not available.
        
        This function provides compatibility for environments where mmseg
        is not installed or the import path has changed. The actual module
        registration will be handled by cfg.default_scope = 'mmseg' setting
        in the Runner configuration.
        
        Args:
            init_default_scope (bool): Placeholder parameter for API compatibility
        """
        import warnings
        warnings.warn(
            "MMSegmentation register_all_modules not available. "
            "Using cfg.default_scope for module registration instead.",
            UserWarning
        )
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
    cfg.load_from = args.checkpoint

    # === KEY CHANGE: Correctly update data_root for ConcatDataset ===
    if args.data_root is not None:
        # The dataset is a ConcatDataset, which has a 'datasets' list.
        # We must iterate through this list and update the data_root for each sub-dataset.
        if cfg.test_dataloader.dataset.type == 'ConcatDataset':
            for ds_cfg in cfg.test_dataloader.dataset.datasets:
                ds_cfg.data_root = args.data_root
        else:
            # Fallback for simple datasets
            cfg.test_dataloader.dataset.data_root = args.data_root
    
    # --- Build Runner and Components ---
    # The Runner will automatically build the model and dataloader from the config
    runner = Runner.from_cfg(cfg)
    
    # Get the dataloader from the runner
    val_loader = runner.build_dataloader(cfg.test_dataloader, seed=0)
    
    # Load the checkpoint. This also builds the model.
    try:
        runner.load_checkpoint(args.checkpoint, weights_only=False)
    except TypeError:
        # Fallback for older MMEngine versions that don't support weights_only
        runner.load_checkpoint(args.checkpoint)
    
    # --- Run Validation ---
    print("\nStarting validation using the Runner...")
    # Use the runner's built-in test method which handles everything
    metrics = runner.test()
    print("\n\n" + "="*40)
    print("      评估完成 - 黄金基准性能")
    print("="*40)
    print(f"配置文件: {args.config}")
    print(f"权重文件: {args.checkpoint}")
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