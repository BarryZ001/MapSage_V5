# scripts/validate.py (V7 - Runner-based)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
import mmcv
import os

# Call the registration function once when the script starts
register_all_modules()

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
    if args.data_root is not None:
        cfg.test_dataloader.dataset.data_root = args.data_root
    
    # Use a temporary work directory
    cfg.work_dir = '/kaggle/working/tmp_eval'
    
    # --- Build Runner and Components ---
    # The Runner will automatically build the model and dataloader from the config
    runner = Runner.from_cfg(cfg)
    
    # Get the dataloader from the runner
    val_loader = runner.build_dataloader(cfg.test_dataloader, seed=0)
    
    # Load the checkpoint. This also builds the model.
    runner.load_checkpoint(args.checkpoint, weights_only=True)
    model = runner.model
    
    # Get the evaluation metric from the config
    metric = runner.build_metric(cfg.test_evaluator)
    metric.dataset_meta = val_loader.dataset.metainfo
    
    # --- Run Evaluation ---
    model.eval()
    progress_bar = mmcv.ProgressBar(len(val_loader.dataset))
    for data in val_loader:
        with torch.no_grad():
            # Let the model handle data movement to the correct device
            result = model.test_step(data)
        
        metric.process(data_batch=data, data_samples=result)
        progress_bar.update()

    # --- Compute and Print Results ---
    metrics = metric.compute_metrics(metric.results)
    print("\n\n" + "="*40)
    print("      评估完成 - 黄金基准性能")
    print("="*40)
    print(f"配置文件: {args.config}")
    print(f"权重文件: {args.checkpoint}")
    print(f"数据集路径: {cfg.test_dataloader.dataset.data_root}")
    print("\n--- 指标 ---")
    
    # Print metrics in a more readable format
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
            
    print("="*40 + "\n")

if __name__ == '__main__':
    main()