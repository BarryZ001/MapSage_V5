# scripts/validate.py (V3 - With module registration)

import argparse
import torch
from mmengine.config import Config
from mmengine.dataset import build_dataset
from mmengine.runner import build_dataloader
from mmseg.models import build_segmentor
from mmseg.evaluation.metrics import IoUMetric
from mmseg import register_all_modules  # 1. Import the helper function
import mmcv
import os

# 2. Call the registration function once when the script starts
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
    cfg.load_from = args.checkpoint

    # --- Build Dataloader and Dataset ---
    val_dataset = build_dataset(cfg.test_dataloader.dataset)
    val_loader = build_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False)
    )

    # --- Build and Load Model ---
    model = build_segmentor(cfg.model)
    checkpoint = torch.load(cfg.load_from, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    # --- Run Evaluation ---
    metric = IoUMetric(iou_metrics=['mIoU'])
    metric.dataset_meta = val_dataset.metainfo
    
    progress_bar = mmcv.ProgressBar(len(val_dataset))
    for data in val_loader:
        # Move data to GPU
        # In modern mmseg, data is a dict of lists, tensors need to be moved
        data['inputs'][0] = data['inputs'][0].cuda()
        
        with torch.no_grad():
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
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"mAcc: {metrics['mAcc']:.4f}")
    print(f"aAcc: {metrics['aAcc']:.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()