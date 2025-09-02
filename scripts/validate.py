# scripts/validate.py (V7 - Manual Control)

import argparse
import torch
from mmengine.config import Config
try:
    from mmengine.dataset import build_dataloader
except ImportError:
    try:
        from mmengine.runner import build_dataloader
    except ImportError:
        # Fallback: 如果都导入失败，我们手动实现一个简单版本
        def build_dataloader(dataset, **kwargs):
            from torch.utils.data import DataLoader
            return DataLoader(dataset, **kwargs)
# 关键修改：直接导入我们将要手动创建的Dataset类
from mmseg.datasets import LoveDADataset, ConcatDataset
from mmseg.models import build_segmentor
from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import MODELS
import mmcv
import os

# Register all modules
try:
    from mmseg.utils import register_all_modules
    register_all_modules()
except ImportError:
    # Fallback registration
    pass

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
    cfg.load_from = args.checkpoint

    # --- Manually Build Datasets for Full Control ---
    print("\n--- Manually Building Datasets ---")

    # 从配置中获取Rural和Urban各自的定义
    rural_cfg = cfg.test_dataloader.dataset.datasets[0]
    urban_cfg = cfg.test_dataloader.dataset.datasets[1]
    
    # 确保传入的data_root覆盖配置文件中的默认值
    if args.data_root:
        rural_cfg.data_root = args.data_root
        urban_cfg.data_root = args.data_root

    # 手动创建Rural验证集
    print("Building Rural dataset...")
    rural_dataset = LoveDADataset(**rural_cfg)
    print(f"✅ Rural dataset loaded. Found {len(rural_dataset)} images.")

    # 手动创建Urban验证集
    print("Building Urban dataset...")
    urban_dataset = LoveDADataset(**urban_cfg)
    print(f"✅ Urban dataset loaded. Found {len(urban_dataset)} images.")

    # 手动将它们合并
    print("Combining datasets...")
    val_dataset = ConcatDataset(datasets=[rural_dataset, urban_dataset])
    print(f"🎉 Final combined dataset size: {len(val_dataset)}\n")

    # --- Build Dataloader ---
    # 使用我们手动创建好的 val_dataset
    val_loader = build_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False)
    )

    # --- Build and Load Model ---
    model = build_segmentor(cfg.model)
    checkpoint = torch.load(cfg.load_from, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    # --- Run Evaluation ---
    metric = IoUMetric(iou_metrics=['mIoU'])
    # Set dataset metadata
    try:
        metric.dataset_meta = val_dataset.metainfo
    except AttributeError:
        # Fallback for metainfo access
        metric.dataset_meta = getattr(val_dataset.datasets[0], 'metainfo', None)
    
    progress_bar = mmcv.ProgressBar(len(val_dataset))
    for data in val_loader:
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
    # ... [rest of the printing logic remains the same]
    print(f"配置文件: {args.config}")
    print(f"权重文件: {args.checkpoint}")
    print(f"数据集路径: {args.data_root}")  # Use args.data_root for clarity
    print("\n--- 指标 ---")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"mAcc: {metrics['mAcc']:.4f}")
    print(f"aAcc: {metrics['aAcc']:.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()