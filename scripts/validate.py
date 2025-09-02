# scripts/validate.py (V2 - 支持命令行参数)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import DATASETS, MODELS
from mmseg.models import build_segmentor
from mmseg.evaluation import IoUMetric
from mmengine.utils import ProgressBar
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation validation script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-root', help='the root path of the dataset')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- 1. 加载配置 ---
    cfg = Config.fromfile(args.config)
    
    # --- 2. 更新配置文件中的路径 ---
    # 优先使用命令行传入的数据集根目录
    if args.data_root is not None:
        cfg.test_dataloader.dataset.data_root = args.data_root
        
    # 确保配置文件中的权重路径被命令行参数覆盖
    cfg.load_from = args.checkpoint

    # --- 3. 构建数据集和数据加载器 ---
    val_dataset = DATASETS.build(cfg.test_dataloader.dataset)
    # 简化的数据加载器构建
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: x[0]  # 简化的collate函数
    )

    # --- 4. 构建并加载模型 ---
    model = build_segmentor(cfg.model)
    checkpoint = torch.load(cfg.load_from, map_location='cpu')
    
    # 兼容旧版权重文件可能没有'state_dict'的情况
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    # --- 5. 执行评估 ---
    metric = IoUMetric(iou_metrics=['mIoU'])
    metric.dataset_meta = val_dataset.metainfo
    
    progress_bar = ProgressBar(len(val_dataset))
    for data in val_loader:
        # 将数据移动到GPU
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
        
        with torch.no_grad():
            result = model.test_step(data)
            
        metric.process(data_batch=data, data_samples=result)
        progress_bar.update()

    # --- 6. 计算并打印结果 ---
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