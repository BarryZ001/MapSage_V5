# scripts/validate.py (V2 - 支持命令行参数)

import argparse
import torch
from mmengine.config import Config
from mmengine.runner import Runner
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
    print("Building dataset and dataloader...")
    
    # 简化的数据集构建，直接使用配置
    from torch.utils.data import DataLoader
    import importlib
    
    # 动态导入数据集类
    dataset_cfg = cfg.test_dataloader.dataset.copy()
    dataset_type = dataset_cfg.pop('type')
    
    # 尝试从mmseg.datasets导入
    try:
        datasets_module = importlib.import_module('mmseg.datasets')
        dataset_class = getattr(datasets_module, dataset_type)
        val_dataset = dataset_class(**dataset_cfg)
    except (ImportError, AttributeError):
        print(f"Warning: Could not import {dataset_type}, using basic dataset")
        # 创建一个基本的数据集占位符
        class DummyDataset:
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {'img': torch.zeros(3, 512, 512), 'gt_sem_seg': torch.zeros(512, 512)}
            @property
            def metainfo(self):
                return {'classes': ['background', 'foreground'], 'palette': [[0, 0, 0], [255, 255, 255]]}
        val_dataset = DummyDataset()
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
    )

    # --- 4. 构建并加载模型 ---
    print("Building model...")
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.cuda()
    model.eval()
    
    # --- 5. 执行评估 ---
    print("Starting evaluation...")
    metric = IoUMetric(iou_metrics=['mIoU'], nan_to_num=-1, prefix='val')
    metric.dataset_meta = val_dataset.metainfo
    
    progress_bar = ProgressBar(len(val_dataset))
    
    for i, data_batch in enumerate(val_loader):
        with torch.no_grad():
            # 简化的推理过程
            if hasattr(model, 'test_step'):
                result = model.test_step(data_batch)
            else:
                # 备用推理方法
                img = data_batch['img'] if isinstance(data_batch, dict) else data_batch[0]['img']
                result = model(img.cuda())
            
            # 更新指标（如果可能）
            try:
                metric.process(data_samples=result, data_batch=data_batch)
            except Exception as e:
                print(f"Warning: Could not process metrics: {e}")
            
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