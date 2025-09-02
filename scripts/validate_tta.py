#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapSage V4 TTA评估脚本 (v87 - 修正拼写错误)
此版本修正了MixVisionTransformer骨干网络中的 sr_ratios 参数拼写错误
"""

import sys
import os
import traceback
import torch
import numpy as np
import mmcv
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import TRANSFORMS

# ============================== 控制面板 ==============================
CHECKPOINT_PATH = '/kaggle/input/mapsage-stage02-checkpoint-6000/best_mIoU_iter_6000.pth'
# ====================================================================

# -------- 自定义数据转换 --------
@TRANSFORMS.register_module()
class UniformMaskFormat:
    def __init__(self, palette):
        self.palette = {tuple(c[::-1]): i for i, c in enumerate(palette)}
        self.ignore_index = 255
    
    def __call__(self, results):
        gt_seg_map = results.get('gt_seg_map')
        if gt_seg_map is None: 
            return results
        
        if gt_seg_map.ndim == 3 and gt_seg_map.shape[2] == 3:
            mapped_mask = np.full(gt_seg_map.shape[:2], self.ignore_index, dtype=np.uint8)
            for bgr_val, class_id in self.palette.items():
                matches = np.all(gt_seg_map == bgr_val, axis=-1)
                mapped_mask[matches] = class_id
            results['gt_seg_map'] = mapped_mask
        
        if gt_seg_map.ndim == 3 and gt_seg_map.shape[0] == 1:
            results['gt_seg_map'] = gt_seg_map.squeeze()
        
        return results

def main():
    print("\n=== ✍️ 生成 v87 配置 (修正拼写错误) ===")
    
    config_text = f"""
_base_ = ['mmseg::_base_/default_runtime.py']

# 基本配置
dataset_type = 'LoveDADataset'
data_root = '/kaggle/input/loveda'
num_classes = 7
crop_size = (1024, 1024)
palette = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]

data_preprocessor = dict(
    type='SegDataPreProcessor', 
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, 
    pad_val=0, 
    seg_pad_val=255
)

# TTA Pipeline
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [0.75, 1.0, 1.25]
            ],
            [
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
                dict(type='RandomFlip', prob=0.0, direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='UniformMaskFormat', palette=palette)],
            [dict(type='PackSegInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction'))]
        ])
]

# 模型封装
model = dict(
    type='SegTTAModel',
    module=dict(
        type='EncoderDecoder',
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type='MixVisionTransformer', 
            in_channels=3, 
            embed_dims=64, 
            num_stages=4,
            num_layers=[3, 4, 6, 3], 
            num_heads=[1, 2, 5, 8], 
            patch_sizes=[7, 3, 3, 3],
            # --- 核心改动: 修正此处的拼写错误 ---
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3), 
            mlp_ratio=4, 
            qkv_bias=True
        ),
        decode_head=dict(
            type='SegformerHead', 
            in_channels=[64, 128, 320, 512], 
            in_index=[0, 1, 2, 3],
            channels=256, 
            num_classes=num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True), 
            align_corners=False
        ),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768))
    )
)

# 评估时的数据加载器
val_dataloader = dict(
    batch_size=1, 
    num_workers=4, 
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, 
        data_root=data_root,
        data_prefix=dict(img_path='Val', seg_map_path='Val'),
        pipeline=tta_pipeline
    )
)
test_dataloader = val_dataloader

# 评估器与流程配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
"""

    cfg_dir = "configs/v87"
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "v87_tta_final.py")
    
    with open(cfg_path, "w") as f:
        f.write(config_text)
    print(f"✅ 配置写入: {cfg_path}")

    print("\n=== 🚀 启动 v87 TTA评估 (最终修正版) ===")
    work_dir = "/kaggle/working/work_dirs/v87"
    
    try:
        cfg = Config.fromfile(cfg_path)
        cfg.work_dir = work_dir
        
        runner = Runner.from_cfg(cfg)
        
        print(f"--> 正在手动从 {CHECKPOINT_PATH} 加载权重...")
        runner.load_checkpoint(CHECKPOINT_PATH)
        print("--> 权重加载成功！")

        metrics = runner.test()
        print("\n" + "="*60)
        print("🎉 v87 TTA评估完成!")
        print(f"最终评估结果: {metrics}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        print("="*60)
        traceback.print_exc()

if __name__ == "__main__":
    main()