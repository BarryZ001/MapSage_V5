# configs/train_earthvqa_finetune.py (V2 - Using BaseSegDataset)

_base_ = './final_standalone_config.py' 

num_classes = 8 
crop_size = (512, 512) 
data_root = '/kaggle/input/2024earthvqa/2024EarthVQA'
# === KEY CHANGE 1: Use the generic BaseSegDataset ===
dataset_type = 'BaseSegDataset' 

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Train/images_png', seg_map_path='Train/masks_png'),
        # === KEY CHANGE 2: Add suffixes, since BaseSegDataset doesn't know them by default ===
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural', 'playground')),
        pipeline=train_pipeline))

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Val/images_png', seg_map_path='Val/masks_png'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural', 'playground')),
        pipeline=val_pipeline))
val_evaluator = dict(type='IoUMetric')

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=500, end=40000, by_epoch=False)
]

model = dict(decode_head=dict(num_classes=num_classes))

default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'))
log_level = 'INFO'
load_from = '/kaggle/working/best_mIoU_iter_6000_cleaned.pth'
resume = False