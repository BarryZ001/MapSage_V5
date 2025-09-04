# configs/resume_earthvqa_kaggle.py (最终迁移学习版)

# 1. 继承我们最终版的、路径正确的独立配置文件
_base_ = './final_standalone_config.py' 

# 2. 定义训练数据加载器 (使用简化的EarthVQA目录结构)
data_root = '/kaggle/input/2024earthvqa/2024EarthVQA'
dataset_type = 'BaseSegDataset' 
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='Train/images_png',
            seg_map_path='Train/masks_png'),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline))

# 3. 定义验证配置（简化路径）
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Val/images_png', seg_map_path='Val/masks_png'),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=val_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# 4. 定义一个较短的微调训练流程
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 5. 使用较低的学习率进行微调
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=20000,
        by_epoch=False,
    )
]

# 6. 移除模型中关于ImageNet预训练的定义
model = dict(backbone=dict(init_cfg=None))


# 7. 运行时设置
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'))
log_level = 'INFO'

# === 关键修改：使用 load_from 加载权重，并确保 resume 为 False ===
load_from = '/kaggle/input/temp-8000tier/iter_8000.pth'
resume = False