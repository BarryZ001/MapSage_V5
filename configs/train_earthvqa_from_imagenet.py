# configs/train_earthvqa_from_imagenet.py

# --- 1. 基础配置 ---
_base_ = './final_standalone_config.py'

# --- 2. 核心参数定义 ---
num_classes = 8
crop_size = (512, 512)
data_root = '/kaggle/input/2024earthvqa/2024EarthVQA'
dataset_type = 'BaseSegDataset'

# --- 3. 训练数据增强流程 ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# --- 4. 训练数据加载器 ---
train_dataloader = dict(
    batch_size=4,  # P100 16G显存对于512x512尺寸，batch_size=4是比较合适的
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Train/images_png', seg_map_path='Train/masks_png'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural', 'playground')),
        pipeline=train_pipeline))

# --- 5. 验证组件 ---
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

# --- 6. 训练策略 ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=80000, by_epoch=False)
]

# --- 7. 模型定义 ---
# 关键：从我们上传到Kaggle的ImageNet权重开始加载
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            # 路径指向您在第一步中创建的Kaggle数据集中的文件
            checkpoint='/kaggle/input/mit-b2-imagenet-weights/mit-b2_in1k-20230209-4d95315b.pth'
        )),
    decode_head=dict(num_classes=num_classes))

# --- 8. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, save_best='mIoU'))
log_level = 'INFO'
load_from = None
resume = False