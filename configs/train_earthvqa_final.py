# configs/train_earthvqa_final.py

# --- 1. 基础配置 ---
_base_ = './final_standalone_config.py'

# --- 2. 核心参数定义 ---
# 根据EarthVQA官方文档：8个有效类别(1-8) + no-data区域(0) = 9个索引
num_classes = 9
crop_size = (512, 512)
data_root = '/kaggle/input/2024earthvqa/2024EarthVQA'
dataset_type = 'BaseSegDataset'

# --- 2.1 模型配置覆盖 ---
# 覆盖基础配置中的解码头类别数量和数据预处理器
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=crop_size,
        # EarthVQA数据集特定的标准化参数
        mean=[73.53223947628777, 80.01710095339912, 74.59297778068898],
        std=[41.511366098369635, 35.66528876209687, 33.75830885257866],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    decode_head=dict(
        num_classes=num_classes
    )
)

# --- 3. 训练数据增强流程 (完全复刻自您成功的v79.1配置) ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # <-- 关键的数据增强
    dict(type='PackSegInputs')
]

# --- 4. 训练数据加载器 (使用简化的EarthVQA目录结构) ---
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Train/images_png', seg_map_path='Train/masks_png'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture', 'playground')),
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
        metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture', 'playground')),
        pipeline=val_pipeline))
val_evaluator = dict(type='IoUMetric')

# --- 6. 训练策略 (使用较低的学习率进行微调) ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False)
]

# --- 7. 模型定义 (加入复合损失函数) ---
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        # 关键：使用您成功的复合损失函数
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.6),
            dict(type='DiceLoss', loss_weight=0.3),
            dict(type='LovaszLoss', loss_weight=0.1, reduction='none')
        ]
    )
)

# --- 8. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'))
log_level = 'INFO'
# 路径将在Kaggle中动态指定
load_from = None
resume = False