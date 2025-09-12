# configs/train_dinov3_backbone.py (完全独立版本)

# --- 1. 核心参数定义 ---
num_classes = 7
crop_size = (512, 512)
data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

# --- 2. 数据增强管道 ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# --- 3. 数据集定义 ---
train_dataset = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path='Train/Rural/images_png', seg_path='Train/Rural/masks_png'),
            pipeline=train_pipeline
        ),
        dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path='Train/Urban/images_png', seg_path='Train/Urban/masks_png'),
            pipeline=train_pipeline
        )
    ]
)

val_dataset = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path='Val/Rural/images_png', seg_path='Val/Rural/masks_png'),
            pipeline=test_pipeline
        ),
        dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path='Val/Urban/images_png', seg_path='Val/Urban/masks_png'),
            pipeline=test_pipeline
        )
    ]
)

# --- 4. 数据加载器 ---
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)

# --- 5. 评估器 ---
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# --- 6. 训练策略 ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False)
]

# --- 7. 模型架构 ---
backbone = dict(
    type='mmpretrain.VisionTransformer',
    arch='l',
    patch_size=16,
    frozen_stages=20,
    out_type='featmap',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/kaggle/input/dinov3-vitl16-pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        prefix='backbone.'
    )
)

decode_head = dict(
    type='SegformerHead',
    in_channels=[1024, 1024, 1024, 1024],
    in_index=[0, 1, 2, 3],
    channels=256,
    num_classes=num_classes,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', loss_weight=0.3)
    ]
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[73.53223947628777, 80.01710095339912, 74.59297778068898],
        std=[41.511366098369635, 35.66528876209687, 33.75830885257866],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size
    ),
    backbone=backbone,
    decode_head=decode_head,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

# --- 8. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU')
)
log_level = 'INFO'
load_from = None
resume = False