# configs/train_dinov3_backbone.py (完全独立版本)

# --- 1. 核心参数定义 ---
num_classes = 7
crop_size = (512, 512)
data_root = '/workspace/data/loveda'
dataset_type = 'LoveDADataset'

# --- 2. 数据增强管道 ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
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
            data_prefix=dict(img_path='Train/Rural/images_png', seg_map_path='Train/Rural/masks_png'),
            pipeline=train_pipeline
        ),
        dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path='Train/Urban/images_png', seg_map_path='Train/Urban/masks_png'),
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
            data_prefix=dict(img_path='Val/Rural/images_png', seg_map_path='Val/Rural/masks_png'),
            pipeline=test_pipeline
        ),
        dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path='Val/Urban/images_png', seg_map_path='Val/Urban/masks_png'),
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

test_dataloader = val_dataloader

# --- 5. 评估器 ---
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# --- 6. 训练策略 ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False)
]

# --- 4. 模型配置 ---
backbone = dict(
    type='DINOv3ViT',
    arch='large',
    img_size=512,
    patch_size=16,
    out_indices=[23],  # Last layer output for ViT-Large
    interpolate_mode='bicubic',
    init_cfg=dict(type='Normal', layer='Linear', std=0.01)
)

# 解码头
decode_head = dict(
    type='FCNHead',
    in_channels=1024,
    in_index=0,
    channels=256,
    num_convs=1,
    concat_input=False,
    dropout_ratio=0.1,
    num_classes=num_classes,
    norm_cfg=dict(type='BN', requires_grad=True),
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
    ]
)

# 完整模型
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
    test_cfg=dict(mode='whole')
)

# --- 8. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU')
)
log_level = 'INFO'
load_from = None
resume = False