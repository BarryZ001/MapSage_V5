# configs/train_dinov3_backbone.py

# --- 1. 核心参数定义 ---
num_classes = 7
crop_size = (512, 512)
data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

# --- 2. 数据流水线和加载器 ---

# 训练数据流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 测试/验证数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 训练数据集定义
# LoveDA包含两个子集：Rural和Urban
# 我们将它们合并进行训练
rural_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='Train/Rural/images_png', seg_map_path='Train/Rural/masks_png'),
    pipeline=train_pipeline
)

urban_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='Train/Urban/images_png', seg_map_path='Train/Urban/masks_png'),
    pipeline=train_pipeline
)


# 训练数据加载器
train_dataloader = dict(
    batch_size=1,  # DINOv3-Large模型很大，batch_size必须减小
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[rural_train_dataset, urban_train_dataset]
    )
)

# 验证数据集定义
rural_val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='Val/Rural/images_png', seg_map_path='Val/Rural/masks_png'),
    pipeline=test_pipeline
)

urban_val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='Val/Urban/images_png', seg_map_path='Val/Urban/masks_png'),
    pipeline=test_pipeline
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[rural_val_dataset, urban_val_dataset]
    )
)

# 验证评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# --- 3. 训练策略 ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False)
]

# --- 4. 关键：定义新的模型架构 ---
# 数据预处理器
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

# 骨干网络 (Backbone)
backbone = dict(
    type='mmpretrain.VisionTransformer',
    arch='l',  # ViT-Large
    patch_size=16,
    frozen_stages=20,
    out_type='featmap',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/kaggle/input/dinov3-sat-weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        prefix='backbone.'
    )
)

# 解码头 (Decode Head)
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

# 最终模型
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=backbone,
    decode_head=decode_head,
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

# --- 5. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

log_level = 'INFO'
load_from = None
resume = False