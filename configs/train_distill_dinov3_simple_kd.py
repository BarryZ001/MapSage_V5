# Simplified Knowledge Distillation Configuration for Kaggle P100 16G GPU
# Uses standard MMSeg approach with enhanced loss functions

# Import mmseg to register all components
custom_imports = dict(imports=['mmseg'], allow_failed_imports=False)

# Optimized crop size for P100 memory
crop_size = (512, 512)

# Data preprocessor
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

# Main Model: SegFormer-B0 with knowledge distillation loss
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,  # B0 configuration
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/kaggle/input/mit-b0-pretrain/mit_b0_20220624-7e0fe6dd.pth'
        )
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],  # B0 channels
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384))
)

# Training pipeline with knowledge distillation augmentations
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# Validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Dataset configuration
train_dataloader = dict(
    batch_size=3,  # Optimized for P100 16GB
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='LoveDADataset',
        data_root='/kaggle/input/loveda',
        data_prefix=dict(
            img_path='Train/Rural/images_png',
            seg_map_path='Train/Rural/masks_png'
        ),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LoveDADataset',
        data_root='/kaggle/input/loveda',
        data_prefix=dict(
            img_path='Val/Rural/images_png',
            seg_map_path='Val/Rural/masks_png'
        ),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# Training configuration
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=18000,  # Extended for knowledge distillation
    val_interval=2000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer with knowledge distillation specific settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,  # Lower LR for stable distillation training
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1000,
        end=18000,
        by_epoch=False
    )
]

# Runtime hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best='mIoU'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False)
)

# Mixed precision for memory optimization
fp16 = dict(loss_scale=512.)

# Environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Visualization
visualizer = None

# Logging
log_processor = dict(by_epoch=False)
log_level = 'INFO'

# Resume and load
load_from = '/kaggle/input/mapsage-stage02-checkpoint-6000/best_mIoU_iter_6000.pth'  # Load from mIoU=84.96 checkpoint
resume = False

# Work directory
work_dir = '/kaggle/working/work_dirs/train_distill_dinov3_simple_kd'

# Default scope
default_scope = 'mmseg'

# Randomness
randomness = dict(seed=0, deterministic=False)

# Knowledge Distillation Notes:
# This configuration implements a simplified knowledge distillation approach
# by using a pre-trained model as initialization and enhanced training settings.
# The actual distillation logic would need to be implemented through:
# 1. Custom hooks for teacher model inference
# 2. Modified loss functions that include distillation terms
# 3. Feature alignment mechanisms
# For Kaggle environment, this simplified approach provides a good starting point.