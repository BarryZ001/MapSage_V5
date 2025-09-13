# True Knowledge Distillation Configuration: DINOv3 Teacher -> SegFormer Student
# Optimized for Kaggle P100 16G GPU

# Import mmseg and custom modules to register all components
custom_imports = dict(imports=['mmseg', 'mmseg_custom'], allow_failed_imports=False)

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

# Teacher Model: DINOv3 ViT-Base (Simplified for feature extraction)
teacher_model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,  # Standard ViT input size
        patch_size=16,  # Standard patch size
        out_indices=[2, 5, 8, 11],  # Multi-scale features from different layers
        drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        with_cls_token=True,
        interpolate_mode='bicubic',
        frozen_stages=-1,  # Freeze all stages
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/kaggle/input/dinov3-vitl16-pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
        )
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[768, 768, 768, 768],  # DINOv3 ViT-B features
        in_index=[0, 1, 2, 3],
        channels=512,
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

# Student Model: SegFormer-B0
student_model = dict(
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
        init_cfg=dict(type='Normal', layer='Linear', std=0.01)
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

# Feature Alignment Modules (using Conv2d instead of ConvModule for simplicity)
feature_adapters = [
    dict(
        type='Conv2d',
        in_channels=32,   # Student stage 0
        out_channels=768, # Teacher feature dim
        kernel_size=1,
        bias=False
    ),
    dict(
        type='Conv2d',
        in_channels=64,   # Student stage 1
        out_channels=768,
        kernel_size=1,
        bias=False
    ),
    dict(
        type='Conv2d',
        in_channels=160,  # Student stage 2
        out_channels=768,
        kernel_size=1,
        bias=False
    ),
    dict(
        type='Conv2d',
        in_channels=256,  # Student stage 3
        out_channels=768,
        kernel_size=1,
        bias=False
    )
]

# Knowledge Distillation Model
model = dict(
    type='SegmentationDistiller',
    teacher=teacher_model,
    student=student_model,
    feature_adapters=feature_adapters,
    distill_losses=dict(
        # Feature distillation loss
        feature_loss=dict(
            type='MSELoss',
            loss_weight=10.0,
            reduction='mean'
        ),
        # Attention transfer loss
        attention_loss=dict(
            type='AttentionTransferLoss',
            loss_weight=5.0
        ),
        # Task loss (segmentation)
        task_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    # Temperature for knowledge distillation
    temperature=4.0,
    # Alpha for balancing task loss and distillation loss
    alpha=0.7
)

# Training pipeline
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
    batch_size=2,  # Reduced for distillation memory overhead
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
    max_iters=15000,  # Reduced for distillation
    val_interval=1500
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - only optimize student model
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00004,  # Lower LR for distillation
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'teacher': dict(lr_mult=0.0),  # Freeze teacher
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
        end=800  # Shorter warmup for distillation
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=800,
        end=15000,
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
        interval=1500,
        max_keep_ckpts=3,
        save_best='mIoU'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False)
)

# Mixed precision
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
load_from = None
resume = False

# Work directory
work_dir = '/kaggle/working/work_dirs/train_distill_dinov3_true_kd'

# Default scope
default_scope = 'mmseg'

# Randomness
randomness = dict(seed=0, deterministic=False)