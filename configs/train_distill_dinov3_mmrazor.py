# MMRazor-based Knowledge Distillation Configuration for DINOv3 -> SegFormer
# This configuration uses MMRazor framework for knowledge distillation

_base_ = [
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# Set default scope to mmrazor
default_scope = 'mmrazor'

# Crop size
crop_size = (512, 1024)

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

# Student model (SegFormer-B0)
student = dict(
    type='mmseg.EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
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
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mit_b0.pth')
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Teacher model (DINOv3 ViT-Large)
teacher = dict(
    type='mmpretrain.VisionTransformer',
    arch='large',
    img_size=518,
    patch_size=14,
    out_indices=-1,
    final_norm=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='pretrained/dinov3_vitl14_pretrain.pth'
    )
)

# Algorithm configuration using MMRazor
model = dict(
    scope='mmrazor',
    type='GeneralDistill',
    architecture=dict(
        type='MMSegArchitecture',
        model=student
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='backbone',
                teacher_module='',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_backbone',
                        tau=1,
                        loss_weight=3.0
                    )
                ]
            ),
            dict(
                student_module='decode_head',
                teacher_module='',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_kl_head',
                        tau=4,
                        loss_weight=2.0
                    )
                ]
            )
        ]
    )
)

# Training settings
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,
    val_interval=8000
)

val_cfg = dict(
    type='mmrazor.SingleTeacherDistillValLoop'
)

test_cfg = dict(
    type='TestLoop'
)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
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
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False
    )
]

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=3
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# Enable automatic mixed precision
fp16 = dict(loss_scale=512.)

# Find unused parameters
find_unused_parameters = True

# Load teacher checkpoint
load_from = None
resume_from = None

# Environment settings
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Visualization settings
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Log settings
log_processor = dict(by_epoch=False)
log_level = 'INFO'