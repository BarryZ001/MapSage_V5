# Minimal test configuration for local testing
import os.path as osp

# Custom imports to register components
custom_imports = dict(imports=['mmseg'], allow_failed_imports=False)

# Dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/test_data'  # We'll create dummy data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

# Model configuration
model = dict(
    type='EncoderDecoder',
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
        drop_path_rate=0.1
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

# Validation pipeline (compatible with old mmseg)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

# Dataset configuration for new MMEngine format
# Data configuration for old mmseg format
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline,
        classes=['background', 'object'],
        palette=[[0, 0, 0], [255, 255, 255]]
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',  # Use train data for validation to simplify
        ann_dir='annotations/train',
        pipeline=val_pipeline,
        classes=['background', 'object'],
        palette=[[0, 0, 0], [255, 255, 255]]
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=val_pipeline,
        classes=['background', 'object'],
        palette=[[0, 0, 0], [255, 255, 255]]
    )
)

# Optimizer configuration for old mmseg format
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# Learning rate configuration for old mmseg format
lr_config = dict(policy='poly', power=1.0, min_lr=1e-4, by_epoch=False)

# Training configuration for old mmseg format
runner = dict(type='IterBasedRunner', max_iters=2)
checkpoint_config = dict(by_epoch=False, interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ]
)

# Work directory
work_dir = './work_dirs/test_minimal'

# Disable evaluation for simplicity
evaluation = dict(interval=1000, metric='mIoU')  # Set very high interval to effectively disable

# Other settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True