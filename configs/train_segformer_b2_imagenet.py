# configs/train_segformer_b2_imagenet.py (V2 - Corrected Checkpoint URL)

# --- 1. Inherit from our final, verified inference config ---
_base_ = './final_standalone_config.py'

# --- 2. Modify/add training-specific configurations ---

# Dataset paths are now correctly defined in the base config,
# but we need to define the training pipeline and dataloader here.
data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

# Training data processing pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

# Training dataloader
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='Train/Rural/images_png',
                    seg_map_path='Train/Rural/masks_png'),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='Train/Urban/images_png',
                    seg_map_path='Train/Urban/masks_png'),
                pipeline=train_pipeline)
        ]))

# Validation data processing pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Validation dataloader (reusing our verified configuration)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='Val/Rural/images_png',
                    seg_map_path='Val/Rural/masks_png'),
                pipeline=val_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='Val/Urban/images_png',
                    seg_map_path='Val/Urban/masks_png'),
                pipeline=val_pipeline)
        ]))

# Validation evaluator (reusing our verified configuration)
val_evaluator = dict(type='IoUMetric')

# --- 3. Define training loop and hyperparameters ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

# --- 4. KEY CHANGE: Update the model's pretrained checkpoint URL ---
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            # This is the new, correct URL for the MiT-B2 backbone
            checkpoint='https://download.openmmlab.com/mmpretrain/v1.0/mit/mit-b2_3rdparty_8xb128-in1k_20221122-cba83a22.pth'
        )
    )
)

# --- 5. Default runtime settings ---
default_scope = 'mmseg'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False