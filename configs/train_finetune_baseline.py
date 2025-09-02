# configs/train_finetune_baseline.py (V2 - With Data Preprocessor Size)

_base_ = './final_standalone_config.py'

data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

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

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

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

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=20000,
        by_epoch=False,
    )
]

model = dict(
    # === KEY CHANGE: Added 'size' to the data_preprocessor ===
    data_preprocessor=dict(
        size=(512, 512),
        type='SegDataPreProcessor',
        mean=[73.53223947628777, 80.01710095339912, 74.59297778068898],
        std=[41.511366098369635, 35.66528876209687, 33.75830885257866],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    backbone=dict(init_cfg=None))

default_scope = 'mmseg'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

log_level = 'INFO'
load_from = '/kaggle/working/best_mIoU_iter_6000_cleaned.pth'
resume = False