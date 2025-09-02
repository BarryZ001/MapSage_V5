# configs/train_finetune_baseline.py

# 1. 继承我们最终版的、路径正确的独立配置文件
_base_ = './final_standalone_config.py'

# 2. 定义训练数据加载器
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

# 3. 定义验证数据加载器和评估器
# Validation pipeline
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# Validation dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
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

# Validation evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# 4. 定义一个较短的微调训练流程
# 总迭代次数减少，验证间隔也相应缩短
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 5. 使用较低的学习率进行微调
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),  # 学习率降低
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),  # warmup缩短
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=20000,  # 总迭代次数
        by_epoch=False,
    )
]

# 6. 移除模型中关于ImageNet预训练的定义
# 因为我们将通过 `load_from` 来加载完整的模型
model = dict(backbone=dict(init_cfg=None))

# 7. 运行时设置
default_scope = 'mmseg'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'),
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

# === 关键修改：从清理后的权重文件开始加载 ===
load_from = '/kaggle/working/best_mIoU_iter_6000_cleaned.pth'
resume = False