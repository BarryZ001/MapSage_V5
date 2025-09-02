# configs/train_segformer_b2_imagenet.py

# --- 1. 继承基础配置 ---
# 我们不再手动定义模型，而是从我们已验证的独立配置中"导入"模型定义
# 这确保了模型结构与我们验证时完全一致
_base_ = './final_standalone_config.py'

# --- 2. 修改/添加训练专用配置 ---

# 数据集路径 (硬编码为Kaggle路径，简化训练脚本)
data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

# 训练数据处理流程
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True), # 训练尺寸为512x512
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

# 训练数据加载器
train_dataloader = dict(
    batch_size=4,  # 在T4 GPU上可以适当增加batch size
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

# 验证数据处理流程
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs'),
]

# 验证数据加载器 (复用我们已验证的配置)
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

# 验证评估器 (复用我们已验证的配置)
val_evaluator = dict(type='IoUMetric')

# --- 3. 定义训练循环和超参数 ---

# 训练循环设置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop') # 只是定义，本次不使用

# 优化器封装 (AdamW)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2)) # 梯度裁剪

# 学习率调度器 (Cosine + Warmup)
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

# --- 4. 修改模型配置以加载预训练权重 ---
# 加载在ImageNet上预训练的SegFormer-B2骨干网络权重
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8add0.pth')
    )
)

# --- 5. 默认运行时设置 ---
default_scope = 'mmseg'
# 设置默认的钩子 (hooks)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 环境设置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 日志和可视化
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False