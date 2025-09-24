#!/usr/bin/env python3
"""
DINOv3 + MMRS-1M 单卡测试配置文件
基于train_dinov3_mmrs1m_t20_gcu_8card.py，但针对单卡测试进行了优化
"""

# 自定义模块导入
custom_imports = dict(
    imports=[
        'mmseg_custom.datasets',
        'mmseg_custom.transforms',
        'mmseg_custom.models'  # 只导入自定义模型，避免与mmseg官方模块冲突
    ],
    allow_failed_imports=False
)

# 工作目录和实验名称
work_dir = './work_dirs/dinov3_mmrs1m_single_test'
exp_name = 'dinov3_mmrs1m_single_test'

# 数据集配置
dataset_type = 'MMRS1MDataset'
data_root = '/workspace/data/mmrs1m/data'  # T20服务器路径
local_data_root = '/Users/barryzhang/myDev3/MapSage_V5/data'  # 本地开发路径

# 图像和类别配置
img_size = (512, 512)
crop_size = (512, 512)
num_classes = 7  # MMRS-1M的类别数

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# 数据预处理器
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  # ImageNet统计值
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

# DINOv3-ViT-L/16 模型配置（单卡测试版本）
model = dict(
    type='CustomEncoderDecoder',  # 使用自定义的EncoderDecoder
    data_preprocessor=data_preprocessor,
    
    # DINOv3 backbone
    backbone=dict(
        type='DINOv3ViT',
        arch='large',
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        with_cls_token=True,
        output_cls_token=False,
        interpolate_mode='bicubic',
        out_indices=(23,),  # 输出最后一层
        final_norm=True,
        drop_path_rate=0.1,
        init_cfg=None  # 移除预训练权重配置，使用默认初始化
    ),
    
    # 解码头（使用BN而不是SyncBN）
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=-1,
        img_size=img_size,
        embed_dims=1024,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),  # 单卡测试使用BN
        num_conv=2,
        upsampling_method='bilinear',
        num_upsample_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=None  # 可以根据数据集类别分布调整
        )
    ),
    
    # 辅助头（使用BN而不是SyncBN）
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),  # 单卡测试使用BN
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),
    
    # 训练和测试配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 训练数据管道
train_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='CustomLoadAnnotations'),
    dict(
        type='CustomResize',
        img_scale=img_size,
        keep_ratio=True
    ),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75
    ),
    dict(type='CustomRandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # 恢复光度变换增强
    dict(type='CustomNormalize', **img_norm_cfg),
    dict(type='CustomPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='CustomDefaultFormatBundle'),
    dict(type='CustomCollect', keys=['img', 'gt_semantic_seg'])
]

# 验证数据管道
val_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='CustomLoadAnnotations'),
    dict(
        type='CustomResize',
        img_scale=img_size,
        keep_ratio=True
    ),
    dict(type='CustomNormalize', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundle'),
    dict(type='CustomCollect', keys=['img', 'gt_semantic_seg'])
]

# 测试数据管道
test_pipeline = val_pipeline

# 数据加载器配置（单卡版本）
train_dataloader = dict(
    batch_size=1,  # 单卡测试使用小batch_size
    num_workers=2,  # 减少worker数量
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        task_type='classification',  # 开始时使用分类任务
        modality='optical',
        instruction_format=True,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        task_type='classification',
        modality='optical',
        instruction_format=True,
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore']
)
test_evaluator = val_evaluator

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,  # 单卡测试使用较小学习率
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # backbone使用较小学习率
            'norm': dict(decay_mult=0.0),   # 不对norm层进行权重衰减
            'bias': dict(decay_mult=0.0),   # 不对bias进行权重衰减
        }
    )
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=100  # 单卡测试减少warmup步数
    ),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=100,
        end=1000,  # 单卡测试减少总步数
        by_epoch=False
    )
]

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1000,  # 单卡测试减少总迭代数
    val_interval=200  # 验证间隔
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=200,
        max_keep_ckpts=2,
        save_best='mIoU',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=100,
        show=False,
        wait_time=0.01,
        backend_args=None
    )
)

# 环境配置（单卡版本）
env_cfg = dict(
    cudnn_benchmark=False,  # GCU环境下禁用cudnn
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='eccl'),  # 保持eccl后端配置
    resource_limit=4096
)

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='./work_dirs/dinov3_mmrs1m_single_test/tf_logs'
    )
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 日志配置
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None  # 从头开始训练
resume = False

# 随机种子
randomness = dict(seed=42)

# 自动学习率缩放
auto_scale_lr = dict(
    enable=True,
    base_batch_size=1  # 单卡测试
)

# 多模态配置
multimodal_config = dict(
    modalities=['optical', 'sar', 'infrared'],
    modality_weights=[0.6, 0.3, 0.1],  # 不同模态的权重
    cross_modal_learning=True,
    modal_specific_augmentation=True
)

# 指令配置
instruction_config = dict(
    enable_instruction_tuning=True,
    instruction_templates=[
        "What is the category of this remote sensing image?",
        "Classify this satellite image.",
        "Identify the land cover type in this image.",
        "What type of terrain is shown in this remote sensing data?"
    ],
    response_format='single_word'
)

# 蒸馏配置
distillation_config = dict(
    enable=False,  # 第一阶段不使用蒸馏
    teacher_model=None,
    distill_loss_weight=0.5,
    temperature=4.0
)

# EMA配置
model_ema_config = dict(
    enable=False,  # 暂时禁用EMA以避免设备冲突
    momentum=0.9999
)

# 梯度累积
accumulative_counts = 1  # 单卡测试不需要梯度累积

print(f"🚀 DINOv3 + MMRS-1M 单卡测试配置已加载")
print(f"📊 数据集: {dataset_type}")
print(f"🏗️ 模型: DINOv3-ViT-L/16 + VisionTransformerUpHead")
print(f"💾 工作目录: {work_dir}")
print(f"🔄 最大迭代数: {train_cfg['max_iters']}")
batch_size = 1  # 从train_dataloader配置中获取
print(f"📈 批次大小: {batch_size} (单卡测试)")
print(f"🔥 计算环境: 燧原T20 GCU - 单卡测试")

# 设备配置
device = None  # 由训练脚本动态设置为gcu:0

print(f"⚙️ 设备配置: 动态分配 - 训练脚本将设置为gcu:0")