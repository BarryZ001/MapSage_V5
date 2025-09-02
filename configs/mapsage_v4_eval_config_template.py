# MapSage V4 模型配置文件模板
# 适用于CPU推理环境，包含滑窗推理设置
# 请根据您的实际模型架构调整相关参数

# 基础配置继承（如果使用MMSegmentation标准配置）
# _base_ = [
#     '_base_/models/segformer_mit-b2.py',
#     '_base_/datasets/loveda.py',
#     '_base_/default_runtime.py'
# ]

# 数据集类型和路径
dataset_type = 'LoveDADataset'
data_root = 'data/LoveDA'  # 如果需要的话

# 图像归一化配置（ImageNet预训练标准）
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # ImageNet mean
    std=[58.395, 57.12, 57.375],     # ImageNet std
    to_rgb=True
)

# 类别数量（LoveDA数据集）
num_classes = 7

# 模型配置 - SegFormer-B2 示例
model = dict(
    type='EncoderDecoder',
    
    # 主干网络配置
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mit_b2.pth')  # 如果有预训练权重
    ),
    
    # 解码头配置
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],  # 对应B2的通道数
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            ignore_index=255
        )
    ),
    
    # 辅助头配置（可选）
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=320,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=num_classes,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
    #         ignore_index=255
    #     )
    # ),
    
    # 训练配置
    train_cfg=dict(),
    
    # 测试配置 - 关键：滑窗推理设置
    test_cfg=dict(
        mode='slide',           # 滑窗模式
        crop_size=(1024, 1024), # 滑窗大小
        stride=(768, 768),      # 滑窗步长（重叠区域）
        # crop_size=(512, 512),   # 如果内存不足，可以使用更小的窗口
        # stride=(384, 384),      # 对应的步长
    )
)

# 数据预处理管道
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),  # 可以根据需要调整
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],  # 多尺度测试（可选）
        flip=False,  # 是否翻转（TTA的一部分）
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

# 数据配置
data = dict(
    samples_per_gpu=1,  # CPU推理建议使用1
    workers_per_gpu=1,  # CPU环境建议使用较少的worker
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',  # 根据实际路径调整
        ann_dir='ann_dir/test',  # 根据实际路径调整
        pipeline=test_pipeline
    )
)

# 评估配置
evaluation = dict(
    interval=1000,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU'
)

# 运行时配置
runtime = dict(
    # 日志配置
    log_config=dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            # dict(type='TensorboardLoggerHook')  # 如果需要tensorboard
        ]
    ),
    
    # 分布式配置
    dist_params=dict(backend='nccl'),
    log_level='INFO',
    load_from=None,  # 预训练权重路径
    resume_from=None,  # 恢复训练路径
    workflow=[('train', 1)],
    
    # CPU优化设置
    cudnn_benchmark=False,  # CPU环境设为False
    mp_start_method='fork',  # macOS推荐使用fork
)

# 优化器配置（如果需要fine-tuning）
optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

# 学习率调度器配置
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# 检查点保存配置
checkpoint_config = dict(
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=3
)

# 总训练步数（如果需要训练）
total_iters = 160000

# 类别名称（用于可视化）
classes = [
    'background',    # 背景
    'building',      # 建筑
    'road',          # 道路
    'water',         # 水体
    'barren',        # 贫瘠土地
    'forest',        # 森林
    'agricultural'   # 农业
]

# 调色板（RGB格式）
palette = [
    [255, 255, 255],  # 背景 - 白色
    [255, 0, 0],      # 建筑 - 红色
    [255, 255, 0],    # 道路 - 黄色
    [0, 0, 255],      # 水体 - 蓝色
    [159, 129, 183],  # 贫瘠土地 - 紫色
    [0, 255, 0],      # 森林 - 绿色
    [255, 195, 128]   # 农业 - 橙色
]

# 如果使用EarthVQA预训练或其他特殊配置，请在此处添加
# 例如：
# model['backbone']['init_cfg'] = dict(
#     type='Pretrained', 
#     checkpoint='path/to/earthvqa_pretrained.pth'
# )

# CPU特定优化
# 如果在CPU上运行，建议设置以下参数：
# - 减小batch_size
# - 减少worker数量
# - 使用较小的crop_size
# - 关闭CUDA相关设置