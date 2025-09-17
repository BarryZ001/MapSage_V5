# DINOv3 + MMRS-1M 训练配置文件 - 燧原T20 GCU版本
# 阶段一：基础模型训练，使用DINOv3-ViT-L/16作为backbone
# 针对MMRS-1M多模态遥感数据集进行优化
# 专门适配燧原T20 GCU计算环境

# 导入自定义模块
custom_imports = dict(
    imports=[
        'mmseg_custom.datasets',
        'mmseg_custom.transforms',
        'mmseg',
        'mmseg.models',
        'mmseg.datasets'
    ],
    allow_failed_imports=False
)

# 基础配置
work_dir = './work_dirs/dinov3_mmrs1m_t20_gcu'
exp_name = 'dinov3_mmrs1m_t20_gcu'

# 数据集配置
dataset_type = 'MMRS1MDataset'
data_root = '/workspace/data/mmrs1m/data'  # T20服务器路径
local_data_root = '/Users/barryzhang/myDev3/MapSage_V5/data'  # 本地开发路径

# 图像配置
img_size = (512, 512)
crop_size = (512, 512)
num_classes = 7  # MMRS-1M的类别数

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

# DINOv3-ViT-L/16 模型配置
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    
    # DINOv3 ViT-Large/16 backbone
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/weights/pretrained/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',  # 修正为正确权重路径
            prefix='backbone.'
        )
    ),
    
    # 解码头
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=1024,
        channels=512,
        in_index=-1,
        img_size=img_size,
        embed_dims=1024,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_conv=2,
        upsampling_method='bilinear',
        num_upsampe_layer=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=None  # 可以根据数据集类别分布调整
        )
    ),
    
    # 辅助头
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
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

# 训练数据变换管道
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiModalResize',
        scale=img_size,
        modality='optical',  # 默认光学模态
        keep_ratio=True
    ),
    dict(
        type='MultiModalRandomCrop',
        crop_size=crop_size,
        modality='optical'
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='MultiModalNormalize',
        modality='optical',
        to_rgb=True
    ),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                  'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                  'modality', 'task_type')
    )
]

# 验证数据变换管道
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiModalResize',
        scale=img_size,
        modality='optical',
        keep_ratio=True
    ),
    dict(
        type='MultiModalNormalize',
        modality='optical',
        to_rgb=True
    ),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                  'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                  'modality', 'task_type')
    )
]

# 测试管道
test_pipeline = val_pipeline

# 训练数据加载器
train_dataloader = dict(
    batch_size=4,  # 适配T20 GCU内存限制
    num_workers=4,
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

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
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

# 测试数据加载器
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
        lr=5e-5,  # 适配GCU环境的学习率
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
        end=1500  # warmup步数
    ),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=1500,
        end=80000,  # 总训练步数
        by_epoch=False
    )
]

# 训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,  # 总训练迭代数
    val_interval=2000  # 验证间隔
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认钩子配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=1000,
        show=False,
        wait_time=0.01,
        backend_args=None
    )
)

# 环境配置 - 适配燧原T20 GCU
env_cfg = dict(
    cudnn_benchmark=False,  # GCU环境关闭cudnn
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo')  # 使用gloo后端适配GCU
)

# 可视化后端
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='./work_dirs/dinov3_mmrs1m_t20_gcu/tf_logs'
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
    base_batch_size=32  # 4 GCUs * 4 batch_size * 2 accumulative
)

# 编译配置（GCU环境可能不支持）
# compile_cfg = dict(
#     backend='inductor',
#     mode='reduce-overhead'
# )

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

# 模型EMA配置
model_ema_config = dict(
    enable=True,
    momentum=0.9999,
    device='cpu'  # GCU环境使用CPU进行EMA
)

# 混合精度训练（GCU环境可能需要调整）
fp16 = dict(loss_scale=512.0)

# 梯度累积
accumulative_counts = 2  # 等效batch_size = 4 * 4 * 2 = 32

print(f"🚀 DINOv3 + MMRS-1M 燧原T20 GCU训练配置已加载")
print(f"📊 数据集: {dataset_type}")
print(f"🏗️ 模型: DINOv3-ViT-L/16 + VisionTransformerUpHead")
print(f"💾 工作目录: {work_dir}")
print(f"🔄 最大迭代数: {train_cfg['max_iters']}")
batch_size = 4  # 从train_dataloader配置中获取
print(f"📈 批次大小: {batch_size} x {accumulative_counts} = {batch_size * accumulative_counts}")
print(f"🔥 计算环境: 燧原T20 GCU")