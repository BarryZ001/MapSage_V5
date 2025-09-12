# configs/train_distill_dinov3.py (完全独立版本)

# --- 1. 核心参数定义 ---
num_classes = 7
crop_size = (512, 512)
data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

# --- 2. 数据增强与加载器 (复刻自您成功的v79.1配置) ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type, data_root=data_root,
                data_prefix=dict(img_path='Train/Rural/images_png', seg_map_path='Train/Rural/masks_png'),
                pipeline=train_pipeline),
            dict(
                type=dataset_type, data_root=data_root,
                data_prefix=dict(img_path='Train/Urban/images_png', seg_map_path='Train/Urban/masks_png'),
                pipeline=train_pipeline)
        ]))

# --- 3. 验证数据管道和加载器 ---
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

rural_val_dataset = dict(
    type='LoveDADataset',
    data_root='/kaggle/input/loveda',
    data_prefix=dict(img_path='Val/Rural/images_png', seg_map_path='Val/Rural/masks_png'),
    pipeline=test_pipeline
)

urban_val_dataset = dict(
    type='LoveDADataset',
    data_root='/kaggle/input/loveda',
    data_prefix=dict(img_path='Val/Urban/images_png', seg_map_path='Val/Urban/masks_png'),
    pipeline=test_pipeline
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[rural_val_dataset, urban_val_dataset]
    )
)

val_evaluator = dict(type='IoUMetric')

# --- 4. 训练策略 ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=80000, by_epoch=False)
]

# --- 5. 基础骨干网络定义 ---
base_backbone = dict(
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
    drop_path_rate=0.1
)

base_test_cfg = dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))

# --- 6. 教师和学生模型定义 ---
teacher_model = dict(
    type='mmpretrain.VisionTransformer',
    arch='l', patch_size=16, img_size=518,
    out_type='featmap', frozen_stages=-1,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/kaggle/input/dinov3-sat-weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        prefix='backbone.'
    )
)

student_model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[73.53223947628777, 80.01710095339912, 74.59297778068898],
        std=[41.511366098369635, 35.66528876209687, 33.75830885257866],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)
    ),
    backbone=base_backbone,
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512], in_index=[0, 1, 2, 3],
        channels=256, num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True), align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.6),
            dict(type='DiceLoss', loss_weight=0.3),
            dict(type='LovaszLoss', loss_weight=0.1, reduction='none')
        ]
    ),
    test_cfg=base_test_cfg
)

# --- 7. 最终模型：知识蒸馏封装 ---
model = dict(
    type='DistillEncoderDecoder',
    teacher_model=teacher_model,
    student_model=student_model,
    distill_strategy=dict(
        features=dict(
            type='MGDLoss',
            student_channels=512,
            teacher_channels=1024,
            loss_weight=5.0
        )
    )
)

# --- 8. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, save_best='mIoU'))
log_level = 'INFO'
load_from = '/kaggle/working/best_mIoU_iter_6000_cleaned.pth'
resume = False
work_dir = './work_dirs/train_distill_dinov3'