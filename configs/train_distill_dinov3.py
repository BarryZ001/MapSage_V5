# configs/train_distill_dinov3.py (最终修正版)

# --- 1. 基础配置 ---
_base_ = './final_standalone_config.py'

# --- 2. 核心参数定义 ---
num_classes = 7
crop_size = (512, 512)
data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

# --- 3. 数据增强与加载器 ---
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

# 验证组件保持不变
val_dataloader = _base_.test_dataloader
val_evaluator = _base_.test_evaluator

# --- 4. 训练策略 ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=80000, by_epoch=False)
]

# --- 5. 关键：定义教师和学生模型 ---
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
    backbone=_base_.model.backbone,
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
    test_cfg=_base_.model.test_cfg
)

# --- 6. 最终模型：知识蒸馏封装 ---
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

# === 关键修改：补上缺失的顶层 test_cfg ===
test_cfg = dict(type='TestLoop')

# --- 7. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000, save_best='mIoU'))
log_level = 'INFO'
load_from = '/kaggle/working/best_mIoU_iter_6000_cleaned.pth'
resume = False