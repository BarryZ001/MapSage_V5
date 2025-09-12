# configs/train_dinov3_backbone.py (最终修正版)

# --- 1. 基础配置 ---
_base_ = './final_standalone_config.py' 

# --- 2. 核心参数定义 ---
num_classes = 7
crop_size = (512, 512) 
data_root = '/kaggle/input/loveda'
dataset_type = 'LoveDADataset'

# --- 3. 数据增强与加载器 (与之前成功配置一致) ---
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
    batch_size=1, 
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=_base_.test_dataloader.dataset)
train_dataloader['dataset']['datasets'][0]['pipeline'] = train_pipeline
train_dataloader['dataset']['datasets'][1]['pipeline'] = train_pipeline

# 验证组件保持不变
val_dataloader = _base_.test_dataloader
val_evaluator = _base_.test_evaluator

# --- 4. 训练策略 ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False)
]

# --- 5. 关键：定义新的模型架构 ---
backbone = dict(
    type='mmpretrain.VisionTransformer',
    arch='l',
    patch_size=16,
    frozen_stages=20, 
    out_type='featmap',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='/kaggle/input/dinov3-vitl16-pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    )
)

decode_head = dict(
    type='SegformerHead',
    in_channels=[1024, 1024, 1024, 1024], 
    in_index=[0, 1, 2, 3],
    channels=256,
    num_classes=num_classes,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', loss_weight=0.3)
    ]
)

model = dict(
    type='EncoderDecoder',
    # === 关键修改：直接在此处完整定义 data_preprocessor ===
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[73.53223947628777, 80.01710095339912, 74.59297778068898],
        std=[41.511366098369635, 35.66528876209687, 33.75830885257866],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512) # 确保训练时有size
    ),
    backbone=backbone,
    decode_head=decode_head,
    test_cfg=_base_.model.test_cfg
)

# === 关键修改：补上缺失的顶层 test_cfg ===
test_cfg = dict(type='TestLoop')

# --- 6. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'))
log_level = 'INFO'
load_from = None
resume = False