# configs/train_earthvqa_finetune.py

# --- 1. 基础配置 ---
_base_ = './final_standalone_config.py'

# --- 2. 核心参数定义 ---
num_classes = 8
crop_size = (512, 512)
data_root = '/kaggle/input/2024earthvqa/2024EarthVQA'
dataset_type = 'LoveDADataset'

# --- 3. 训练数据增强流程 (与之前相同) ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# --- 4. 训练数据加载器 (使用简化的EarthVQA目录结构) ---
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Train/images_png', seg_map_path='Train/masks_png'),
        pipeline=train_pipeline))

# --- 5. 验证组件 (同样使用简化路径) ---
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Val/images_png', seg_map_path='Val/masks_png'),
        metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural', 'playground')),
        pipeline=val_pipeline))
val_evaluator = dict(type='IoUMetric')

# --- 6. 训练策略 (使用较低的学习率进行微调) ---
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)  # 迭代次数减半
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01),  # 使用较低的学习率
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000, by_epoch=False)
]

# --- 7. 模型定义 ---
# 关键：我们不再需要init_cfg，因为将从完整的模型加载
# 只需要确保解码头的类别数被正确设置为8
model = dict(decode_head=dict(num_classes=num_classes))

# --- 8. 运行时设置 ---
default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'))
log_level = 'INFO'
# === 关键修改：从您自己的、清理过的权重文件开始加载 ===
load_from = '/kaggle/working/best_mIoU_iter_6000_cleaned.pth'
resume = False