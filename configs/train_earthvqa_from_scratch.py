# configs/train_earthvqa_from_scratch.py

_base_ = './final_standalone_config.py'

num_classes = 8
crop_size = (512, 512)
# 注意：此处的data_root是一个占位符，实际路径将在Colab中硬编码或动态设置
data_root = '/content/datasets/EarthVQA/EarthVQA'  # Colab本地运行时路径
dataset_type = 'BaseSegDataset'

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
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Train/images_png', seg_map_path='Train/masks_png'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural', 'playground')),
        pipeline=train_pipeline))

# 定义验证数据管道和加载器
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
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(img_path='Val/Rural/images_png', seg_map_path='Val/Rural/masks_png'),
                img_suffix='.png',
                seg_map_suffix='.png',
                metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural', 'playground')),
                pipeline=val_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(img_path='Val/Urban/images_png', seg_map_path='Val/Urban/masks_png'),
                img_suffix='.png',
                seg_map_suffix='.png',
                metainfo=dict(classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural', 'playground')),
                pipeline=val_pipeline)
        ]))

val_evaluator = dict(type='IoUMetric')

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=500, end=40000, by_epoch=False)
]

# 我们将从清理过的best_mIoU_iter_6000.pth开始微调
# 但路径将在Colab中指定，这里仅定义模型结构
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(512, 512),
        mean=[73.53223947628777, 80.01710095339912, 74.59297778068898],
        std=[41.511366098369635, 35.66528876209687, 33.75830885257866],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    decode_head=dict(num_classes=num_classes))

default_scope = 'mmseg'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'))
log_level = 'INFO'
# 关键：我们将通过脚本的 Cell 动态指定 load_from 路径
load_from = None
resume = False