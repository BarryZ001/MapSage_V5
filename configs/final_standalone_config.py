# filename: configs/final_standalone_config.py (Definitive Version with Correct Normalization)

work_dir = './work_dirs/test'

model = dict(
    type='EncoderDecoder',
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
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=False),
        align_corners=False),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

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

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[rural_val_dataset, urban_val_dataset]
    )
)

test_evaluator = dict(type='IoUMetric')

# Data configuration for inference
data = dict(
    test=dict(
        type='LoveDADataset',
        data_root='/data/datasets/loveda_dataset',
        data_prefix=dict(img_path='Val/Rural/images_png', seg_map_path='Val/Rural/masks_png'),
        pipeline=test_pipeline
    )
)