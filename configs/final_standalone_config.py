# filename: final_standalone_config.py (V4 - Final)

# --- Model Configuration ---
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
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
    # This test_cfg is for the model's internal inference logic
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# --- Dataloader and Pipeline Configuration ---
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LoveDADataset',
        data_root='data/LoveDA',
        data_prefix=dict(img_path='Val/images_png', seg_map_path='Val/annotations_png'),
        pipeline=test_pipeline
    )
)

# === FIX FOR THE ERROR ===
# The Runner requires these three components to be defined for a test run.
# We were missing the top-level `test_cfg`.
# =========================
test_cfg = dict()  # An empty dictionary is sufficient to satisfy the Runner
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mAcc', 'aAcc'])