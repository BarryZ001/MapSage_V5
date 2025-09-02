# filename: final_standalone_config.py
# FINAL VERSION - Includes a 'data' dictionary for compatibility with the inference API.

# --- Model Configuration ---
model = dict(
    type='EncoderDecoder',
    
    # Data Preprocessor (Modern v1.x format)
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
    ),
    
    # Backbone: SegFormer-B2 (Official standard definition)
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
        drop_path_rate=0.1
    ),
    
    # Decode Head
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=False),
        align_corners=False,
    ),
    
    # Training Config (Not used for inference)
    train_cfg=dict(),
    
    # Test/Inference Config
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)


# ======================================================================
# === FIX FOR THE CRASH ===
# Add this 'data' dictionary to satisfy the legacy structure that the
# inference_segmentor function is expecting.
# ======================================================================
test_pipeline = [
    dict(type='LoadImageFromFile'), # Placeholder to make the structure valid
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs')
]
data = dict(
    test=dict(
        # These values are placeholders to prevent the crash.
        type='LoveDADataset',
        data_root='data/LoveDA',
        pipeline=test_pipeline
    )
)