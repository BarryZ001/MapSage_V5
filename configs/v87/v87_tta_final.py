
# 基本配置
dataset_type = 'LoveDADataset'
data_root = './data/loveda'
num_classes = 7
crop_size = (1024, 1024)
palette = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]



# TTA Pipeline
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [0.75, 1.0, 1.25]
            ],
            [
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
                dict(type='RandomFlip', prob=0.0, direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='UniformMaskFormat', palette=palette)],
            [dict(type='PackSegInputs', meta_keys=('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction'))]
        ])
]

# 数据预处理配置
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)

# 模型配置
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
        # --- 核心改动: 修正此处的拼写错误 ---
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), 
        mlp_ratio=4, 
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ),
    decode_head=dict(
        type='SegformerHead', 
        in_channels=[64, 128, 320, 512], 
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
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768))
)

# 评估时的数据加载器
val_dataloader = dict(
    batch_size=1, 
    num_workers=4, 
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, 
        data_root=data_root,
        data_prefix=dict(img_path='Val', seg_map_path='Val'),
        pipeline=tta_pipeline
    )
)
test_dataloader = val_dataloader

# 评估器与流程配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 运行时配置
default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=True, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='eccl'))  # 修改为eccl支持GCU
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

# 工作目录
work_dir = './work_dirs/v87_tta_results'
