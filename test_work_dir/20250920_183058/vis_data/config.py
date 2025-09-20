accumulative_counts = 1
auto_scale_lr = dict(base_batch_size=16, enable=True)
batch_size = 2
crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmseg_custom.datasets',
        'mmseg_custom.transforms',
        'mmseg_custom.models',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/workspace/data/mmrs1m/data'
dataset_type = 'MMRS1MDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        rule='greater',
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        backend_args=None,
        draw=True,
        interval=500,
        show=False,
        type='SegVisualizationHook',
        wait_time=0.01))
distillation_config = dict(
    distill_loss_weight=0.5, enable=False, teacher_model=None, temperature=4.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
exp_name = 'dinov3_mmrs1m_t20_gcu_8card'
fp16 = dict(loss_scale=512.0)
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
img_size = (
    512,
    512,
)
instruction_config = dict(
    enable_instruction_tuning=True,
    instruction_templates=[
        'What is the category of this remote sensing image?',
        'Classify this satellite image.',
        'Identify the land cover type in this image.',
        'What type of terrain is shown in this remote sensing data?',
    ],
    response_format='single_word')
load_from = None
local_data_root = '/Users/barryzhang/myDev3/MapSage_V5/data'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=-1,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        arch='large',
        drop_path_rate=0.1,
        final_norm=True,
        img_size=(
            512,
            512,
        ),
        in_channels=3,
        init_cfg=None,
        interpolate_mode='bicubic',
        out_indices=(23, ),
        output_cls_token=False,
        patch_size=16,
        type='DINOv3ViT',
        with_cls_token=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        embed_dims=1024,
        img_size=(
            512,
            512,
        ),
        in_channels=1024,
        in_index=-1,
        loss_decode=dict(
            class_weight=None,
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=7,
        num_conv=2,
        num_upsampe_layer=2,
        type='VisionTransformerUpHead',
        upsampling_method='bilinear'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='CustomEncoderDecoder')
model_ema_config = dict(enable=False, momentum=0.9999)
multimodal_config = dict(
    cross_modal_learning=True,
    modal_specific_augmentation=True,
    modalities=[
        'optical',
        'sar',
        'infrared',
    ],
    modality_weights=[
        0.6,
        0.3,
        0.1,
    ])
num_classes = 7
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            bias=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=40000,
        eta_min=1e-06,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root='/workspace/data/mmrs1m/data',
        instruction_format=True,
        modality='optical',
        pipeline=[
            dict(type='CustomLoadImageFromFile'),
            dict(type='CustomLoadAnnotations'),
            dict(img_scale=(
                512,
                512,
            ), keep_ratio=True, type='CustomResize'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='CustomNormalize'),
            dict(type='CustomDefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_semantic_seg',
            ], type='CustomCollect'),
        ],
        task_type='classification',
        type='MMRS1MDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='CustomLoadAnnotations'),
    dict(img_scale=(
        512,
        512,
    ), keep_ratio=True, type='CustomResize'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='CustomNormalize'),
    dict(type='CustomDefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_semantic_seg',
    ], type='CustomCollect'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root='/workspace/data/mmrs1m/data',
        instruction_format=True,
        modality='optical',
        pipeline=[
            dict(type='CustomLoadImageFromFile'),
            dict(type='CustomLoadAnnotations'),
            dict(img_scale=(
                512,
                512,
            ), keep_ratio=True, type='CustomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='CustomRandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='CustomNormalize'),
            dict(
                pad_val=0,
                seg_pad_val=255,
                size=(
                    512,
                    512,
                ),
                type='CustomPad'),
            dict(type='CustomDefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_semantic_seg',
            ], type='CustomCollect'),
        ],
        task_type='classification',
        type='MMRS1MDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='CustomLoadAnnotations'),
    dict(img_scale=(
        512,
        512,
    ), keep_ratio=True, type='CustomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='CustomRandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='CustomNormalize'),
    dict(pad_val=0, seg_pad_val=255, size=(
        512,
        512,
    ), type='CustomPad'),
    dict(type='CustomDefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_semantic_seg',
    ], type='CustomCollect'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root='/workspace/data/mmrs1m/data',
        instruction_format=True,
        modality='optical',
        pipeline=[
            dict(type='CustomLoadImageFromFile'),
            dict(type='CustomLoadAnnotations'),
            dict(img_scale=(
                512,
                512,
            ), keep_ratio=True, type='CustomResize'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='CustomNormalize'),
            dict(type='CustomDefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_semantic_seg',
            ], type='CustomCollect'),
        ],
        task_type='classification',
        type='MMRS1MDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
val_pipeline = [
    dict(type='CustomLoadImageFromFile'),
    dict(type='CustomLoadAnnotations'),
    dict(img_scale=(
        512,
        512,
    ), keep_ratio=True, type='CustomResize'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='CustomNormalize'),
    dict(type='CustomDefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_semantic_seg',
    ], type='CustomCollect'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        save_dir='./work_dirs/dinov3_mmrs1m_t20_gcu_8card/tf_logs',
        type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            save_dir='./work_dirs/dinov3_mmrs1m_t20_gcu_8card/tf_logs',
            type='TensorboardVisBackend'),
    ])
work_dir = './test_work_dir'
