# DINOv3 + MMRS-1M CPU训练配置文件（应急方案）
# 基于原配置修改为CPU训练

# 导入原配置
_base_ = ['./train_dinov3_mmrs1m.py']

# 覆盖GPU相关配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo')  # 使用gloo后端支持CPU
)

# 修改数据加载器配置
train_dataloader = dict(
    batch_size=2,  # 减小批次大小
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MMRS1MDataset',
        data_root='/workspace/data/mmrs1m/data',
        task_type='classification',
        modality='optical',
        instruction_format=True,
        pipeline='{{_base_.train_pipeline}}'
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MMRS1MDataset',
        data_root='/workspace/data/mmrs1m/data',
        task_type='classification',
        modality='optical',
        instruction_format=True,
        pipeline='{{_base_.val_pipeline}}'
    )
)

# 修改训练配置
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1000,  # 减少迭代数用于测试
    val_interval=200
)

# 禁用GPU相关功能
fp16 = None
model_ema_config = dict(
    enable=False
)

# 修改优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-5,  # 降低学习率
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
)

print("⚠️  使用CPU训练配置（应急方案）")
print("🐌 CPU训练速度较慢，建议优先解决GPU环境问题")