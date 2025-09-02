# 文件名: fixed_mapsage_config.py
# 适用于 MMSegmentation 1.x API 的、简洁的推理配置文件

# --- 步骤1: 继承官方经过验证的基础配置 ---
# 我们不再手动定义模型细节，而是直接继承官方的SegFormer-B2配置
# _base_ = './configs/_base_/models/segformer_mit-b2.py'

# 由于我们没有官方基础配置文件，我们手动定义一个简化的SegFormer-B2模型
model = dict(
    type='EncoderDecoder',
    
    # 定义数据预处理器 (这是新版格式，取代了旧的img_norm_cfg)
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,  # mmseg默认为BGR，但根据训练情况可能需要调整
        pad_val=0,
        seg_pad_val=255
    ),
    
    # 主干网络配置 - SegFormer-B2
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
    
    # 修改解码头的类别数以匹配LoveDA数据集
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],  # 对应B2的通道数
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,  # LoveDA数据集有7个类别
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            ignore_index=255
        )
    ),
    
    # 训练配置（推理时不需要，但保留以防万一）
    train_cfg=dict(),
    
    # 测试配置 - 关键：滑窗推理设置
    test_cfg=dict(
        mode='slide',           # 滑窗模式
        crop_size=(1024, 1024), # 滑窗大小
        stride=(768, 768)       # 滑窗步长
    )
)

# --- 步骤3: 定义推理配置 ---
# 这是独立的顶层字段，用于指导推理行为
test_cfg = dict(
    mode='slide',           # 滑窗模式
    crop_size=(1024, 1024), # 滑窗大小
    stride=(768, 768)       # 滑窗步长
)

# --- 步骤4: (可选) 定义测试时的数据处理流程 ---
# 这个pipeline仅在通过 test.py 脚本评估时使用，
# 对于我们的Streamlit应用 (调用inference_model)，它会被模型内部的
# data_preprocessor处理，所以这个字段实际上可以省略。
# 但为了完整性，我们保留一个最简版本。
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),  # 如果需要加载真值图
    dict(type='PackSegInputs')
]

# 类别名称（用于可视化）
classes = [
    'background',    # 背景
    'building',      # 建筑
    'road',          # 道路
    'water',         # 水体
    'barren',        # 贫瘠土地
    'forest',        # 森林
    'agricultural'   # 农业
]

# 调色板（RGB格式）
palette = [
    [255, 255, 255],  # 背景 - 白色
    [255, 0, 0],      # 建筑 - 红色
    [255, 255, 0],    # 道路 - 黄色
    [0, 0, 255],      # 水体 - 蓝色
    [159, 129, 183],  # 贫瘠土地 - 紫色
    [0, 255, 0],      # 森林 - 绿色
    [255, 195, 128]   # 农业 - 橙色
]

# --- 训练相关的配置全部删除 ---
# train_cfg, optimizer, lr_config 等等在推理时完全不需要