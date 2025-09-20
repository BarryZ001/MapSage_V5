#!/usr/bin/env python3
"""
调试训练过程中的数据流问题
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from mmengine.config import Config
from mmengine.runner import Runner

def test_data_loading():
    """测试数据加载过程"""
    print("=== 测试数据加载过程 ===")
    
    # 导入自定义模块
    import mmseg_custom.models
    import mmseg_custom.datasets
    
    # 加载配置
    config_path = 'configs/train_dinov3_backbone.py'
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    cfg = Config.fromfile(config_path)
    
    # 直接构建数据集
    from mmengine.registry import DATASETS
    
    # 构建数据集
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"数据集类型: {type(train_dataset)}")
    print(f"数据集长度: {len(train_dataset)}")
    
    # 测试获取单个样本
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"样本键: {sample.keys()}")
        
        if 'inputs' in sample:
            inputs = sample['inputs']
            print(f"输入类型: {type(inputs)}")
            if hasattr(inputs, 'shape'):
                print(f"输入形状: {inputs.shape}")
        
        if 'data_samples' in sample:
            data_samples = sample['data_samples']
            print(f"数据样本类型: {type(data_samples)}")
    else:
        print("❌ 数据集为空")

def test_model_forward():
    """测试模型前向传播"""
    print("\n=== 测试模型前向传播 ===")
    
    # 导入自定义模块
    import mmseg_custom.models
    import mmseg_custom.datasets
    
    from mmengine.registry import MODELS
    
    # 创建模型配置
    model_cfg = {
        'type': 'EncoderDecoder',
        'data_preprocessor': {
            'type': 'SegDataPreProcessor',
            'mean': [73.53, 80.02, 74.59],
            'std': [41.51, 35.67, 33.76],
            'bgr_to_rgb': True,
            'pad_val': 0,
            'seg_pad_val': 255,
            'size': (512, 512)
        },
        'backbone': {
            'type': 'DINOv3ViT',
            'arch': 'large',
            'img_size': 512,
            'patch_size': 16,
            'out_indices': [23],
            'interpolate_mode': 'bicubic',
            'init_cfg': {'type': 'Normal', 'layer': 'Linear', 'std': 0.01}
        },
        'decode_head': {
            'type': 'FCNHead',
            'in_channels': 1024,
            'in_index': 0,
            'channels': 256,
            'num_convs': 1,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'num_classes': 7,
            'norm_cfg': {'type': 'BN', 'requires_grad': True},
            'align_corners': False,
            'loss_decode': [
                {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}
            ]
        },
        'test_cfg': {'mode': 'whole'}
    }
    
    # 构建模型
    model = MODELS.build(model_cfg)
    print(f"模型类型: {type(model)}")
    
    # 测试不同的输入情况
    test_cases = [
        ("空列表", {'inputs': [], 'data_samples': []}),
        ("空张量", {'inputs': torch.empty(0), 'data_samples': []}),
        ("正常张量", {'inputs': [torch.randn(3, 512, 512)], 'data_samples': []}),
    ]
    
    for case_name, data in test_cases:
        print(f"\n--- 测试 {case_name} ---")
        try:
            # 测试数据预处理器
            preprocessed = model.data_preprocessor(data)
            print(f"预处理后输入形状: {preprocessed['inputs'].shape}")
            
            # 测试模型前向传播
            with torch.no_grad():
                result = model(preprocessed, mode='tensor')
                print(f"模型输出: {type(result)}")
                if isinstance(result, dict) and 'seg_logits' in result:
                    print(f"分割输出形状: {result['seg_logits'].shape}")
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    test_data_loading()
    test_model_forward()