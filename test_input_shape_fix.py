#!/usr/bin/env python3
"""
测试输入形状修复
验证encoder_decoder模型是否能正确处理data_preprocessor的输出
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

# 导入自定义模块
from mmseg_custom.models.encoder_decoder import EncoderDecoder
from mmseg_custom.models.seg_data_preprocessor import SegDataPreProcessor
from mmseg_custom.models.dinov3_backbone import DINOv3ViT
from mmseg_custom.models.fcn_head import FCNHead

def test_input_shape_handling():
    """测试输入形状处理"""
    print("🧪 测试输入形状处理...")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    # 模拟原始输入数据（来自dataloader）
    test_data = {
        'inputs': [
            torch.randn(height, width, channels),  # HWC格式
            torch.randn(height, width, channels)   # HWC格式
        ],
        'data_samples': []
    }
    
    print(f"📊 原始输入数据格式: {[img.shape for img in test_data['inputs']]}")
    
    # 创建data_preprocessor
    data_preprocessor = SegDataPreProcessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    )
    
    # 处理数据
    processed_data = data_preprocessor.forward(test_data, training=True)
    print(f"📊 预处理后数据格式: {processed_data['inputs'].shape}")
    
    # 创建backbone
    backbone_config = {
        'type': 'DINOv3ViT',
        'arch': 'small',
        'img_size': 224,
        'patch_size': 16,
        'out_indices': [11],
        'with_cls_token': False
    }
    
    backbone = DINOv3ViT(**{k: v for k, v in backbone_config.items() if k != 'type'})
    
    # 创建decode_head
    decode_head_config = {
        'type': 'FCNHead',
        'in_channels': 384,  # DINOv3-small的输出通道数
        'in_index': 0,
        'channels': 256,
        'num_convs': 1,
        'concat_input': False,
        'dropout_ratio': 0.1,
        'num_classes': 7,
        'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
        'align_corners': False
    }
    
    decode_head = FCNHead(**{k: v for k, v in decode_head_config.items() if k != 'type'})
    
    # 创建encoder_decoder模型
    model = EncoderDecoder(
        backbone={'type': 'DINOv3ViT', **{k: v for k, v in backbone_config.items() if k != 'type'}},
        decode_head={'type': 'FCNHead', **{k: v for k, v in decode_head_config.items() if k != 'type'}},
        data_preprocessor={'type': 'SegDataPreProcessor', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}
    )
    
    # 测试1: 直接传入tensor（旧格式）
    print("\n🧪 测试1: 直接传入tensor")
    try:
        tensor_input = processed_data['inputs']
        result1 = model.forward(tensor_input, mode='tensor')
        print(f"✅ 成功处理tensor输入，输出形状: {result1['seg_logits'].shape}")
    except Exception as e:
        print(f"❌ tensor输入失败: {e}")
    
    # 测试2: 传入dict（新格式，来自data_preprocessor）
    print("\n🧪 测试2: 传入dict（data_preprocessor输出）")
    try:
        dict_input = processed_data  # {'inputs': tensor, 'data_samples': []}
        result2 = model.forward(dict_input, mode='tensor')
        print(f"✅ 成功处理dict输入，输出形状: {result2['seg_logits'].shape}")
    except Exception as e:
        print(f"❌ dict输入失败: {e}")
    
    # 测试3: loss模式
    print("\n🧪 测试3: loss模式")
    try:
        loss_result = model.forward(processed_data, mode='loss')
        print(f"✅ 成功计算损失: {list(loss_result.keys())}")
    except Exception as e:
        print(f"❌ loss模式失败: {e}")
    
    print("\n✅ 输入形状处理测试完成!")

if __name__ == '__main__':
    test_input_shape_handling()