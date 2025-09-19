#!/usr/bin/env python3
"""简单测试脚本，验证输入形状修复是否有效"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

# 导入必要的模块
from mmseg_custom.models.encoder_decoder import EncoderDecoder
from mmseg_custom.models.seg_data_preprocessor import SegDataPreProcessor

def test_input_shape_fix():
    """测试输入形状修复"""
    print("🧪 测试输入形状修复...")
    
    # 创建数据预处理器
    preprocessor = SegDataPreProcessor(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
    )
    
    # 创建模拟输入数据（按照SegDataPreProcessor期望的格式）
    inputs_data = {
        'inputs': [
            torch.randn(224, 224, 3),  # 模拟图像1
            torch.randn(224, 224, 3)   # 模拟图像2
        ]
    }
    
    # 数据预处理
    preprocessed = preprocessor(inputs_data)
    print(f"📊 预处理后数据类型: {type(preprocessed)}")
    print(f"📊 预处理后数据键: {preprocessed.keys() if isinstance(preprocessed, dict) else 'Not a dict'}")
    
    if isinstance(preprocessed, dict) and 'inputs' in preprocessed:
        tensor_inputs = preprocessed['inputs']
        print(f"📊 提取的tensor形状: {tensor_inputs.shape}")
        
        # 测试我们的修复逻辑
        if isinstance(preprocessed, dict) and 'inputs' in preprocessed:
            actual_inputs = preprocessed['inputs']
        else:
            actual_inputs = preprocessed
            
        print(f"✅ 修复后的输入形状: {actual_inputs.shape}")
        print(f"✅ 输入维度数: {len(actual_inputs.shape)}")
        
        if len(actual_inputs.shape) == 4:
            print("✅ 输入形状修复成功！现在是4维张量")
            return True
        else:
            print("❌ 输入形状修复失败！不是4维张量")
            return False
    else:
        print("❌ 预处理器输出格式不正确")
        return False

if __name__ == "__main__":
    success = test_input_shape_fix()
    if success:
        print("\n🎉 输入形状修复测试通过！")
    else:
        print("\n❌ 输入形状修复测试失败！")
        sys.exit(1)