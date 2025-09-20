#!/usr/bin/env python3
"""
测试张量维度问题的调试脚本
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from mmseg_custom.models.seg_data_preprocessor import SegDataPreProcessor

def test_empty_batch():
    """测试空批次的处理"""
    print("=== 测试空批次处理 ===")
    
    preprocessor = SegDataPreProcessor()
    
    # 测试空输入列表
    data = {
        'inputs': [],
        'data_samples': []
    }
    
    result = preprocessor.forward(data)
    print(f"空输入结果: {result['inputs'].shape}")
    
    # 测试单个空张量
    data2 = {
        'inputs': torch.empty(0),
        'data_samples': []
    }
    
    try:
        result2 = preprocessor.forward(data2)
        print(f"空张量结果: {result2['inputs'].shape}")
    except Exception as e:
        print(f"空张量处理错误: {e}")

def test_normal_batch():
    """测试正常批次的处理"""
    print("\n=== 测试正常批次处理 ===")
    
    preprocessor = SegDataPreProcessor()
    
    # 创建模拟图像数据
    img1 = torch.randn(3, 224, 224)
    img2 = torch.randn(3, 224, 224)
    
    data = {
        'inputs': [img1, img2],
        'data_samples': []
    }
    
    result = preprocessor.forward(data)
    print(f"正常批次结果: {result['inputs'].shape}")

if __name__ == "__main__":
    test_empty_batch()
    test_normal_batch()