#!/usr/bin/env python3
"""
调试空张量问题
诊断为什么数据预处理器返回空张量 torch.Size([0])
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner

# 导入自定义模块
import mmseg_custom.models
import mmseg_custom.datasets

def test_data_preprocessor():
    """测试数据预处理器"""
    print("🔍 测试数据预处理器...")
    
    # 创建测试数据
    test_data = {
        'inputs': [torch.randn(3, 512, 512)],  # 模拟图像数据
        'data_samples': []
    }
    
    print(f"📊 输入数据格式: {type(test_data['inputs'])}")
    print(f"📊 输入数据长度: {len(test_data['inputs'])}")
    print(f"📊 第一个图像形状: {test_data['inputs'][0].shape}")
    
    # 创建数据预处理器
    preprocessor_cfg = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)
    )
    
    preprocessor = MODELS.build(preprocessor_cfg)
    print(f"✅ 数据预处理器创建成功: {type(preprocessor)}")
    
    # 测试预处理
    try:
        result = preprocessor(test_data, training=True)
        print(f"📊 预处理结果类型: {type(result)}")
        print(f"📊 预处理结果键: {result.keys()}")
        
        if 'inputs' in result:
            inputs = result['inputs']
            print(f"📊 处理后inputs类型: {type(inputs)}")
            print(f"📊 处理后inputs形状: {inputs.shape}")
            print(f"📊 处理后inputs数据类型: {inputs.dtype}")
            
            if inputs.numel() == 0:
                print("❌ 发现空张量问题！")
                return False
            else:
                print("✅ 张量不为空")
                return True
        else:
            print("❌ 结果中没有inputs键")
            return False
            
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_empty_input_handling():
    """测试空输入处理"""
    print("\n🔍 测试空输入处理...")
    
    # 测试空列表输入
    empty_data = {
        'inputs': [],
        'data_samples': []
    }
    
    preprocessor_cfg = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)
    )
    
    preprocessor = MODELS.build(preprocessor_cfg)
    
    try:
        result = preprocessor(empty_data, training=True)
        print(f"📊 空输入处理结果: {result}")
        
        if 'inputs' in result:
            inputs = result['inputs']
            print(f"📊 空输入处理后形状: {inputs.shape}")
            
            if inputs.shape == torch.Size([0]):
                print("⚠️ 确认：空输入导致空张量")
                return True
                
    except Exception as e:
        print(f"❌ 空输入处理失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n🔍 测试配置加载...")
    
    config_file = 'configs/train_dinov3_mmrs1m_t20_gcu.py'
    
    try:
        cfg = Config.fromfile(config_file)
        print(f"✅ 配置文件加载成功")
        
        # 检查数据预处理器配置
        if hasattr(cfg.model, 'data_preprocessor'):
            print(f"📊 数据预处理器配置: {cfg.model.data_preprocessor}")
        
        # 检查数据加载器配置
        if hasattr(cfg, 'train_dataloader'):
            print(f"📊 训练数据加载器批次大小: {cfg.train_dataloader.batch_size}")
            print(f"📊 数据集类型: {cfg.train_dataloader.dataset.type}")
            
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始调试空张量问题...")
    
    # 测试1: 数据预处理器
    success1 = test_data_preprocessor()
    
    # 测试2: 空输入处理
    success2 = test_empty_input_handling()
    
    # 测试3: 配置加载
    success3 = test_config_loading()
    
    print(f"\n📊 测试结果总结:")
    print(f"   数据预处理器测试: {'✅' if success1 else '❌'}")
    print(f"   空输入处理测试: {'✅' if success2 else '❌'}")
    print(f"   配置加载测试: {'✅' if success3 else '❌'}")
    
    if not success1:
        print("\n🔧 建议修复方案:")
        print("1. 检查数据加载器是否正确返回数据")
        print("2. 检查数据预处理器的输入处理逻辑")
        print("3. 确保数据集路径和文件存在")

if __name__ == '__main__':
    main()