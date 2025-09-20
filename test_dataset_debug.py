#!/usr/bin/env python3
"""
调试数据集返回的数据格式问题
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from mmengine.config import Config
from mmengine.registry import DATASETS

def test_dataset_format():
    """测试数据集返回的数据格式"""
    print("=== 测试数据集返回的数据格式 ===")
    
    # 导入自定义模块
    import mmseg_custom.models
    import mmseg_custom.datasets
    
    # 加载配置
    config_path = 'configs/train_dinov3_backbone.py'
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    cfg = Config.fromfile(config_path)
    
    # 构建数据集
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"数据集类型: {type(train_dataset)}")
    print(f"数据集长度: {len(train_dataset)}")
    
    if len(train_dataset) > 0:
        # 获取第一个样本
        sample = train_dataset[0]
        print(f"\n样本类型: {type(sample)}")
        print(f"样本键: {sample.keys() if isinstance(sample, dict) else 'not a dict'}")
        
        # 检查每个键的内容
        if isinstance(sample, dict):
            for key, value in sample.items():
                print(f"  {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    形状: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"    长度: {len(value)}")
                    if len(value) > 0:
                        print(f"    第一个元素类型: {type(value[0])}")
                elif isinstance(value, str):
                    print(f"    值: {value}")
        
        # 测试数据管道处理
        print("\n=== 测试数据管道处理 ===")
        
        # 检查数据集的pipeline
        if hasattr(train_dataset, 'pipeline'):
            print(f"数据管道: {train_dataset.pipeline}")
        
        # 尝试获取多个样本
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            print(f"\n样本 {i}:")
            print(f"  类型: {type(sample)}")
            if isinstance(sample, dict):
                for key, value in sample.items():
                    if key == 'inputs':
                        print(f"  {key}: {type(value)}, 形状: {getattr(value, 'shape', 'N/A')}")
                    elif key == 'data_samples':
                        print(f"  {key}: {type(value)}")
                        if isinstance(value, list) and len(value) > 0:
                            print(f"    第一个数据样本: {type(value[0])}")
                            if hasattr(value[0], 'keys'):
                                print(f"    键: {value[0].keys()}")
                        elif hasattr(value, 'keys') and callable(getattr(value, 'keys')):
                            print(f"    键: {value.keys()}")
                    else:
                        print(f"  {key}: {type(value)}")

if __name__ == "__main__":
    test_dataset_format()