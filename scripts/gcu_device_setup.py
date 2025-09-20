#!/usr/bin/env python3
"""
GCU设备配置脚本
用于在训练开始前正确配置GCU设备
"""

import os
import sys
import torch

def setup_gcu_device():
    """配置GCU设备环境"""
    try:
        # 导入GCU相关模块
        import torch_gcu
        import ptex
        
        print("✓ torch_gcu 和 ptex 模块导入成功")
        
        # 设置GCU设备
        device = ptex.device("xla")
        print(f"✓ GCU设备配置成功: {device}")
        
        # 检查GCU设备数量
        if hasattr(torch_gcu, 'device_count'):
            device_count = torch_gcu.device_count()
            print(f"✓ 可用GCU设备数量: {device_count}")
        
        # 设置环境变量
        os.environ['TORCH_DEVICE'] = 'xla'
        os.environ['XLA_USE_BF16'] = '1'  # 启用BF16精度
        
        print("✓ GCU环境变量设置完成")
        
        return device
        
    except ImportError as e:
        print(f"✗ GCU模块导入失败: {e}")
        print("请确保在GCU环境中运行此脚本")
        return None
    except Exception as e:
        print(f"✗ GCU设备配置失败: {e}")
        return None

def move_model_to_gcu(model, device):
    """将模型移动到GCU设备"""
    try:
        if device is not None:
            model = model.to(device)
            print("✓ 模型已移动到GCU设备")
        return model
    except Exception as e:
        print(f"✗ 模型移动到GCU设备失败: {e}")
        return model

def move_data_to_gcu(data, device):
    """将数据移动到GCU设备"""
    try:
        if device is not None and isinstance(data, torch.Tensor):
            data = data.to(device)
        return data
    except Exception as e:
        print(f"✗ 数据移动到GCU设备失败: {e}")
        return data

if __name__ == "__main__":
    print("=== GCU设备配置测试 ===")
    device = setup_gcu_device()
    
    if device is not None:
        print("✓ GCU设备配置成功，可以开始训练")
        sys.exit(0)
    else:
        print("✗ GCU设备配置失败")
        sys.exit(1)