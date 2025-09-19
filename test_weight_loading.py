#!/usr/bin/env python3
"""
测试DINOv3权重加载的不同prefix配置
"""

import torch
import os
from collections import OrderedDict

def test_weight_loading():
    """测试权重加载的不同prefix配置"""
    
    # 权重文件路径
    weight_path = 'checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        return
    
    print(f"🔍 加载权重文件: {weight_path}")
    
    # 加载权重
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    print(f"📊 权重文件类型: {type(checkpoint)}")
    
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        print(f"❌ 未知的权重文件格式")
        return
    
    print(f"🔑 权重键数量: {len(state_dict)}")
    print(f"🔑 前10个键名:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # 测试不同的prefix配置
    test_prefixes = [
        '',           # 空prefix
        'backbone.',  # backbone前缀
        'model.',     # model前缀
        None          # 无prefix配置
    ]
    
    print(f"\n🧪 测试不同prefix配置:")
    
    for prefix in test_prefixes:
        print(f"\n--- 测试 prefix='{prefix}' ---")
        
        if prefix is None:
            print("✅ 无prefix配置，直接使用原始键名")
            continue
            
        # 模拟MMEngine的prefix处理逻辑
        if prefix == '':
            processed_prefix = '.'
        else:
            processed_prefix = prefix
            
        print(f"🔄 处理后的prefix: '{processed_prefix}'")
        
        # 检查是否有匹配的键名
        matching_keys = []
        for key in state_dict.keys():
            if processed_prefix == '.':
                # 空prefix情况，检查是否有以'.'开头的键
                if key.startswith('.'):
                    matching_keys.append(key)
            else:
                # 有prefix情况，检查是否有匹配的键
                if key.startswith(processed_prefix):
                    matching_keys.append(key)
        
        if matching_keys:
            print(f"✅ 找到 {len(matching_keys)} 个匹配的键")
            print(f"   示例: {matching_keys[:3]}")
        else:
            print(f"❌ 没有找到匹配的键")
            
            # 如果是空prefix，尝试直接匹配
            if processed_prefix == '.':
                print("💡 建议: 移除init_cfg配置，让模型使用默认初始化")

if __name__ == "__main__":
    test_weight_loading()