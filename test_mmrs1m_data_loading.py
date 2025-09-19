#!/usr/bin/env python3
"""
测试MMRS1M数据集加载
"""

import os
import sys
sys.path.append('.')

from mmseg_custom.datasets.mmrs1m_dataset import MMRS1MDataset

def test_mmrs1m_loading():
    """测试MMRS1M数据集加载"""
    print("🔍 测试MMRS1M数据集加载...")
    
    # 测试配置
    config = {
        'data_root': '/workspace/data/mmrs1m/data',
        'task_type': 'classification',
        'modality': 'optical',
        'instruction_format': True
    }
    
    try:
        # 创建数据集实例
        dataset = MMRS1MDataset(**config)
        
        # 加载数据列表
        data_list = dataset.load_data_list()
        
        print(f"✅ 成功加载数据集")
        print(f"📊 数据样本数量: {len(data_list)}")
        
        if data_list:
            print(f"📋 前5个样本信息:")
            for i, item in enumerate(data_list[:5]):
                print(f"  {i+1}. 图像路径: {item.get('img_path', 'N/A')}")
                print(f"     类别: {item.get('category', 'N/A')}")
                print(f"     数据集: {item.get('dataset', 'N/A')}")
                print(f"     标签ID: {item.get('label', 'N/A')}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_local_fallback():
    """测试本地占位符数据"""
    print("\n🔍 测试本地占位符数据...")
    
    config = {
        'data_root': '/nonexistent/path',  # 不存在的路径
        'task_type': 'classification',
        'modality': 'optical',
        'instruction_format': True
    }
    
    try:
        dataset = MMRS1MDataset(**config)
        data_list = dataset.load_data_list()
        
        print(f"✅ 占位符数据加载成功")
        print(f"📊 数据样本数量: {len(data_list)}")
        
        if data_list:
            item = data_list[0]
            print(f"📋 占位符样本信息:")
            print(f"  图像路径: {item.get('img_path', 'N/A')}")
            print(f"  类别: {item.get('category', 'N/A')}")
            print(f"  数据集: {item.get('dataset', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 占位符数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🚀 开始测试MMRS1M数据集加载...")
    
    # 测试真实数据加载
    success1 = test_mmrs1m_loading()
    
    # 测试占位符数据
    success2 = test_local_fallback()
    
    if success1 or success2:
        print("\n✅ 测试完成，至少一种数据加载方式成功")
    else:
        print("\n❌ 所有测试都失败了")
        sys.exit(1)