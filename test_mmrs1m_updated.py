#!/usr/bin/env python3
"""
测试更新后的MMRS1M数据集加载功能
"""

import os
import sys
sys.path.append('/Users/barryzhang/myDev3/MapSage_V5')

from mmseg_custom.datasets.mmrs1m_dataset import MMRS1MDataset

def test_mmrs1m_dataset():
    """测试MMRS1M数据集的各种任务类型"""
    
    # 测试数据根目录（使用服务端路径进行模拟测试）
    data_root = '/workspace/data/mmrs1m/data'
    
    print("=== 测试MMRS1M数据集加载功能 ===\n")
    
    # 测试不同任务类型
    task_types = ['classification', 'detection', 'caption', 'vqa', 'rsvg']
    
    for task_type in task_types:
        print(f"--- 测试 {task_type.upper()} 任务 ---")
        
        try:
            # 创建数据集实例
            dataset = MMRS1MDataset(
                data_root=data_root,
                task_type=task_type,
                modality='optical',
                instruction_format=True
            )
            
            print(f"✓ {task_type} 数据集实例创建成功")
            print(f"  数据集长度: {len(dataset)}")
            
            if len(dataset) > 0:
                # 测试获取第一个数据项
                data_item = dataset[0]
                print(f"  第一个数据项键: {list(data_item.keys())}")
                
                # 显示任务特定信息
                if task_type == 'classification':
                    print(f"  类别: {data_item.get('category', 'N/A')}")
                    print(f"  数据集: {data_item.get('dataset', 'N/A')}")
                elif task_type == 'detection':
                    print(f"  数据集: {data_item.get('dataset', 'N/A')}")
                    print(f"  标注文件: {data_item.get('ann_file', 'N/A')}")
                elif task_type == 'caption':
                    print(f"  描述: {data_item.get('caption', 'N/A')[:50]}...")
                elif task_type == 'vqa':
                    print(f"  问题: {data_item.get('question', 'N/A')[:50]}...")
                    print(f"  答案: {data_item.get('answer', 'N/A')[:50]}...")
                elif task_type == 'rsvg':
                    print(f"  表达式: {data_item.get('expression', 'N/A')[:50]}...")
                    print(f"  边界框: {data_item.get('bbox', 'N/A')}")
                
                if 'instruction' in data_item:
                    print(f"  指令: {data_item['instruction'][:50]}...")
                    
            else:
                print("  ⚠️ 数据集为空，可能是因为数据目录不存在")
                
        except Exception as e:
            print(f"✗ {task_type} 数据集测试失败: {e}")
        
        print()
    
    print("=== 测试完成 ===")

def test_dataset_with_local_data():
    """使用本地测试数据测试数据集"""
    print("\n=== 使用本地测试数据 ===")
    
    # 使用本地数据路径
    local_data_root = '/Users/barryzhang/myDev3/MapSage_V5/data'
    
    try:
        dataset = MMRS1MDataset(
            data_root=local_data_root,
            task_type='classification',
            modality='optical',
            instruction_format=True
        )
        
        print(f"✓ 本地数据集创建成功")
        print(f"  数据集长度: {len(dataset)}")
        
        if len(dataset) > 0:
            data_item = dataset[0]
            print(f"  数据项键: {list(data_item.keys())}")
            print(f"  图像路径: {data_item.get('img_path', 'N/A')}")
            print(f"  类别: {data_item.get('category', 'N/A')}")
            
    except Exception as e:
        print(f"✗ 本地数据集测试失败: {e}")

def test_class_mappings():
    """测试类别映射功能"""
    print("\n=== 测试类别映射 ===")
    
    dataset = MMRS1MDataset(
        data_root=None,  # 使用None触发模拟数据
        task_type='classification',
        modality='optical'
    )
    
    # 测试不同数据集的类别映射
    test_cases = [
        ('EuroSAT_split', 'Forest'),
        ('NWPU-RESISC45_split', 'airplane'),
        ('UCMerced_split', 'agricultural'),
        ('unknown_dataset', 'building')
    ]
    
    for dataset_name, category in test_cases:
        class_id = dataset._get_class_id_from_dataset(dataset_name, category)
        print(f"  {dataset_name} - {category}: ID = {class_id}")

if __name__ == "__main__":
    test_mmrs1m_dataset()
    test_dataset_with_local_data()
    test_class_mappings()