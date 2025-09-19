#!/usr/bin/env python3
"""
测试数据集加载功能
验证MMRS1MDataset是否能正确加载数据
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from mmseg_custom.datasets.mmrs1m_dataset import MMRS1MDataset

def test_dataset_loading():
    """测试数据集加载"""
    print("=" * 60)
    print("Testing MMRS1MDataset Loading")
    print("=" * 60)
    
    # 测试配置
    data_root = '/Users/barryzhang/myDev3/MapSage_V5/data'
    
    try:
        # 创建数据集实例
        print("1. Creating MMRS1MDataset instance...")
        dataset = MMRS1MDataset(
            data_root=data_root,
            task_type='classification',
            modality='optical',
            instruction_format=True
        )
        
        print(f"   ✅ Dataset created successfully")
        print(f"   📊 Dataset length: {len(dataset)}")
        
        # 测试数据加载
        if len(dataset) > 0:
            print("\n2. Testing data loading...")
            
            # 获取第一个数据项
            data_info = dataset[0]
            print(f"   📁 Sample data keys: {list(data_info.keys())}")
            
            if 'img_path' in data_info:
                img_path = data_info['img_path']
                print(f"   🖼️ Image path: {img_path}")
                print(f"   📂 File exists: {os.path.exists(img_path)}")
                
                if os.path.exists(img_path):
                    file_size = os.path.getsize(img_path)
                    print(f"   📏 File size: {file_size} bytes")
                
            if 'label' in data_info:
                print(f"   🏷️ Label: {data_info['label']}")
                
            if 'category' in data_info:
                print(f"   📝 Category: {data_info['category']}")
                
            if 'instruction' in data_info:
                print(f"   💬 Instruction: {data_info['instruction']}")
                
            if 'response' in data_info:
                print(f"   💭 Response: {data_info['response']}")
                
            print("   ✅ Data loading test passed")
        else:
            print("   ❌ Dataset is empty")
            
        # 测试多个数据项
        print("\n3. Testing multiple data items...")
        sample_count = min(3, len(dataset))
        for i in range(sample_count):
            data_info = dataset[i]
            img_path = data_info.get('img_path', 'N/A')
            category = data_info.get('category', 'N/A')
            exists = os.path.exists(img_path) if img_path != 'N/A' else False
            print(f"   Sample {i}: {category} -> {os.path.basename(img_path)} (exists: {exists})")
            
        print("   ✅ Multiple data items test passed")
        
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def test_data_directory_structure():
    """测试数据目录结构"""
    print("\n" + "=" * 60)
    print("Testing Data Directory Structure")
    print("=" * 60)
    
    data_root = '/Users/barryzhang/myDev3/MapSage_V5/data'
    test_data_dir = os.path.join(data_root, 'test_data', 'images')
    
    print(f"📁 Data root: {data_root}")
    print(f"📂 Data root exists: {os.path.exists(data_root)}")
    
    print(f"📁 Test data dir: {test_data_dir}")
    print(f"📂 Test data dir exists: {os.path.exists(test_data_dir)}")
    
    if os.path.exists(test_data_dir):
        print("\n📋 Directory contents:")
        for root, dirs, files in os.walk(test_data_dir):
            level = root.replace(test_data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size} bytes)")

if __name__ == "__main__":
    print("🚀 Starting Dataset Loading Tests")
    
    # 测试数据目录结构
    test_data_directory_structure()
    
    # 测试数据集加载
    success = test_dataset_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)