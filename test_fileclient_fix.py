#!/usr/bin/env python3
"""测试FileClient修复是否正常工作"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_fileclient_import():
    """测试FileClient导入是否正常"""
    print("Testing FileClient import compatibility...")
    
    try:
        from mmseg_custom.transforms.standard_transforms import LoadImageFromFile, LoadAnnotations
        print("✅ Successfully imported transforms with FileClient")
        
        # 测试实例化
        load_img = LoadImageFromFile()
        load_ann = LoadAnnotations()
        print("✅ Successfully instantiated transforms")
        
        # 检查FileClient是否正确初始化
        print(f"LoadImageFromFile.file_client: {load_img.file_client}")
        print(f"LoadAnnotations.file_client: {load_ann.file_client}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing or using transforms: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mmcv_compatibility():
    """测试不同mmcv版本的兼容性"""
    print("\nTesting MMCV compatibility...")
    
    try:
        import mmcv
        print(f"MMCV version: {mmcv.__version__}")
        
        # 测试不同的FileClient导入路径
        paths_to_test = [
            ("mmcv", "FileClient"),
            ("mmcv.fileio", "FileClient"),
            ("mmengine.fileio", "FileClient")
        ]
        
        for module_path, class_name in paths_to_test:
            try:
                if module_path == "mmcv":
                    from mmcv import FileClient
                elif module_path == "mmcv.fileio":
                    from mmcv.fileio import FileClient
                elif module_path == "mmengine.fileio":
                    from mmengine.fileio import FileClient
                    
                print(f"✅ {module_path}.{class_name} is available")
                
                # 测试实例化
                client = FileClient(backend='disk')
                print(f"✅ Successfully created FileClient instance: {type(client)}")
                break
                
            except ImportError as e:
                print(f"❌ {module_path}.{class_name} not available: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error testing MMCV compatibility: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FileClient Fix")
    print("=" * 60)
    
    success1 = test_fileclient_import()
    success2 = test_mmcv_compatibility()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed! FileClient fix is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)