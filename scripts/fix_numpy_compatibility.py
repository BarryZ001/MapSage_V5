#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumPy兼容性修复脚本
解决TensorBoard与新版NumPy的兼容性问题
"""

import numpy as np
import sys

def fix_numpy_compatibility():
    """修复NumPy兼容性问题"""
    print(f"当前NumPy版本: {np.__version__}")
    
    # 检查并修复np.object问题
    if not hasattr(np, 'object'):
        # 为旧版TensorBoard添加np.object别名
        np.object = object
        print("✅ 已添加np.object兼容性别名")
    else:
        print("✅ np.object已存在")
    
    # 测试TensorBoard导入
    try:
        from tensorboard.compat.tensorflow_stub import dtypes
        print("✅ TensorBoard导入测试成功")
        return True
    except Exception as e:
        print(f"❌ TensorBoard导入测试失败: {e}")
        return False

def test_training_imports():
    """测试训练相关的导入"""
    try:
        # 测试MMEngine相关导入
        from mmengine import Config
        from mmengine.runner import Runner
        print("✅ MMEngine导入成功")
        
        # 测试MMSeg相关导入
        import mmseg
        print("✅ MMSeg导入成功")
        
        # 测试自定义模块导入
        from mmseg.visualization import SegLocalVisualizer
        print("✅ SegLocalVisualizer导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练模块导入失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("NumPy兼容性修复脚本")
    print("=" * 50)
    
    # 修复兼容性
    compatibility_ok = fix_numpy_compatibility()
    
    # 测试训练导入
    training_ok = test_training_imports()
    
    print("=" * 50)
    if compatibility_ok and training_ok:
        print("✅ 所有测试通过，可以开始训练")
        sys.exit(0)
    else:
        print("❌ 存在兼容性问题，需要进一步修复")
        sys.exit(1)