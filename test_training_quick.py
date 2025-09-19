#!/usr/bin/env python3
"""Quick test to verify training setup works without running full training"""

import sys
sys.path.insert(0, '.')

# Import mmseg first to register components
import mmseg
from mmseg.models import *
from mmseg.datasets import *

# Import our custom modules
import mmseg_custom.models
import mmseg_custom.datasets

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS

def test_model_build():
    """Test if model can be built successfully"""
    print("🔧 Testing model build...")
    
    # Load config
    cfg = Config.fromfile('configs/train_dinov3_backbone.py')
    
    try:
        # Build model
        model = MODELS.build(cfg.model)
        print(f"✅ Model built successfully: {type(model)}")
        
        # Test with dummy data
        dummy_data = {
            'inputs': [torch.randn(3, 512, 512) for _ in range(1)],
            'data_samples': []
        }
        
        # Test data preprocessor
        processed = model.data_preprocessor(dummy_data)
        print(f"✅ Data preprocessor works: {processed['inputs'].shape}")
        
        # Test forward pass
        result = model(processed, mode='tensor')
        print(f"✅ Forward pass works: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_build():
    """Test if dataset can be built successfully"""
    print("🔧 Testing dataset build...")
    
    # Load config
    cfg = Config.fromfile('configs/train_dinov3_backbone.py')
    
    try:
        # Build train dataset
        train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
        print(f"✅ Train dataset built successfully: {type(train_dataset)}")
        
        # Build val dataset  
        val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
        print(f"✅ Val dataset built successfully: {type(val_dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🚀 Starting quick training setup test...")
    
    model_ok = test_model_build()
    dataset_ok = test_dataset_build()
    
    if model_ok and dataset_ok:
        print("✅ All tests passed! Training setup is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)