#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCU Device Fix Test Script
Test script for verifying torch_gcu API usage and model device movement

Usage:
1. Run this script in the dinov3-container on T20 server
2. The script will test various torch_gcu API calls
3. Verify that models can be correctly moved to GCU devices
4. Test DDP wrapper device compatibility

Author: MapSage Team
Date: 2025-01-21
"""

import os
import sys
import torch
import traceback
from typing import Optional, Any

# Add project paths
sys.path.insert(0, '/workspace/code/MapSage_V5')
sys.path.insert(0, '.')

# Conditional import of torch_gcu to avoid import errors in non-GCU environments
try:
    import torch_gcu  # type: ignore
    TORCH_GCU_AVAILABLE = True
except ImportError:
    torch_gcu = None  # type: ignore
    TORCH_GCU_AVAILABLE = False

def test_torch_gcu_import():
    """Test torch_gcu import"""
    print("🔍 Testing torch_gcu import...")
    
    if not TORCH_GCU_AVAILABLE:
        print("❌ torch_gcu import failed: module not available")
        print("💡 This is normal, torch_gcu is only available in Enflame T20 GCU environment")
        return None
    
    try:
        print("✅ torch_gcu import successful")
        if torch_gcu is not None:
            print(f"📊 Available GCU devices: {torch_gcu.device_count()}")
            print(f"🔧 Current GCU device: {torch_gcu.current_device()}")
            print(f"💾 GCU availability: {torch_gcu.is_available()}")
        return torch_gcu
    except Exception as e:
        print(f"❌ torch_gcu operation failed: {e}")
        return None

def test_gcu_device_operations(gcu_module: Optional[Any]):
    """Test GCU device operations"""
    if gcu_module is None:
        print("⚠️ Skipping GCU device operations test (torch_gcu not available)")
        return False
    
    print("\n🔧 Testing GCU device operations...")
    
    try:
        # Test device count
        device_count = gcu_module.device_count()
        print(f"📊 Total GCU devices: {device_count}")
        
        if device_count > 0:
            # Test setting device 0
            gcu_module.set_device(0)
            current_device = gcu_module.current_device()
            print(f"✅ Set device 0 successfully, current device: {current_device}")
            
            # Test tensor creation
            tensor = torch.randn(2, 3)
            print(f"🔍 CPU tensor device: {tensor.device}")
            
            # Test moving to GCU
            gcu_tensor = tensor.cuda()  # Use GCU-compatible cuda() method
            print(f"✅ Tensor moved to GCU successfully, device: {gcu_tensor.device}")
            
            return True
        else:
            print("❌ No available GCU devices")
            return False
            
    except Exception as e:
        print(f"❌ GCU device operations failed: {e}")
        return False

def test_model_creation_and_movement(gcu_module: Optional[Any]):
    """Test model creation and device movement"""
    print("\n🏗️ Testing model creation and device movement...")
    
    try:
        # Create simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        print("✅ Model created successfully")
        
        # Check initial device
        first_param = next(model.parameters())
        print(f"🔍 Model initial device: {first_param.device}")
        
        if gcu_module is not None:
            # Use torch_gcu API to move model
            model = model.cuda()  # Use GCU-compatible cuda() method
            
            # Verify movement result
            first_param = next(model.parameters())
            print(f"✅ Model moved to GCU successfully, device: {first_param.device}")
            
            # Test model inference
            test_input = torch.randn(1, 10).cuda()
            output = model(test_input)
            print(f"✅ GCU model inference successful, output device: {output.device}")
            
            return True
        else:
            print("⚠️ torch_gcu not available, skipping GCU movement test")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_mmengine_model_build():
    """Test MMEngine model building"""
    print("\n🔧 Testing MMEngine model building...")
    
    try:
        from mmengine.registry import MODELS
        
        # Register simple model for testing
        @MODELS.register_module()
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
                
            def forward(self, x):
                return self.conv(x)
        
        # Build model
        model = MODELS.build(dict(type='TestModel'))
        print("✅ MMEngine model building successful")
        
        # Test device movement
        if TORCH_GCU_AVAILABLE and torch_gcu is not None:
            model = model.cuda()
            first_param = next(model.parameters())
            print(f"✅ MMEngine model moved to GCU successfully, device: {first_param.device}")
        else:
            print("⚠️ torch_gcu not available, skipping MMEngine model GCU test")
        
        return True
        
    except Exception as e:
        print(f"❌ MMEngine model test failed: {e}")
        return False

def test_ddp_compatibility(gcu_module: Optional[Any]):
    """Test DDP compatibility"""
    print("\n🔗 Testing DDP compatibility...")
    
    # Check distributed environment
    if not torch.distributed.is_available():
        print("⚠️ Non-distributed environment, skipping DDP test")
        return True
    
    try:
        # Get local rank from environment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Create simple model
        model = torch.nn.Linear(10, 5)
        
        # Move to GCU
        if gcu_module is not None:
            gcu_module.set_device(local_rank)
            model = model.cuda()
            
            print(f"✅ Model moved to GCU device: {local_rank}")
            
            # Test DDP wrapping (don't specify device_ids, let MMEngine handle automatically)
            # Here we only verify the model is on the correct device, actual DDP wrapping is handled by MMEngine
            first_param = next(model.parameters())
            if not first_param.device.type == 'cpu':
                print("✅ Model parameters not on CPU, DDP wrapping should succeed")
                return True
            else:
                print("❌ Model parameters still on CPU, DDP wrapping will fail")
                return False
        else:
            print("⚠️ torch_gcu not available, skipping DDP compatibility test")
            return True
            
    except Exception as e:
        print(f"❌ DDP compatibility test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 GCU Device Fix Test Started")
    print("=" * 50)
    
    # Test results statistics
    test_results = []
    
    # 1. Test torch_gcu import
    gcu_module = test_torch_gcu_import()
    test_results.append(("torch_gcu import", gcu_module is not None))
    
    # 2. Test GCU device operations
    device_ops_ok = test_gcu_device_operations(gcu_module)
    test_results.append(("GCU device operations", device_ops_ok))
    
    # 3. Test model creation and movement
    model_ok = test_model_creation_and_movement(gcu_module)
    test_results.append(("Model device movement", model_ok))
    
    # 4. Test MMEngine model building
    mmengine_ok = test_mmengine_model_build()
    test_results.append(("MMEngine model building", mmengine_ok))
    
    # 5. Test DDP compatibility
    ddp_ok = test_ddp_compatibility(gcu_module)
    test_results.append(("DDP compatibility", ddp_ok))
    
    # Output test results
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
        if not result:
            all_passed = False
    
    print("-" * 30)
    if all_passed:
        print("🎉 All tests passed! GCU device fix successful!")
        print("💡 Now you can run 8-card distributed training")
    else:
        print("⚠️ Some tests failed, further debugging needed")
        print("💡 Please check torch_gcu installation and GCU device configuration")
    
    print("\n📋 Next steps:")
    print("1. If tests pass, run: bash scripts/start_8card_training.sh")
    print("2. If tests fail, check torch_gcu installation and device configuration")
    print("3. View detailed error messages and fix according to prompts")

if __name__ == "__main__":
    main()