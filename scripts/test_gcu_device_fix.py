#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCU Device Fix Test Script
Test torch_gcu functionality and compatibility for T20 server
"""

import os
import sys
import torch
import traceback

# Add project paths
sys.path.insert(0, '/workspace/code/MapSage_V5')
sys.path.insert(0, '.')

# Try to import torch_gcu
try:
    import torch_gcu  # type: ignore
    TORCH_GCU_AVAILABLE = True
except ImportError:
    torch_gcu = None  # type: ignore
    TORCH_GCU_AVAILABLE = False

def test_torch_gcu_import():
    """Test torch_gcu import and basic functionality"""
    print("Testing torch_gcu import...")
    
    if not TORCH_GCU_AVAILABLE:
        print("ERROR: torch_gcu import failed: module not available")
        print("INFO: This is normal, torch_gcu is only available in Enflame T20 GCU environment")
        return None
    
    try:
        print("SUCCESS: torch_gcu import successful")
        if torch_gcu is not None:
            device_count = torch_gcu.device_count()
            current_device = torch_gcu.current_device()
            is_available = torch_gcu.is_available()
            print("INFO: Available GCU devices: " + str(device_count))
            print("INFO: Current GCU device: " + str(current_device))
            print("INFO: GCU availability: " + str(is_available))
        return torch_gcu
    except Exception as e:
        print("ERROR: torch_gcu operation failed: " + str(e))
        return None

def test_gcu_device_operations(gcu_module):
    """Test GCU device operations"""
    if gcu_module is None:
        print("WARNING: Skipping GCU device operations test (torch_gcu not available)")
        return False
    
    print("\nTesting GCU device operations...")
    
    try:
        # Test device count
        device_count = gcu_module.device_count()
        print("INFO: Total GCU devices: " + str(device_count))
        
        if device_count > 0:
            # Test device setting
            gcu_module.set_device(0)
            current_device = gcu_module.current_device()
            print("SUCCESS: Set device 0 successfully, current device: " + str(current_device))
            
            # Test tensor operations
            tensor = torch.randn(2, 3)
            print("INFO: CPU tensor device: " + str(tensor.device))
            
            # Move tensor to GCU (using official GCU device interface)
            gcu_tensor = tensor.to('gcu:0')
            print("SUCCESS: Tensor moved to GCU successfully, device: " + str(gcu_tensor.device))
            
            return True
        else:
            print("ERROR: No available GCU devices")
            return False
            
    except Exception as e:
        print("ERROR: GCU device operations failed: " + str(e))
        return False

def test_model_creation_and_movement(gcu_module):
    """Test model creation and device movement"""
    print("\nTesting model creation and device movement...")
    
    try:
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        print("SUCCESS: Model created successfully")
        
        # Check initial device
        first_param = next(model.parameters())
        print("INFO: Model initial device: " + str(first_param.device))
        
        if gcu_module is not None:
            # Move model to GCU
            model = model.to('gcu:0')
            
            # Check device after movement
            first_param = next(model.parameters())
            print("SUCCESS: Model moved to GCU successfully, device: " + str(first_param.device))
            
            # Test inference
            test_input = torch.randn(1, 10).to('gcu:0')
            output = model(test_input)
            print("SUCCESS: GCU model inference successful, output device: " + str(output.device))
        else:
            print("WARNING: torch_gcu not available, skipping GCU movement test")
        
        return True
        
    except Exception as e:
        print("ERROR: Model test failed: " + str(e))
        return False

def test_mmengine_model_build():
    """Test MMEngine model building"""
    print("\nTesting MMEngine model building...")
    
    try:
        from mmengine.registry import MODELS
        
        # Simple model config
        model_cfg = {
            'type': 'torch.nn.Linear',
            'in_features': 10,
            'out_features': 5
        }
        
        # Register if not exists
        if 'torch.nn.Linear' not in MODELS:
            MODELS.register_module(name='torch.nn.Linear', module=torch.nn.Linear)
        
        # Build model
        model = MODELS.build(model_cfg)
        print("SUCCESS: MMEngine model building successful")
        
        if TORCH_GCU_AVAILABLE and torch_gcu is not None:
            # Move to GCU
            model = model.to('gcu:0')
            first_param = next(model.parameters())
            print("SUCCESS: MMEngine model moved to GCU successfully, device: " + str(first_param.device))
        else:
            print("WARNING: torch_gcu not available, skipping MMEngine model GCU test")
        
        return True
        
    except Exception as e:
        print("ERROR: MMEngine model test failed: " + str(e))
        return False

def test_ddp_compatibility(gcu_module):
    """Test DDP wrapper compatibility"""
    print("\nTesting DDP compatibility...")
    
    try:
        # Check if we're in a distributed environment
        if not hasattr(torch, 'distributed') or not torch.distributed.is_available():
            print("WARNING: Non-distributed environment, skipping DDP test")
            return True
        
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        
        if gcu_module is not None:
            # Move model to GCU device
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = 'gcu:' + str(local_rank)
            model = model.to(device)
            
            print("SUCCESS: Model moved to GCU device: " + str(local_rank))
            
            # Check if model parameters are on the correct device
            param_devices = [p.device for p in model.parameters()]
            cpu_params = [d for d in param_devices if d.type == 'cpu']
            
            if len(cpu_params) == 0:
                print("SUCCESS: Model parameters not on CPU, DDP wrapping should succeed")
                return True
            else:
                print("ERROR: Model parameters still on CPU, DDP wrapping will fail")
                return False
        else:
            print("WARNING: torch_gcu not available, skipping DDP compatibility test")
            return True
            
    except Exception as e:
        print("ERROR: DDP compatibility test failed: " + str(e))
        return False

def main():
    """Main test function"""
    print("GCU Device Fix Test Started")
    print("=" * 50)
    
    results = {}
    
    # Test 1: torch_gcu import
    print("\n1. Testing torch_gcu import...")
    gcu_module = test_torch_gcu_import()
    results['torch_gcu_import'] = gcu_module is not None or not TORCH_GCU_AVAILABLE
    
    # Test 2: GCU device operations
    print("\n2. Testing GCU device operations...")
    results['gcu_operations'] = test_gcu_device_operations(gcu_module)
    
    # Test 3: Model creation and movement
    print("\n3. Testing model creation and movement...")
    results['model_operations'] = test_model_creation_and_movement(gcu_module)
    
    # Test 4: MMEngine model building
    print("\n4. Testing MMEngine model building...")
    results['mmengine_model'] = test_mmengine_model_build()
    
    # Test 5: DDP compatibility
    print("\n5. Testing DDP compatibility...")
    results['ddp_compatibility'] = test_ddp_compatibility(gcu_module)
    
    # Print results
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(test_name.ljust(20) + " : " + status)
    
    # Overall result
    all_passed = all(results.values())
    
    print("\n" + "=" * 50)
    if all_passed:
        print("SUCCESS: All tests passed! GCU environment is ready.")
        print("INFO: Now you can run 8-card distributed training")
    else:
        print("WARNING: Some tests failed, further debugging needed")
        print("INFO: Please check torch_gcu installation and GCU device configuration")
    
    return all_passed

if __name__ == "__main__":
    main()