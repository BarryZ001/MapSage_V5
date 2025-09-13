#!/usr/bin/env python3
"""
MapSage V5 Knowledge Distillation Environment Setup
This script sets up the environment for knowledge distillation using MMRazor
"""

import sys
import os
import subprocess
import traceback

def run_command(cmd, description, timeout=600):
    """Run a command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            timeout=timeout,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - Timeout")
        return False

def main():
    print("=== 🚀 MapSage V5 Knowledge Distillation Environment Setup ===\n")
    
    # Step 1: Clean old environment
    print("🧹 Cleaning old environment...")
    cleanup_packages = [
        'torch', 'torchvision', 'torchaudio', 
        'mmcv', 'mmcv-full', 'mmengine', 
        'mmsegmentation', 'mmpretrain', 'mmrazor',
        'transformers'
    ]
    
    subprocess.run([
        sys.executable, '-m', 'pip', 'uninstall', '-y', '-q'
    ] + cleanup_packages, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Step 2: Install PyTorch
    if not run_command([
        sys.executable, '-m', 'pip', 'install', '-q',
        'torch==2.1.2', 'torchvision==0.16.2', 'torchaudio==2.1.2',
        '--index-url', 'https://download.pytorch.org/whl/cu118'
    ], "Installing PyTorch 2.1.2 with CUDA 11.8", timeout=900):
        return False
    
    # Step 3: Install basic dependencies
    if not run_command([
        sys.executable, '-m', 'pip', 'install', '-U', '-q',
        'mmengine==0.10.3', 'openmim', 'ftfy', 'timm'
    ], "Installing basic dependencies", timeout=300):
        return False
    
    # Step 4: Install OpenMMLab packages
    if not run_command([
        sys.executable, '-m', 'mim', 'install',
        'mmcv==2.1.0', 'mmsegmentation==1.2.2', 'mmpretrain==1.2.0'
    ], "Installing MMSegmentation and MMPretrain", timeout=1200):
        return False
    
    # Step 5: Install MMRazor
    if not run_command([
        sys.executable, '-m', 'pip', 'install', '-U',
        'mmrazor'
    ], "Installing MMRazor for knowledge distillation", timeout=600):
        return False
    
    # Step 6: Verify installation
    print("\n" + "="*60)
    print("🔬 Verifying installation...")
    
    try:
        import torch
        import mmcv
        import mmengine
        import mmseg
        import mmpretrain
        import mmrazor
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ MMCV version: {mmcv.__version__}")
        print(f"✅ MMEngine version: {mmengine.__version__}")
        print(f"✅ MMSegmentation version: {mmseg.__version__}")
        print(f"✅ MMPretrain version: {mmpretrain.__version__}")
        print(f"✅ MMRazor version: {mmrazor.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - using CPU")
        
        print("\n🎉 Environment setup completed successfully!")
        print("You can now run knowledge distillation training with MMRazor.")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"❌ Import verification failed: {e}")
        print("="*60)
        return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        traceback.print_exc()
        print("="*60)
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Environment setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Ready for knowledge distillation training!")