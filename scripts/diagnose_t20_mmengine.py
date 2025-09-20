#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T20环境MMEngine诊断脚本
检查MMEngine及相关依赖包的安装状态
"""

import sys
import subprocess
import importlib.util
import os

def print_header(title):
    """打印标题"""
    print(f"\n{'='*50}")
    print(f"=== {title} ===")
    print('='*50)

def check_python_environment():
    """检查Python环境"""
    print_header("Python环境检查")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查pip
    try:
        import pip
        print(f"pip版本: {pip.__version__}")
    except ImportError:
        print("❌ pip未安装")
    
    # 检查Python路径
    result = subprocess.run(['which', 'python3'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Python路径(which): {result.stdout.strip()}")
    else:
        print("❌ python3命令未找到")

def check_module_installation(module_name, package_name=None):
    """检查模块安装状态"""
    if package_name is None:
        package_name = module_name
    
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', '未知')
            location = spec.origin if spec.origin else '未知'
            print(f"✅ {module_name}: Version: {version}")
            print(f"   位置: {location}")
            return True
        else:
            print(f"❌ {module_name}: 模块未找到")
            return False
    except Exception as e:
        print(f"❌ {module_name}: 导入错误 - {e}")
        return False

def check_pip_packages():
    """检查pip包安装状态"""
    print_header("pip包检查")
    
    packages_to_check = [
        'mmengine',
        'mmcv',
        'mmsegmentation',
        'torch',
        'torchvision',
        'torch_gcu',
        'numpy',
        'opencv-python',
        'pillow',
        'matplotlib',
        'tqdm',
        'tensorboard'
    ]
    
    for package in packages_to_check:
        try:
            result = subprocess.run(['pip3', 'show', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                version_line = next((line for line in lines if line.startswith('Version:')), None)
                location_line = next((line for line in lines if line.startswith('Location:')), None)
                
                version = version_line.split(': ')[1] if version_line else '未知'
                location = location_line.split(': ')[1] if location_line else '未知'
                
                print(f"✅ {package}: Version: {version}")
                print(f"   位置: {location}")
            else:
                print(f"❌ {package}: 未安装")
        except Exception as e:
            print(f"❌ {package}: 检查失败 - {e}")

def check_mmengine_modules():
    """检查MMEngine相关模块"""
    print_header("MMEngine模块检查")
    
    modules_to_check = [
        'mmengine',
        'mmengine.config',
        'mmengine.runner',
        'mmengine.hooks',
        'mmengine.optim',
        'mmcv',
        'mmcv.ops',
        'mmseg',
        'mmseg.apis',
        'mmseg.models'
    ]
    
    for module in modules_to_check:
        check_module_installation(module)

def check_environment_variables():
    """检查环境变量"""
    print_header("环境变量检查")
    
    env_vars = [
        'PYTHONPATH',
        'LD_LIBRARY_PATH',
        'CUDA_VISIBLE_DEVICES',
        'GCU_VISIBLE_DEVICES',
        'MMCV_WITH_OPS',
        'MMCV_WITH_CUDA'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: 未设置")

def suggest_fixes():
    """提供修复建议"""
    print_header("修复建议")
    
    print("如果MMEngine相关模块缺失，请尝试以下步骤：")
    print()
    print("1. 更新pip:")
    print("   pip3 install --upgrade pip")
    print()
    print("2. 安装MMEngine:")
    print("   pip3 install mmengine")
    print()
    print("3. 安装MMCV:")
    print("   pip3 install mmcv>=2.0.0")
    print()
    print("4. 安装MMSegmentation:")
    print("   pip3 install mmsegmentation")
    print()
    print("5. 如果安装失败，尝试使用清华源:")
    print("   pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple mmengine mmcv mmsegmentation")
    print()
    print("6. 验证安装:")
    print("   python3 -c \"import mmengine; print('MMEngine version:', mmengine.__version__)\"")
    print("   python3 -c \"import mmcv; print('MMCV version:', mmcv.__version__)\"")
    print("   python3 -c \"import mmseg; print('MMSeg version:', mmseg.__version__)\"")

def main():
    """主函数"""
    print("🔍 T20环境MMEngine诊断开始...")
    
    check_python_environment()
    check_pip_packages()
    check_mmengine_modules()
    check_environment_variables()
    suggest_fixes()
    
    print_header("诊断完成")
    print("请根据上述诊断结果和修复建议进行相应的修复操作。")

if __name__ == "__main__":
    main()