#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T20环境验证脚本
验证数据集路径、权重文件和训练环境配置
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 尝试导入torch，如果失败则设置为None
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_success(message):
    """打印成功信息"""
    print(f"✅ {message}")

def print_error(message):
    """打印错误信息"""
    print(f"❌ {message}")

def print_warning(message):
    """打印警告信息"""
    print(f"⚠️  {message}")

def check_python_environment():
    """检查Python环境"""
    print_header("Python环境检查")
    
    # Python版本
    python_version = sys.version
    print(f"Python版本: {python_version}")
    
    # 检查关键包
    packages = [
        'torch',
        'torchvision', 
        'mmcv',
        'mmsegmentation',
        'numpy',
        'opencv-python'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print_success(f"{package} 已安装")
        except ImportError:
            print_error(f"{package} 未安装")
    
    # PyTorch版本和CUDA支持
    if TORCH_AVAILABLE and torch is not None:
        print_success(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            try:
                # 尝试获取CUDA版本信息
                cuda_version = "未知"
                try:
                    # 方法1: 尝试通过torch内置方法获取
                    import torch.version as tv
                    if hasattr(tv, 'cuda') and tv.cuda is not None:
                        cuda_version = tv.cuda
                except (ImportError, AttributeError):
                    # 方法2: 通过nvidia-smi获取驱动版本
                    try:
                        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            cuda_version = f"Driver: {result.stdout.strip()}"
                    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                        pass
                
                print_success(f"CUDA版本: {cuda_version}")
            except Exception:
                print_warning("无法获取CUDA版本信息")
            
            print_success(f"可用GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print_success(f"GPU {i}: {gpu_name}")
        else:
            print_warning("CUDA不可用，将使用CPU训练")
    else:
        print_error("PyTorch未安装或导入失败")

def check_dataset_paths():
    """检查数据集路径"""
    print_header("数据集路径检查")
    
    # LoveDA数据集
    loveda_root = Path('/workspace/data/loveda')
    if loveda_root.exists():
        print_success(f"LoveDA数据集根目录存在: {loveda_root}")
        
        # 检查子目录结构
        expected_dirs = [
            'Train/Rural/images_png',
            'Train/Rural/masks_png',
            'Train/Urban/images_png', 
            'Train/Urban/masks_png',
            'Val/Rural/images_png',
            'Val/Rural/masks_png',
            'Val/Urban/images_png',
            'Val/Urban/masks_png',
            'Test/Rural/images_png',
            'Test/Urban/images_png'
        ]
        
        for dir_path in expected_dirs:
            full_path = loveda_root / dir_path
            if full_path.exists():
                file_count = len(list(full_path.glob('*.png')))
                print_success(f"{dir_path}: {file_count} 个文件")
            else:
                print_error(f"{dir_path}: 目录不存在")
    else:
        print_error(f"LoveDA数据集根目录不存在: {loveda_root}")
    
    # MMRS1M数据集
    mmrs1m_root = Path('/workspace/data/mmrs1m/data')
    if mmrs1m_root.exists():
        print_success(f"MMRS1M数据集根目录存在: {mmrs1m_root}")
        
        # 检查主要子目录
        main_dirs = ['classification', 'detection', 'caption', 'VQA', 'RSVG', 'json']
        for dir_name in main_dirs:
            dir_path = mmrs1m_root / dir_name
            if dir_path.exists():
                print_success(f"MMRS1M/{dir_name}: 目录存在")
            else:
                print_warning(f"MMRS1M/{dir_name}: 目录不存在")
    else:
        print_error(f"MMRS1M数据集根目录不存在: {mmrs1m_root}")

def check_pretrained_weights():
    """检查预训练权重"""
    print_header("预训练权重检查")
    
    weights_dir = Path('/workspace/weights')
    if not weights_dir.exists():
        print_error(f"权重目录不存在: {weights_dir}")
        return
    
    print_success(f"权重目录存在: {weights_dir}")
    
    # 检查权重文件
    weight_files = [
        'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        'best_mIoU_iter_6000.pth'
    ]
    
    for weight_file in weight_files:
        weight_path = weights_dir / weight_file
        if weight_path.exists():
            file_size = weight_path.stat().st_size / (1024 * 1024)  # MB
            print_success(f"{weight_file}: 存在 ({file_size:.1f} MB)")
            
            # 尝试加载权重文件
            if TORCH_AVAILABLE and torch is not None:
                try:
                    checkpoint = torch.load(weight_path, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        keys = list(checkpoint.keys())
                        print_success(f"  权重文件结构: {keys[:5]}...")
                    else:
                        print_success(f"  权重文件类型: {type(checkpoint)}")
                except Exception as e:
                    print_error(f"  无法加载权重文件: {e}")
            else:
                print_warning("  PyTorch不可用，跳过权重文件加载测试")
        else:
            print_error(f"{weight_file}: 不存在")

def check_project_structure():
    """检查项目结构"""
    print_header("项目结构检查")
    
    project_root = Path('/workspace/code/MapSage_V5')
    if not project_root.exists():
        print_error(f"项目根目录不存在: {project_root}")
        return
    
    print_success(f"项目根目录存在: {project_root}")
    
    # 检查关键目录和文件
    important_paths = [
        'configs',
        'mmseg_custom',
        'scripts',
        'scripts/train.py',
        'configs/train_dinov3_mmrs1m_t20_gcu.py',
        'configs/train_dinov3_loveda_t20_gcu.py'
    ]
    
    for path_str in important_paths:
        path = project_root / path_str
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.iterdir()))
                print_success(f"{path_str}/: 目录存在 ({file_count} 个项目)")
            else:
                print_success(f"{path_str}: 文件存在")
        else:
            print_error(f"{path_str}: 不存在")

def check_training_configs():
    """检查训练配置文件"""
    print_header("训练配置文件检查")
    
    config_files = [
        '/workspace/code/MapSage_V5/configs/train_dinov3_mmrs1m_t20_gcu.py',
        '/workspace/code/MapSage_V5/configs/train_dinov3_loveda_t20_gcu.py'
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            print_success(f"配置文件存在: {config_path.name}")
            
            # 检查配置文件内容
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 检查关键配置
                if 'data_root' in content:
                    print_success(f"  包含data_root配置")
                if 'checkpoint' in content:
                    print_success(f"  包含checkpoint配置")
                if 'work_dir' in content:
                    print_success(f"  包含work_dir配置")
                    
            except Exception as e:
                print_error(f"  无法读取配置文件: {e}")
        else:
            print_error(f"配置文件不存在: {config_path}")

def check_disk_space():
    """检查磁盘空间"""
    print_header("磁盘空间检查")
    
    try:
        # 检查工作目录空间
        result = subprocess.run(['df', '-h', '/workspace'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                header = lines[0]
                data = lines[1]
                print(f"工作目录磁盘使用情况:")
                print(f"  {header}")
                print(f"  {data}")
                
                # 解析可用空间
                parts = data.split()
                if len(parts) >= 4:
                    available = parts[3]
                    print_success(f"可用空间: {available}")
        else:
            print_warning("无法获取磁盘空间信息")
            
    except Exception as e:
        print_warning(f"磁盘空间检查失败: {e}")

def check_gpu_memory():
    """检查GPU内存"""
    print_header("GPU内存检查")
    
    if not TORCH_AVAILABLE or torch is None:
        print_warning("PyTorch不可用，跳过GPU内存检查")
        return
    
    if not torch.cuda.is_available():
        print_warning("CUDA不可用，跳过GPU内存检查")
        return
    
    try:
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            total_memory_gb = total_memory / (1024**3)
            
            # 获取当前内存使用情况
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)
            
            allocated_gb = allocated / (1024**3)
            cached_gb = cached / (1024**3)
            
            print_success(f"GPU {i} ({gpu_name}):")
            print_success(f"  总内存: {total_memory_gb:.1f} GB")
            print_success(f"  已分配: {allocated_gb:.1f} GB")
            print_success(f"  已缓存: {cached_gb:.1f} GB")
            
    except Exception as e:
        print_error(f"GPU内存检查失败: {e}")

def main():
    """主函数"""
    print("T20环境验证开始")
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行各项检查
    check_python_environment()
    check_dataset_paths()
    check_pretrained_weights()
    check_project_structure()
    check_training_configs()
    check_disk_space()
    check_gpu_memory()
    
    print_header("验证完成")
    print("T20环境验证完成！")
    print("请根据上述检查结果修复任何发现的问题")
    print("环境就绪后即可开始训练")

if __name__ == '__main__':
    main()