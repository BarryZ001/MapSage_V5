#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练环境验证脚本

验证DINOv3+MMRS-1M训练所需的环境、数据和配置。
适用于T20服务器和本地开发环境。
"""

import os
import sys
import torch
import importlib
import importlib.util
import subprocess
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_success(message):
    """打印成功信息"""
    print(f"✅ {message}")

def print_error(message):
    """打印错误信息"""
    print(f"❌ {message}")

def print_warning(message):
    """打印警告信息"""
    print(f"⚠️  {message}")

def print_info(message):
    """打印信息"""
    print(f"ℹ️  {message}")


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro} (需要>=3.8)"


def check_t20_gcu_environment() -> bool:
    """检查是否为燧原T20 GCU环境"""
    # 检查GCU相关环境变量和文件
    gcu_indicators = [
        os.path.exists('/usr/local/gcu'),
        os.path.exists('/opt/gcu'),
        'GCU' in os.environ.get('PATH', ''),
        os.path.exists('/proc/driver/gcu')
    ]
    return any(gcu_indicators)


def check_pytorch() -> Tuple[bool, str]:
    """检查PyTorch安装"""
    try:
        import torch
        version = torch.__version__
        
        # 优先检查GCU环境
        gcu_available = False
        gcu_count = 0
        try:
            import torch_gcu  # type: ignore
            if hasattr(torch, 'gcu'):
                gcu_available = torch.gcu.is_available()  # type: ignore
                gcu_count = torch.gcu.device_count() if gcu_available else 0  # type: ignore
        except (ImportError, AttributeError):
            pass
        
        # 检查CUDA环境（作为备选）
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        # 检查是否为燧原T20 GCU环境
        is_t20_gcu = check_t20_gcu_environment()
        
        if gcu_available:
            return True, f"✅ PyTorch {version}, GCU可用, {gcu_count}个GCU设备"
        elif cuda_available:
            return True, f"✅ PyTorch {version}, CUDA可用, {gpu_count}个GPU"
        elif is_t20_gcu:
            return True, f"✅ PyTorch {version}, 燧原T20 GCU环境"
        else:
            return True, f"✅ PyTorch {version}, CPU环境 (适配T20 GCU)"
    except ImportError:
        return False, "❌ PyTorch未安装"


def check_torch_gcu() -> Tuple[bool, str]:
    """检查torch-gcu框架"""
    try:
        import torch
        if hasattr(torch, 'gcu'):
            return True, "✅ torch-gcu框架可用"
        else:
            return False, "❌ torch-gcu框架不可用"
    except ImportError:
        return False, "❌ PyTorch未安装，无法检查torch-gcu"


def check_ptex() -> Tuple[bool, str]:
    """检查ptex模块"""
    try:
        # 使用importlib动态导入ptex以避免静态分析错误
        ptex_spec = importlib.util.find_spec('ptex')
        if ptex_spec is None:
            return False, "❌ ptex模块未安装"
        
        ptex = importlib.import_module('ptex')
        # 尝试创建XLA设备来验证可用性
        device = ptex.device('xla')
        return True, f"✅ ptex模块可用, XLA设备: {device}"
    except ImportError:
        return False, "❌ ptex模块未安装"
    except Exception as e:
        return False, f"❌ ptex模块错误: {e}"


def check_mmseg() -> Tuple[bool, str]:
    """检查MMSegmentation"""
    try:
        import mmseg
        version = mmseg.__version__
        return True, f"✅ MMSegmentation {version}"
    except ImportError:
        return False, "❌ MMSegmentation未安装"


def check_custom_modules() -> List[Tuple[bool, str]]:
    """检查自定义模块"""
    results = []
    
    # 先运行QuantStub兼容性修复
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'scripts/fix_quantstub_compatibility.py'], 
                              capture_output=True, text=True, cwd='/workspace/code/MapSage_V5')
        if result.returncode == 0:
            results.append((True, "✅ QuantStub兼容性修复完成"))
        else:
            results.append((False, f"⚠️ QuantStub修复警告: {result.stderr}"))
    except Exception as e:
        results.append((False, f"⚠️ QuantStub修复失败: {e}"))
    
    # 检查自定义数据集
    try:
        from mmseg_custom.datasets import MMRS1MDataset
        results.append((True, "✅ MMRS1MDataset导入成功"))
    except ImportError as e:
        results.append((False, f"❌ MMRS1MDataset导入失败: {e}"))
    
    # 检查自定义变换
    try:
        from mmseg_custom.transforms import MultiModalNormalize
        results.append((True, "✅ MultiModalTransforms导入成功"))
    except ImportError as e:
        results.append((False, f"❌ MultiModalTransforms导入失败: {e}"))
    
    # 检查DINOv3模型
    try:
        from mmseg_custom.models import DINOv3ViT
        results.append((True, "✅ DINOv3ViT导入成功"))
    except ImportError as e:
        results.append((False, f"❌ DINOv3ViT导入失败: {e}"))
    
    return results


def check_data_paths() -> List[Tuple[bool, str]]:
    """检查数据路径"""
    results = []
    
    # T20服务器路径 - 修正为正确的数据路径
    t20_data_path = Path("/workspace/data/mmrs1m/data")
    if t20_data_path.exists():
        results.append((True, f"✅ T20数据路径存在: {t20_data_path}"))
        
        # 检查子目录
        subdirs = ['caption', 'classification', 'detection', 'json', 'RSVG', 'VQA']
        for subdir in subdirs:
            subpath = t20_data_path / subdir
            if subpath.exists():
                results.append((True, f"✅ 子目录存在: {subdir}"))
            else:
                results.append((False, f"❌ 子目录缺失: {subdir}"))
    else:
        results.append((False, f"❌ T20数据路径不存在: {t20_data_path}"))
    
    # 本地测试数据路径
    local_data_path = Path("./data/test_data")
    if local_data_path.exists():
        results.append((True, f"✅ 本地测试数据存在: {local_data_path}"))
    else:
        results.append((False, f"❌ 本地测试数据不存在: {local_data_path}"))
    
    return results


def check_pretrained_weights() -> List[Tuple[bool, str]]:
    """检查预训练权重"""
    results = []
    
    # T20服务器权重路径 - 用户提供的正确路径
    t20_weights_path = Path("/workspace/weights/pretrained/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    if t20_weights_path.exists():
        results.append((True, f"✅ T20预训练权重存在: {t20_weights_path}"))
    else:
        results.append((False, f"❌ T20预训练权重不存在: {t20_weights_path}"))
    
    return results


def check_config_file() -> Tuple[bool, str]:
    """检查训练配置文件"""
    config_path = Path("configs/train_dinov3_mmrs1m.py")
    if config_path.exists():
        try:
            # 尝试导入配置
            sys.path.insert(0, str(config_path.parent))
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec is not None and spec.loader is not None:
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)
                return True, f"✅ 配置文件有效: {config_path}"
            else:
                return False, f"❌ 配置文件导入失败: {config_path}"
        except Exception as e:
            return False, f"❌ 配置文件有误: {e}"
    else:
        return False, f"❌ 配置文件不存在: {config_path}"


def check_work_directory() -> Tuple[bool, str]:
    """检查工作目录"""
    work_dir = Path("./work_dirs/dinov3_mmrs1m_stage1")
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        return True, f"✅ 工作目录就绪: {work_dir}"
    except Exception as e:
        return False, f"❌ 工作目录创建失败: {e}"


def check_gpu_environment():
    """检查GPU/GCU环境"""
    print("\n🔍 检查GPU/GCU环境...")
    
    try:
        import torch
        
        # 优先检查GCU环境
        gcu_available = False
        try:
            import torch_gcu  # type: ignore
            if hasattr(torch, 'gcu'):
                gcu_available = torch.gcu.is_available()  # type: ignore
                if gcu_available:
                    gcu_count = torch.gcu.device_count()  # type: ignore
                    print_success(f"可用GCU数量: {gcu_count}")
                    for i in range(gcu_count):
                        gcu_name = torch.gcu.get_device_name(i)  # type: ignore
                        print_info(f"  GCU {i}: {gcu_name}")
                    return True
        except (ImportError, AttributeError):
            pass
        
        # 检查CUDA环境（作为备选）
        if torch.cuda.is_available():
            print_success(f"可用GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print_warning("未检测到GPU/GCU设备，将使用CPU训练")
            return False
            
    except Exception as e:
        print_error(f"检查GPU/GCU环境时出错: {e}")
        return False


def main():
    """主验证函数"""
    print("🔍 DINOv3 + MMRS-1M 训练环境验证")
    
    # 检查是否为T20 GCU环境
    is_t20_gcu = check_t20_gcu_environment()
    if is_t20_gcu:
        print("🔥 检测到燧原T20 GCU环境")
    else:
        print("💻 标准环境")
    
    print("=" * 50)
    
    all_passed = True
    
    # 基础环境检查
    print("\n📋 基础环境检查:")
    checks = [
        check_python_version(),
        check_pytorch(),
        check_mmseg(),
        check_config_file(),
        check_work_directory()
    ]
    
    # 如果是T20 GCU环境，添加GCU相关检查
    if is_t20_gcu:
        checks.extend([
            check_torch_gcu(),
            check_ptex()
        ])
    
    for passed, message in checks:
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    # 自定义模块检查
    print("\n🔧 自定义模块检查:")
    custom_checks = check_custom_modules()
    for passed, message in custom_checks:
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    # 数据路径检查
    print("\n📁 数据路径检查:")
    data_checks = check_data_paths()
    for passed, message in data_checks:
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    # 预训练权重检查
    print("\n⚖️  预训练权重检查:")
    weight_checks = check_pretrained_weights()
    for passed, message in weight_checks:
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    # GPU/XLA设备信息
    if is_t20_gcu:
        print("\n🖥️  XLA设备信息:")
        try:
            # 使用importlib动态导入ptex以避免静态分析错误
            ptex_spec = importlib.util.find_spec('ptex')
            if ptex_spec is None:
                print("  ❌ ptex模块未安装，无法检查XLA设备")
            else:
                ptex = importlib.import_module('ptex')
                # 尝试创建XLA设备来验证可用性
                device = ptex.device('xla')
                print(f"  ✅ XLA设备可用: {device}")
        except Exception as e:
            print(f"  ❌ XLA设备检查错误: {e}")
    else:
        print("\n🖥️  GPU信息:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("  ❌ 无可用GPU")
    
    # 总结
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ 所有检查通过！可以开始训练。")
        print("\n🚀 启动训练命令:")
        print("  bash scripts/train_dinov3_mmrs1m.sh")
        return 0
    else:
        print("❌ 部分检查未通过，请修复后再开始训练。")
        return 1


if __name__ == "__main__":
    sys.exit(main())