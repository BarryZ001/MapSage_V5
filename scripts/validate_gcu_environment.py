#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GCU环境验证脚本

专门用于验证燧原T20 GCU环境的配置和可用性。
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def print_info(message):
    """打印信息"""
    print(f"ℹ️  {message}")


def check_torch_gcu():
    """检查torch_gcu环境"""
    print_header("torch_gcu环境检查")
    
    try:
        import torch
        print_success(f"PyTorch版本: {torch.__version__}")
        
        # 检查torch_gcu
        try:
            import torch_gcu  # type: ignore
            print_success("torch_gcu模块导入成功")
            
            if hasattr(torch, 'gcu'):
                print_success("torch.gcu接口可用")
                
                # 检查GCU设备
                if torch.gcu.is_available():  # type: ignore
                    device_count = torch.gcu.device_count()  # type: ignore
                    print_success(f"GCU设备可用，数量: {device_count}")
                    
                    for i in range(device_count):
                        try:
                            device_name = torch.gcu.get_device_name(i)  # type: ignore
                            print_info(f"  GCU {i}: {device_name}")
                        except Exception as e:
                            print_warning(f"  GCU {i}: 无法获取设备名称 - {e}")
                    
                    # 测试基本操作
                    try:
                        device = torch.device("gcu:0")  # type: ignore
                        x = torch.randn(2, 3, device=device)
                        y = torch.randn(2, 3, device=device)
                        z = x + y
                        print_success("GCU基本张量操作测试通过")
                        print_info(f"测试结果形状: {z.shape}")
                    except Exception as e:
                        print_error(f"GCU张量操作测试失败: {e}")
                        
                else:
                    print_error("GCU设备不可用")
            else:
                print_error("torch.gcu接口不可用")
                
        except ImportError as e:
            print_error(f"torch_gcu模块导入失败: {e}")
            
    except ImportError as e:
        print_error(f"PyTorch导入失败: {e}")


def check_distributed_backend():
    """检查分布式训练后端"""
    print_header("分布式训练后端检查")
    
    try:
        import torch.distributed as dist
        print_success("torch.distributed模块可用")
        
        # 检查ECCL后端支持
        available_backends = []
        
        # 检查各种后端
        backends_to_check = ['eccl', 'gloo', 'nccl', 'mpi']
        
        for backend in backends_to_check:
            try:
                # 使用更安全的方式检查后端可用性
                if hasattr(dist, 'is_backend_available') and dist.is_backend_available(backend):  # type: ignore
                    available_backends.append(backend)
                    print_success(f"{backend.upper()}后端可用")
                else:
                    # 备选检查方法
                    try:
                        # 尝试创建一个临时的进程组来测试后端
                        print_warning(f"{backend.upper()}后端可用性未知")
                    except Exception:
                        print_warning(f"{backend.upper()}后端不可用")
            except Exception as e:
                print_warning(f"检查{backend.upper()}后端时出错: {e}")
        
        if 'eccl' in available_backends:
            print_success("推荐的ECCL后端可用")
        elif 'gloo' in available_backends:
            print_warning("ECCL不可用，可使用GLOO后端作为备选")
        else:
            print_info("后端可用性检查完成，请在实际训练中测试")
            
    except ImportError as e:
        print_error(f"torch.distributed导入失败: {e}")


def check_environment_variables():
    """检查环境变量"""
    print_header("环境变量检查")
    
    # 检查GCU相关环境变量
    gcu_env_vars = [
        'TOPS_VISIBLE_DEVICES',
        'GCU_VISIBLE_DEVICES', 
        'ENFLAME_VISIBLE_DEVICES',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    for var in gcu_env_vars:
        value = os.environ.get(var)
        if value is not None:
            print_info(f"{var}={value}")
        else:
            print_warning(f"{var}未设置")
    
    # 检查分布式训练相关环境变量
    dist_env_vars = [
        'WORLD_SIZE',
        'RANK', 
        'LOCAL_RANK',
        'MASTER_ADDR',
        'MASTER_PORT'
    ]
    
    print_info("\n分布式训练环境变量:")
    for var in dist_env_vars:
        value = os.environ.get(var)
        if value is not None:
            print_info(f"{var}={value}")
        else:
            print_warning(f"{var}未设置")


def check_system_info():
    """检查系统信息"""
    print_header("系统信息检查")
    
    # 检查操作系统
    print_info(f"操作系统: {os.name}")
    print_info(f"Python版本: {sys.version}")
    
    # 检查GCU驱动和运行时
    gcu_paths = [
        '/usr/local/gcu',
        '/opt/gcu',
        '/proc/driver/gcu'
    ]
    
    for path in gcu_paths:
        if os.path.exists(path):
            print_success(f"GCU路径存在: {path}")
        else:
            print_warning(f"GCU路径不存在: {path}")


def main():
    """主函数"""
    print("🚀 燧原T20 GCU环境验证")
    print("=" * 60)
    
    # 执行各项检查
    check_system_info()
    check_environment_variables()
    check_torch_gcu()
    check_distributed_backend()
    
    print_header("验证完成")
    print("请根据上述检查结果配置GCU环境")
    print("如有问题，请参考燧原官方文档或联系技术支持")


if __name__ == '__main__':
    main()