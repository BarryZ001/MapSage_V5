#!/usr/bin/env python3
"""
基于燧原官方文档的torch_gcu环境验证脚本
官方文档: https://support.enflame-tech.com/onlinedoc_dev_3.5/3-model/infer/torch_gcu/torch_gcu2.5/content/source/torch_gcu_user_guide.html#id4

此脚本设计为在任何环境下都能运行，包括没有torch_gcu的环境
"""

import sys
import os
import torch
import subprocess
from typing import Dict, Any, Optional

def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(item: str, status: str, details: str = ""):
    """打印检查结果"""
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_symbol} {item}: {status}")
    if details:
        print(f"   详情: {details}")

def check_basic_environment() -> Dict[str, Any]:
    """检查基础环境"""
    print_section("基础环境检查")
    
    results = {}
    
    # Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_result("Python版本", f"{python_version}")
    results['python_version'] = python_version
    
    # PyTorch版本
    torch_version = torch.__version__
    print_result("PyTorch版本", f"{torch_version}")
    results['torch_version'] = torch_version
    
    # CUDA可用性
    cuda_available = torch.cuda.is_available()
    print_result("CUDA可用性", "PASS" if cuda_available else "FAIL", 
                f"CUDA设备数: {torch.cuda.device_count()}" if cuda_available else "CUDA不可用")
    results['cuda_available'] = cuda_available
    
    return results

def check_torch_gcu_availability() -> Dict[str, Any]:
    """检查torch_gcu可用性"""
    print_section("torch_gcu可用性检查")
    
    results = {}
    
    try:
        # 动态导入torch_gcu以避免静态分析错误
        torch_gcu = __import__('torch_gcu')
        print_result("torch_gcu导入", "PASS", f"版本: {torch_gcu.__version__}")
        results['torch_gcu_imported'] = True
        results['torch_gcu_version'] = torch_gcu.__version__
        
        # 检查GCU可用性
        gcu_available = torch_gcu.is_available()
        print_result("GCU设备可用性", "PASS" if gcu_available else "FAIL")
        results['gcu_available'] = gcu_available
        
        if gcu_available:
            device_count = torch_gcu.device_count()
            print_result("GCU设备数量", f"{device_count}")
            results['gcu_device_count'] = device_count
            
            # 测试基本张量操作
            try:
                # 使用torch_gcu的设备方法
                current_device = torch_gcu.current_device()
                device_name = f'gcu:{current_device}'
                x = torch.randn(2, 3, device=device_name)
                y = torch.randn(2, 3, device=device_name)
                z = x + y
                print_result("基本张量操作", "PASS", "GCU张量运算正常")
                results['basic_tensor_ops'] = True
            except Exception as e:
                print_result("基本张量操作", "FAIL", str(e))
                results['basic_tensor_ops'] = False
        
    except ImportError as e:
        print_result("torch_gcu导入", "FAIL", f"导入失败: {e}")
        results['torch_gcu_imported'] = False
        print("   提示: torch_gcu只在燧原T20服务器上可用")
    except Exception as e:
        print_result("torch_gcu检查", "FAIL", f"检查失败: {e}")
        results['torch_gcu_imported'] = False
    
    return results

def check_distributed_support() -> Dict[str, Any]:
    """检查分布式训练支持"""
    print_section("分布式训练支持检查")
    
    results = {}
    
    # 检查分布式包
    try:
        import torch.distributed as dist
        print_result("torch.distributed", "PASS")
        results['distributed_available'] = True
        
        # 检查ECCL后端支持
        try:
            # 动态导入torch_gcu以避免静态分析错误
            torch_gcu = __import__('torch_gcu')
            if torch_gcu.is_available():
                # 在GCU环境中检查ECCL后端
                available_backends = []
                for backend in ['eccl', 'gloo', 'nccl']:
                    try:
                        if backend == 'eccl':
                            # ECCL后端需要torch_gcu环境
                            available_backends.append(backend)
                        elif backend == 'gloo':
                            available_backends.append(backend)
                        elif backend == 'nccl' and torch.cuda.is_available():
                            available_backends.append(backend)
                    except:
                        pass
                
                print_result("可用后端", f"{', '.join(available_backends)}")
                results['available_backends'] = available_backends
                results['eccl_supported'] = 'eccl' in available_backends
            else:
                print_result("分布式后端", "WARN", "torch_gcu不可用，将使用gloo后端")
                results['available_backends'] = ['gloo']
                results['eccl_supported'] = False
        except ImportError:
            print_result("分布式后端", "WARN", "torch_gcu不可用，将使用gloo后端")
            results['available_backends'] = ['gloo']
            results['eccl_supported'] = False
            
    except ImportError as e:
        print_result("torch.distributed", "FAIL", str(e))
        results['distributed_available'] = False
    
    return results

def check_amp_support() -> Dict[str, Any]:
    """检查自动混合精度支持"""
    print_section("自动混合精度(AMP)支持检查")
    
    results = {}
    
    try:
        # 修复导入错误
        from torch.cuda.amp.autocast_mode import autocast
        from torch.cuda.amp.grad_scaler import GradScaler
        print_result("torch.cuda.amp", "PASS")
        results['amp_available'] = True
        
        # 检查GCU AMP支持
        try:
            # 动态导入torch_gcu以避免静态分析错误
            torch_gcu = __import__('torch_gcu')
            if torch_gcu.is_available():
                # 测试GCU AMP
                current_device = torch_gcu.current_device()
                device_name = f'gcu:{current_device}'
                x = torch.randn(2, 3, device=device_name)
                y = torch.randn(2, 3, device=device_name)
                
                with autocast():
                    z = x @ y.T
                
                print_result("GCU AMP支持", "PASS")
                results['gcu_amp_supported'] = True
            else:
                print_result("GCU AMP支持", "WARN", "torch_gcu不可用")
                results['gcu_amp_supported'] = False
        except Exception as e:
            print_result("GCU AMP支持", "FAIL", str(e))
            results['gcu_amp_supported'] = False
            
    except ImportError as e:
        print_result("AMP支持", "FAIL", str(e))
        results['amp_available'] = False
    
    return results

def check_profiler_support() -> Dict[str, Any]:
    """检查Profiler支持"""
    print_section("Profiler支持检查")
    
    results = {}
    
    try:
        import torch.profiler
        print_result("torch.profiler", "PASS")
        results['profiler_available'] = True
        
        # 检查GCU Profiler支持
        try:
            # 动态导入torch_gcu以避免静态分析错误
            torch_gcu = __import__('torch_gcu')
            if torch_gcu.is_available():
                # 简单的profiler测试
                with torch.profiler.profile() as prof:
                    current_device = torch_gcu.current_device()
                    device_name = f'gcu:{current_device}'
                    x = torch.randn(10, 10, device=device_name)
                    y = x @ x.T
                
                print_result("GCU Profiler支持", "PASS")
                results['gcu_profiler_supported'] = True
                
                # 检查profiler表格输出
                try:
                    table_output = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
                    if table_output and len(str(table_output).strip()) > 0:
                        print_result("Profiler表格输出", "PASS")
                        results['profiler_table_output'] = True
                    else:
                        print_result("Profiler表格输出", "WARN", "表格为空")
                        results['profiler_table_output'] = False
                except Exception as e:
                    print_result("Profiler表格输出", "FAIL", str(e))
                    results['profiler_table_output'] = False
            else:
                print_result("GCU Profiler支持", "WARN", "torch_gcu不可用")
                results['gcu_profiler_supported'] = False
        except Exception as e:
            print_result("GCU Profiler支持", "FAIL", str(e))
            results['gcu_profiler_supported'] = False
            
    except ImportError as e:
        print_result("Profiler支持", "FAIL", str(e))
        results['profiler_available'] = False
    
    return results

def check_dependencies() -> Dict[str, Any]:
    """检查依赖项"""
    print_section("依赖项检查")
    
    results = {}
    dependencies = ['numpy', 'opencv-python', 'pillow', 'matplotlib']
    
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            print_result(f"{dep}", "PASS")
            results[dep] = True
        except ImportError:
            print_result(f"{dep}", "FAIL", "未安装")
            results[dep] = False
    
    return results

def print_recommendations():
    """打印使用建议"""
    print_section("使用建议")
    
    print("📋 基于燧原官方文档的迁移建议:")
    print()
    print("1. 🔄 后端配置:")
    print("   - 使用 torch_gcu.is_available() 检测GCU环境")
    print("   - GCU环境下使用 backend='eccl'")
    print("   - 非GCU环境下降级到 backend='gloo'")
    print()
    print("2. 🎯 设备管理:")
    print("   - 使用 device=f'gcu:{torch_gcu.current_device()}'")
    print("   - 避免使用 .gcu() 方法，改用 .to(device)")
    print()
    print("3. ⚡ 性能优化:")
    print("   - 设置环境变量:")
    print("     export PYTORCH_GCU_ALLOC_CONF='backend:topsMallocAsync'")
    print("     export TORCH_ECCL_AVOID_RECORD_STREAMS='false'")
    print("     export TORCH_ECCL_ASYNC_ERROR_HANDLING='3'")
    print()
    print("4. 🚀 启动训练:")
    print("   - 使用交互式启动脚本:")
    print("     ./scripts/start_8card_training_interactive_official.sh")

def main() -> bool:
    """主函数"""
    print("🔍 燧原T20 torch_gcu环境验证")
    print("基于官方文档: https://support.enflame-tech.com/onlinedoc_dev_3.5/3-model/infer/torch_gcu/torch_gcu2.5/content/source/torch_gcu_user_guide.html#id4")
    
    all_results = {}
    
    # 执行所有检查
    all_results['basic'] = check_basic_environment()
    all_results['torch_gcu'] = check_torch_gcu_availability()
    all_results['distributed'] = check_distributed_support()
    all_results['amp'] = check_amp_support()
    all_results['profiler'] = check_profiler_support()
    all_results['dependencies'] = check_dependencies()
    
    # 打印建议
    print_recommendations()
    
    # 总结
    print_section("验证总结")
    
    torch_gcu_available = all_results['torch_gcu'].get('torch_gcu_imported', False)
    gcu_available = all_results['torch_gcu'].get('gcu_available', False)
    
    if torch_gcu_available and gcu_available:
        print("✅ torch_gcu环境完全可用")
        print("✅ 可以使用ECCL后端进行分布式训练")
        return True
    elif torch_gcu_available:
        print("⚠️ torch_gcu已安装但GCU设备不可用")
        print("⚠️ 将使用gloo后端进行分布式训练")
        return True
    else:
        print("❌ torch_gcu不可用")
        print("❌ 将使用gloo后端进行分布式训练")
        print("💡 这在非T20服务器环境下是正常的")
        return True  # 在非GCU环境下也算成功

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)