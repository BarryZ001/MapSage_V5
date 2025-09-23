#!/usr/bin/env python3
"""
GCU完整功能验证脚本
基于已确认的torch_gcu可用性，进行全面的功能测试

注意: 此脚本设计用于T20服务器环境，本地开发环境可能缺少torch_gcu模块
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime

# 导入torch，torch_gcu在服务器环境中可用
try:
    import torch
except ImportError:
    torch = None  # type: ignore

def log_info(message):
    """记录信息日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO {timestamp}] {message}")

def log_error(message):
    """记录错误日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR {timestamp}] {message}")

def log_success(message):
    """记录成功日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SUCCESS {timestamp}] {message}")

def test_torch_gcu_basic():
    """测试torch_gcu基本功能"""
    log_info("测试torch_gcu基本功能...")
    
    try:
        import torch  # type: ignore
        import torch_gcu  # type: ignore
        
        # 检查可用性
        is_available = torch_gcu.is_available()
        log_success(f"torch_gcu.is_available(): {is_available}")
        
        if not is_available:
            return False, "torch_gcu不可用"
        
        # 检查设备数量
        device_count = torch_gcu.device_count()
        log_success(f"GCU设备数量: {device_count}")
        
        return True, {
            "available": is_available,
            "device_count": device_count
        }
        
    except ImportError as e:
        log_error(f"torch_gcu模块导入失败: {str(e)}")
        return False, f"模块导入失败: {str(e)}"
    except Exception as e:
        log_error(f"torch_gcu基本功能测试失败: {str(e)}")
        return False, str(e)

def test_tensor_operations():
    """测试基本张量操作"""
    log_info("测试GCU张量操作...")
    
    try:
        import torch  # type: ignore
        import torch_gcu  # type: ignore
        
        if not torch_gcu.is_available():
            return False, "torch_gcu不可用"
        
        # 创建CPU张量
        x_cpu = torch.randn(3, 4)
        log_info(f"创建CPU张量: shape={x_cpu.shape}")
        
        # 移动到GCU
        x_gcu = x_cpu.to('gcu:0')
        log_success(f"张量移动到GCU: device={x_gcu.device}")
        
        # 基本运算
        y_gcu = x_gcu * 2
        z_gcu = y_gcu + 1
        result = z_gcu.sum()
        
        log_success(f"GCU计算结果: {result.item()}")
        
        # 移回CPU验证
        result_cpu = result.cpu()
        log_success(f"结果移回CPU: {result_cpu.item()}")
        
        return True, {
            "tensor_shape": list(x_cpu.shape),
            "gcu_device": str(x_gcu.device),
            "computation_result": result_cpu.item()
        }
        
    except ImportError as e:
        log_error(f"模块导入失败: {str(e)}")
        return False, f"模块导入失败: {str(e)}"
    except Exception as e:
        log_error(f"张量操作测试失败: {str(e)}")
        return False, str(e)

def test_eccl_backend():
    """测试ECCL分布式后端"""
    log_info("测试ECCL分布式后端...")
    
    try:
        import torch  # type: ignore
        import torch.distributed as dist  # type: ignore
        
        # 检查ECCL后端可用性
        # 注意：is_eccl_available可能在某些版本中不存在
        eccl_available = hasattr(dist, 'is_eccl_available')
        
        if eccl_available:
            try:
                eccl_status = dist.is_eccl_available()  # type: ignore
                log_success(f"ECCL后端可用性: {eccl_status}")
            except AttributeError:
                eccl_status = False
                log_error("is_eccl_available方法不存在，可能是版本问题")
        else:
            eccl_status = False
            log_error("ECCL后端检查方法不可用")
        
        return True, {"eccl_available": eccl_status}
        
    except ImportError as e:
        log_error(f"分布式模块导入失败: {str(e)}")
        return False, f"模块导入失败: {str(e)}"
    except Exception as e:
        log_error(f"ECCL后端测试失败: {str(e)}")
        return False, str(e)

def test_multi_gpu_detection():
    """测试多GPU检测"""
    log_info("测试多GPU检测...")
    
    try:
        import torch_gcu
        
        device_count = torch_gcu.device_count()
        log_info(f"检测到 {device_count} 个GCU设备")
        
        devices_info = []
        for i in range(device_count):
            try:
                # 尝试在每个设备上创建张量
                device_name = f'gcu:{i}'
                x = torch.randn(2, 2).to(device_name)
                devices_info.append({
                    "device_id": i,
                    "device_name": device_name,
                    "accessible": True
                })
                log_success(f"设备 {device_name} 可访问")
            except Exception as e:
                devices_info.append({
                    "device_id": i,
                    "device_name": f'gcu:{i}',
                    "accessible": False,
                    "error": str(e)
                })
                log_error(f"设备 gcu:{i} 不可访问: {str(e)}")
        
        return True, {
            "total_devices": device_count,
            "devices": devices_info
        }
        
    except Exception as e:
        log_error(f"多GPU检测失败: {str(e)}")
        return False, str(e)

def test_memory_operations():
    """测试内存操作"""
    log_info("测试GCU内存操作...")
    
    try:
        import torch
        import torch_gcu
        
        # 创建较大的张量测试内存
        large_tensor = torch.randn(1000, 1000).to('gcu:0')
        log_success(f"创建大张量成功: shape={large_tensor.shape}")
        
        # 内存拷贝测试
        copied_tensor = large_tensor.clone()
        log_success("张量克隆成功")
        
        # 释放内存
        del large_tensor
        del copied_tensor
        
        # 如果有内存清理函数，调用它
        if hasattr(torch_gcu, 'empty_cache'):
            torch_gcu.empty_cache()
            log_success("内存缓存清理成功")
        
        return True, {"memory_test": "passed"}
        
    except Exception as e:
        log_error(f"内存操作测试失败: {str(e)}")
        return False, str(e)

def generate_report(results):
    """生成测试报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/tmp/gcu_full_validation_report_{timestamp}.json"
    
    report = {
        "timestamp": timestamp,
        "test_results": results,
        "summary": {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results.values() if r["success"]),
            "failed_tests": sum(1 for r in results.values() if not r["success"])
        }
    }
    
    # 写入JSON报告
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    log_success(f"详细报告已保存到: {report_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("GCU功能验证摘要")
    print("="*60)
    print(f"总测试数: {report['summary']['total_tests']}")
    print(f"通过测试: {report['summary']['passed_tests']}")
    print(f"失败测试: {report['summary']['failed_tests']}")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result["success"]:
            print(f"    错误: {result['error']}")
    
    print("="*60)
    return report_file

def main():
    """主函数"""
    log_info("开始GCU完整功能验证...")
    
    # 收集系统信息
    log_info("收集系统信息...")
    system_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "未设置"),
            "TOPS_INSTALL_PATH": os.environ.get("TOPS_INSTALL_PATH", "未设置"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "未设置")
        }
    }
    
    # 执行测试
    tests = {
        "torch_gcu_basic": test_torch_gcu_basic,
        "tensor_operations": test_tensor_operations,
        "eccl_backend": test_eccl_backend,
        "multi_gpu_detection": test_multi_gpu_detection,
        "memory_operations": test_memory_operations
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        log_info(f"执行测试: {test_name}")
        try:
            success, data = test_func()
            results[test_name] = {
                "success": success,
                "data": data if success else None,
                "error": data if not success else None
            }
        except Exception as e:
            log_error(f"测试 {test_name} 执行异常: {str(e)}")
            results[test_name] = {
                "success": False,
                "data": None,
                "error": f"执行异常: {str(e)}"
            }
    
    # 添加系统信息到结果
    results["system_info"] = {
        "success": True,
        "data": system_info,
        "error": None
    }
    
    # 生成报告
    report_file = generate_report(results)
    
    # 返回总体结果
    all_passed = all(r["success"] for k, r in results.items() if k != "system_info")
    
    if all_passed:
        log_success("所有GCU功能验证通过！")
        return 0
    else:
        log_error("部分GCU功能验证失败，请查看详细报告")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)