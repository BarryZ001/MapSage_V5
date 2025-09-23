#!/usr/bin/env python3
"""
ECCL Backend and Distributed Training Test Script
测试ECCL后端和分布式训练功能的脚本

注意：此脚本需要在安装了torch_gcu的环境中运行（如T20服务器）
在本地开发环境中，torch_gcu模块不可用是正常的
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

# 类型检查忽略：torch_gcu只在特定环境中可用
# type: ignore

class ECCLTester:
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
        }
        
    def log(self, message, level='INFO'):
        """日志输出"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {level}: {message}")
        
    def run_test(self, test_name, test_func):
        """运行单个测试"""
        self.log(f"开始测试: {test_name}")
        self.test_results['summary']['total'] += 1
        
        try:
            result = test_func()
            if result:
                self.test_results['tests'][test_name] = {
                    'status': 'PASSED',
                    'message': 'Test completed successfully',
                    'details': result if isinstance(result, dict) else {}
                }
                self.test_results['summary']['passed'] += 1
                self.log(f"✓ {test_name} - PASSED")
            else:
                self.test_results['tests'][test_name] = {
                    'status': 'FAILED',
                    'message': 'Test returned False',
                    'details': {}
                }
                self.test_results['summary']['failed'] += 1
                self.log(f"✗ {test_name} - FAILED")
                
        except Exception as e:
            self.test_results['tests'][test_name] = {
                'status': 'FAILED',
                'message': str(e),
                'details': {}
            }
            self.test_results['summary']['failed'] += 1
            self.log(f"✗ {test_name} - FAILED: {e}")
            
    def test_environment_variables(self):
        """测试环境变量配置"""
        required_vars = [
            'ECCL_ROOT',
            'TOPS_SDK_PATH',
            'LD_LIBRARY_PATH',
            'PATH'
        ]
        
        results = {}
        all_present = True
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                results[var] = value
                self.log(f"  ✓ {var} = {value}")
            else:
                results[var] = None
                all_present = False
                self.log(f"  ✗ {var} 未设置")
                
        # 检查ECCL相关的环境变量
        eccl_vars = [
            'ECCL_ASYNC_DISABLE',
            'ECCL_MAX_NCHANNELS', 
            'ECCL_RUNTIME_3_0_ENABLE'
        ]
        
        for var in eccl_vars:
            value = os.environ.get(var)
            if value:
                results[var] = value
                self.log(f"  ✓ {var} = {value}")
            else:
                results[var] = None
                self.log(f"  ! {var} 未设置 (可选)")
                
        return all_present and results
        
    def test_torch_gcu_import(self):
        """测试torch_gcu导入"""
        try:
            import torch
            torch_version = torch.__version__
            self.log(f"  ✓ torch version: {torch_version}")
            
            try:
                import torch_gcu
                torch_gcu_version = torch_gcu.__version__
                self.log(f"  ✓ torch_gcu version: {torch_gcu_version}")
                
                # 检查GCU设备
                if torch_gcu.is_available():
                    device_count = torch_gcu.device_count()
                    self.log(f"  ✓ GCU devices available: {device_count}")
                    
                    # 获取设备信息
                    devices_info = []
                    for i in range(device_count):
                        device_name = torch_gcu.get_device_name(i)
                        devices_info.append(f"Device {i}: {device_name}")
                        self.log(f"    - {device_name}")
                        
                    return {
                        'torch_version': torch_version,
                        'torch_gcu_version': torch_gcu_version,
                        'gcu_available': True,
                        'device_count': device_count,
                        'devices': devices_info
                    }
                else:
                    self.log("  ! GCU设备不可用")
                    return {
                        'torch_version': torch_version,
                        'torch_gcu_version': torch_gcu_version,
                        'gcu_available': False,
                        'device_count': 0,
                        'devices': []
                    }
            except ImportError as torch_gcu_error:
                self.log(f"  ✗ torch_gcu导入失败: {torch_gcu_error}")
                return {
                    'torch_version': torch_version,
                    'torch_gcu_available': False,
                    'error': str(torch_gcu_error)
                }
                
        except ImportError as e:
            self.log(f"  ✗ torch导入失败: {e}")
            return False
            
    def test_eccl_library(self):
        """测试ECCL库文件"""
        possible_paths = [
            '/usr/local/eccl/lib/libeccl.so',
            '/usr/lib/libeccl.so',
            '/usr/lib/x86_64-linux-gnu/libeccl.so',
            '/opt/eccl/lib/libeccl.so'
        ]
        
        found_paths = []
        for path in possible_paths:
            if os.path.exists(path):
                found_paths.append(path)
                self.log(f"  ✓ 找到ECCL库: {path}")
                
        if found_paths:
            # 尝试使用ldd检查库依赖
            try:
                result = subprocess.run(['ldd', found_paths[0]], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.log(f"  ✓ 库依赖检查通过")
                    return {
                        'library_paths': found_paths,
                        'dependencies_ok': True,
                        'ldd_output': result.stdout
                    }
                else:
                    self.log(f"  ! 库依赖检查警告: {result.stderr}")
                    return {
                        'library_paths': found_paths,
                        'dependencies_ok': False,
                        'ldd_error': result.stderr
                    }
            except Exception as e:
                self.log(f"  ! 无法检查库依赖: {e}")
                return {
                    'library_paths': found_paths,
                    'dependencies_ok': None,
                    'error': str(e)
                }
        else:
            self.log("  ✗ 未找到ECCL库文件")
            return False
            
    def test_eccl_backend_availability(self):
        """测试ECCL后端可用性"""
        try:
            import torch
            import torch.distributed as dist
            
            # 检查可用的后端
            available_backends = []
            
            # 检查NCCL
            if dist.is_nccl_available():
                available_backends.append('nccl')
                
            # 检查Gloo
            if dist.is_gloo_available():
                available_backends.append('gloo')
                
            # 检查MPI
            if dist.is_mpi_available():
                available_backends.append('mpi')
                
            self.log(f"  可用的分布式后端: {available_backends}")
            
            # 尝试检查ECCL后端
            # 注意：这里需要根据实际的torch_gcu实现来调整
            try:
                # 某些版本的torch_gcu可能有特定的ECCL后端检查方法
                try:
                    import torch_gcu
                    if hasattr(torch_gcu, 'is_eccl_available'):
                        eccl_available = torch_gcu.is_eccl_available()
                        if eccl_available:
                            available_backends.append('eccl')
                            self.log("  ✓ ECCL后端可用")
                        else:
                            self.log("  ✗ ECCL后端不可用")
                    else:
                        self.log("  ! 无法直接检查ECCL后端可用性")
                except ImportError:
                    self.log("  ! torch_gcu未安装，无法检查ECCL后端")
                    
            except Exception as e:
                self.log(f"  ! ECCL后端检查异常: {e}")
                
            return {
                'available_backends': available_backends,
                'eccl_available': 'eccl' in available_backends
            }
            
        except ImportError as e:
            self.log(f"  ✗ 导入torch.distributed失败: {e}")
            return False
            
    def test_simple_tensor_operations(self):
        """测试简单的张量操作"""
        try:
            import torch
            
            try:
                import torch_gcu
                
                if not torch_gcu.is_available():
                    self.log("  ! GCU不可用，跳过张量操作测试")
                    return {'skipped': True, 'reason': 'GCU not available'}
                    
                # 创建张量并移动到GCU
                device = torch_gcu.device(0)
                
                # 简单的张量操作
                x = torch.randn(3, 3).to(device)
                y = torch.randn(3, 3).to(device)
                z = torch.mm(x, y)
                
                # 移回CPU验证
                z_cpu = z.cpu()
                
                self.log(f"  ✓ 张量操作成功，结果形状: {z_cpu.shape}")
                
                return {
                    'device': str(device),
                    'operation': 'matrix_multiplication',
                    'input_shape': [3, 3],
                    'output_shape': list(z_cpu.shape),
                    'success': True
                }
                
            except ImportError:
                self.log("  ! torch_gcu未安装，跳过张量操作测试")
                return {'skipped': True, 'reason': 'torch_gcu not available'}
            
        except Exception as e:
            self.log(f"  ✗ 张量操作失败: {e}")
            return False
            
    def test_distributed_init(self):
        """测试分布式初始化（单进程模拟）"""
        try:
            import torch
            import torch.distributed as dist
            import torch.multiprocessing as mp
            
            # 设置环境变量
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['RANK'] = '0'
            
            # 尝试初始化不同的后端
            backends_to_test = ['gloo']  # 先测试最基本的后端
            
            results = {}
            
            for backend in backends_to_test:
                try:
                    self.log(f"  测试 {backend} 后端初始化...")
                    
                    # 初始化进程组
                    dist.init_process_group(
                        backend=backend,
                        world_size=1,
                        rank=0
                    )
                    
                    # 检查初始化状态
                    if dist.is_initialized():
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()
                        
                        self.log(f"    ✓ {backend} 初始化成功 (world_size={world_size}, rank={rank})")
                        results[backend] = {
                            'initialized': True,
                            'world_size': world_size,
                            'rank': rank
                        }
                        
                        # 清理
                        dist.destroy_process_group()
                    else:
                        results[backend] = {'initialized': False}
                        
                except Exception as e:
                    self.log(f"    ✗ {backend} 初始化失败: {e}")
                    results[backend] = {
                        'initialized': False,
                        'error': str(e)
                    }
                    
            return results
            
        except ImportError as e:
            self.log(f"  ✗ 导入分布式模块失败: {e}")
            return False
            
    def test_system_info(self):
        """收集系统信息"""
        try:
            info = {}
            
            # 操作系统信息
            try:
                with open('/etc/os-release', 'r') as f:
                    os_info = f.read()
                    info['os_release'] = os_info
            except:
                info['os_release'] = 'Unknown'
                
            # Python信息
            info['python_version'] = sys.version
            info['python_executable'] = sys.executable
            
            # 内存信息
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal' in line:
                            info['total_memory'] = line.strip()
                            break
            except:
                info['total_memory'] = 'Unknown'
                
            # GPU/GCU信息
            try:
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if result.returncode == 0:
                    # 查找可能的GCU设备
                    gcu_devices = []
                    for line in result.stdout.split('\n'):
                        if any(keyword in line.lower() for keyword in ['enflame', 'gcu', 'dtu']):
                            gcu_devices.append(line.strip())
                    info['gcu_devices'] = gcu_devices
            except:
                info['gcu_devices'] = []
                
            self.log("  ✓ 系统信息收集完成")
            return info
            
        except Exception as e:
            self.log(f"  ✗ 系统信息收集失败: {e}")
            return False
            
    def run_all_tests(self):
        """运行所有测试"""
        self.log("开始ECCL后端和分布式训练功能测试")
        self.log("=" * 50)
        
        # 定义测试列表
        tests = [
            ("系统信息收集", self.test_system_info),
            ("环境变量检查", self.test_environment_variables),
            ("torch_gcu导入测试", self.test_torch_gcu_import),
            ("ECCL库文件检查", self.test_eccl_library),
            ("ECCL后端可用性", self.test_eccl_backend_availability),
            ("简单张量操作", self.test_simple_tensor_operations),
            ("分布式初始化测试", self.test_distributed_init),
        ]
        
        # 运行所有测试
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()  # 空行分隔
            
        # 输出测试摘要
        self.print_summary()
        
        # 保存测试结果
        self.save_results()
        
    def print_summary(self):
        """打印测试摘要"""
        self.log("测试摘要")
        self.log("=" * 30)
        
        summary = self.test_results['summary']
        self.log(f"总测试数: {summary['total']}")
        self.log(f"通过: {summary['passed']}")
        self.log(f"失败: {summary['failed']}")
        self.log(f"跳过: {summary['skipped']}")
        
        success_rate = (summary['passed'] / summary['total'] * 100) if summary['total'] > 0 else 0
        self.log(f"成功率: {success_rate:.1f}%")
        
        if summary['failed'] > 0:
            self.log("\n失败的测试:")
            for test_name, result in self.test_results['tests'].items():
                if result['status'] == 'FAILED':
                    self.log(f"  - {test_name}: {result['message']}")
                    
    def save_results(self):
        """保存测试结果到文件"""
        results_file = '/tmp/eccl_test_results.json'
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            self.log(f"\n测试结果已保存到: {results_file}")
            
            # 同时生成可读的报告
            report_file = '/tmp/eccl_test_report.txt'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("ECCL Backend and Distributed Training Test Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"测试时间: {self.test_results['timestamp']}\n\n")
                
                for test_name, result in self.test_results['tests'].items():
                    f.write(f"测试: {test_name}\n")
                    f.write(f"状态: {result['status']}\n")
                    f.write(f"信息: {result['message']}\n")
                    if result['details']:
                        f.write(f"详情: {json.dumps(result['details'], indent=2, ensure_ascii=False)}\n")
                    f.write("-" * 30 + "\n")
                    
                f.write(f"\n测试摘要:\n")
                f.write(f"总数: {self.test_results['summary']['total']}\n")
                f.write(f"通过: {self.test_results['summary']['passed']}\n")
                f.write(f"失败: {self.test_results['summary']['failed']}\n")
                
            self.log(f"测试报告已保存到: {report_file}")
            
        except Exception as e:
            self.log(f"保存测试结果失败: {e}")

def main():
    """主函数"""
    print("ECCL Backend and Distributed Training Test Script")
    print("=" * 50)
    
    tester = ECCLTester()
    tester.run_all_tests()
    
    # 根据测试结果返回适当的退出码
    if tester.test_results['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()