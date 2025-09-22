#!/usr/bin/env python3
"""
T20 GCU环境完整验证脚本
基于成功经验创建，用于验证训练环境是否正确配置

使用方法:
python3 validate_t20_environment_complete.py

作者: MapSage V5 Team
日期: 2024年
"""

import os
import sys
import subprocess
import importlib
import traceback
from pathlib import Path

class T20EnvironmentValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        
    def log_success(self, message):
        print(f"✅ {message}")
        self.success_count += 1
        
    def log_error(self, message):
        print(f"❌ {message}")
        self.errors.append(message)
        
    def log_warning(self, message):
        print(f"⚠️  {message}")
        self.warnings.append(message)
        
    def log_info(self, message):
        print(f"ℹ️  {message}")
        
    def check_python_environment(self):
        """检查Python环境"""
        print("\n🐍 检查Python环境...")
        self.total_checks += 1
        
        try:
            python_version = sys.version_info
            if python_version.major == 3 and python_version.minor >= 8:
                self.log_success(f"Python版本: {sys.version}")
            else:
                self.log_error(f"Python版本过低: {sys.version}, 需要Python 3.8+")
                
            # 检查Python路径
            self.log_info(f"Python可执行文件: {sys.executable}")
            self.log_info(f"Python路径: {sys.path[:3]}...")  # 只显示前3个路径
            
        except Exception as e:
            self.log_error(f"Python环境检查失败: {e}")
            
    def check_pytorch_installation(self):
        """检查PyTorch安装"""
        print("\n🔥 检查PyTorch安装...")
        self.total_checks += 1
        
        try:
            import torch
            self.log_success(f"PyTorch版本: {torch.__version__}")
            
            # 检查CUDA支持
            if torch.cuda.is_available():
                self.log_success(f"CUDA可用，版本: {torch.version.cuda}")
                self.log_info(f"CUDA设备数量: {torch.cuda.device_count()}")
            else:
                self.log_warning("CUDA不可用")
                
            # 检查分布式支持
            if torch.distributed.is_available():
                self.log_success("分布式训练支持可用")
            else:
                self.log_error("分布式训练支持不可用")
                
        except ImportError:
            self.log_error("PyTorch未安装")
        except Exception as e:
            self.log_error(f"PyTorch检查失败: {e}")
            
    def check_torch_gcu(self):
        """检查torch_gcu (GCU支持)"""
        print("\n🚀 检查torch_gcu (GCU支持)...")
        self.total_checks += 1
        
        try:
            import torch_gcu  # type: ignore
            self.log_success("torch_gcu模块已安装")
            
            # 检查版本
            if hasattr(torch_gcu, '__version__'):
                self.log_info(f"torch_gcu版本: {torch_gcu.__version__}")
                
            # 检查设备可用性
            if hasattr(torch_gcu, 'is_available') and torch_gcu.is_available():
                device_count = getattr(torch_gcu, 'device_count', lambda: 0)()
                self.log_success(f"GCU设备可用，设备数量: {device_count}")
                
                # 测试设备访问
                for i in range(min(device_count, 8)):  # 最多检查8个设备
                    try:
                        device = torch_gcu.device(i)
                        self.log_info(f"  GCU设备 {i}: 可访问")
                    except Exception as e:
                        self.log_warning(f"  GCU设备 {i}: 访问失败 - {e}")
            else:
                self.log_error("GCU设备不可用")
                
        except ImportError:
            self.log_error("torch_gcu未安装 (T20环境必需)")
        except Exception as e:
            self.log_error(f"torch_gcu检查失败: {e}")
            
    def check_distributed_backends(self):
        """检查分布式后端"""
        print("\n🌐 检查分布式后端...")
        self.total_checks += 1
        
        try:
            import torch.distributed as dist
            
            # 检查NCCL
            try:
                if hasattr(dist, 'is_nccl_available') and dist.is_nccl_available():
                    self.log_success("NCCL后端可用")
                else:
                    self.log_warning("NCCL后端不可用")
            except Exception as e:
                self.log_warning(f"NCCL检查失败: {e}")
                
            # 检查Gloo
            try:
                if hasattr(dist, 'is_gloo_available') and dist.is_gloo_available():
                    self.log_success("Gloo后端可用")
                else:
                    self.log_warning("Gloo后端不可用")
            except Exception as e:
                self.log_warning(f"Gloo检查失败: {e}")
                
            # 检查ECCL (GCU专用)
            try:
                import eccl  # type: ignore
                self.log_success("ECCL后端可用")
            except ImportError:
                self.log_warning("ECCL后端不可用")
                
        except Exception as e:
            self.log_error(f"分布式后端检查失败: {e}")
            
    def check_mmlab_ecosystem(self):
        """检查MMSegmentation生态系统"""
        print("\n🔧 检查MMSegmentation生态系统...")
        self.total_checks += 1
        
        required_packages = [
            ('mmcv', 'MMCV'),
            ('mmengine', 'MMEngine'),
            ('mmseg', 'MMSegmentation'),
        ]
        
        for package, name in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.log_success(f"{name}版本: {version}")
            except ImportError:
                self.log_error(f"{name}未安装")
            except Exception as e:
                self.log_error(f"{name}检查失败: {e}")
                
    def check_custom_modules(self):
        """检查自定义模块"""
        print("\n📦 检查自定义模块...")
        self.total_checks += 1
        
        custom_modules = [
            'mmseg_custom',
            'mmseg_custom.datasets',
            'mmseg_custom.models',
            'mmseg_custom.transforms',
        ]
        
        for module_name in custom_modules:
            try:
                importlib.import_module(module_name)
                self.log_success(f"自定义模块 {module_name} 导入成功")
            except ImportError as e:
                self.log_error(f"自定义模块 {module_name} 导入失败: {e}")
            except Exception as e:
                self.log_error(f"自定义模块 {module_name} 检查失败: {e}")
                
    def check_environment_variables(self):
        """检查环境变量"""
        print("\n🌍 检查环境变量...")
        self.total_checks += 1
        
        important_vars = [
            'PYTHONPATH',
            'CUDA_VISIBLE_DEVICES',
            'WORLD_SIZE',
            'MASTER_ADDR',
            'MASTER_PORT',
            'ECCL_BACKEND',
            'ECCL_DEVICE_TYPE',
        ]
        
        for var in important_vars:
            value = os.environ.get(var)
            if value:
                self.log_info(f"{var}={value}")
            else:
                self.log_warning(f"{var} 未设置")
                
    def check_project_structure(self):
        """检查项目结构"""
        print("\n📁 检查项目结构...")
        self.total_checks += 1
        
        required_paths = [
            'configs/',
            'mmseg_custom/',
            'scripts/',
            'start_8card_training_correct.sh',
            'requirements.txt',
        ]
        
        project_root = Path.cwd()
        for path in required_paths:
            full_path = project_root / path
            if full_path.exists():
                self.log_success(f"项目文件/目录存在: {path}")
            else:
                self.log_error(f"项目文件/目录缺失: {path}")
                
    def check_training_config(self):
        """检查训练配置"""
        print("\n⚙️  检查训练配置...")
        self.total_checks += 1
        
        config_files = [
            'configs/train_dinov3_mmrs1m_t20_gcu.py',
            'configs/final_correct_config.py',
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                self.log_success(f"配置文件存在: {config_file}")
            else:
                self.log_warning(f"配置文件不存在: {config_file}")
                
    def test_simple_training_setup(self):
        """测试简单训练设置"""
        print("\n🧪 测试简单训练设置...")
        self.total_checks += 1
        
        try:
            # 测试导入训练脚本
            sys.path.insert(0, str(Path.cwd()))
            
            # 测试基本的MMSeg导入
            from mmseg.apis import init_model
            self.log_success("MMSeg API导入成功")
            
            # 测试自定义模块注册
            from mmseg_custom import *  # noqa
            self.log_success("自定义模块注册成功")
            
        except Exception as e:
            self.log_error(f"训练设置测试失败: {e}")
            
    def run_all_checks(self):
        """运行所有检查"""
        print("🔍 开始T20 GCU环境完整验证...")
        print(f"📅 时间: {subprocess.check_output(['date'], text=True).strip()}")
        print(f"🖥️  主机: {subprocess.check_output(['hostname'], text=True).strip()}")
        print(f"📂 工作目录: {Path.cwd()}")
        
        # 执行所有检查
        self.check_python_environment()
        self.check_pytorch_installation()
        self.check_torch_gcu()
        self.check_distributed_backends()
        self.check_mmlab_ecosystem()
        self.check_custom_modules()
        self.check_environment_variables()
        self.check_project_structure()
        self.check_training_config()
        self.test_simple_training_setup()
        
        # 输出总结
        self.print_summary()
        
    def print_summary(self):
        """打印验证总结"""
        print("\n" + "="*60)
        print("📊 T20环境验证总结")
        print("="*60)
        
        print(f"✅ 成功检查: {self.success_count}/{self.total_checks}")
        
        if self.errors:
            print(f"\n❌ 错误 ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
                
        if self.warnings:
            print(f"\n⚠️  警告 ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        # 给出建议
        if not self.errors:
            print("\n🎉 环境验证通过！可以开始训练。")
            print("\n📝 建议的下一步:")
            print("  1. 运行: source setup_training_env.sh")
            print("  2. 启动训练: ./start_8card_training_correct.sh")
        else:
            print("\n🔧 需要修复错误后再开始训练。")
            print("\n📝 修复建议:")
            print("  1. 检查T20环境是否正确安装")
            print("  2. 确认torch_gcu和eccl模块可用")
            print("  3. 运行环境修复脚本")
            
        print("\n" + "="*60)
        
        return len(self.errors) == 0

def main():
    """主函数"""
    validator = T20EnvironmentValidator()
    success = validator.run_all_checks()
    
    # 返回适当的退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()