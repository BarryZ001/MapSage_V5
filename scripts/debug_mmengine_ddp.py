#!/usr/bin/env python3
"""
MMEngine DDP设备配置调试脚本
用于在T20服务器上修改MMEngine源码，添加DDP包装前的设备诊断日志

使用方法：
1. 在T20服务器的dinov3-container容器内运行此脚本
2. 脚本会自动备份原始文件并添加调试代码
3. 运行训练后查看调试输出
4. 可选择恢复原始文件

作者: MapSage团队
日期: 2025-01-21
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# MMEngine源码路径配置
MMENGINE_RUNNER_PATH = "/usr/local/lib/python3.8/dist-packages/mmengine/runner/runner.py"
BACKUP_SUFFIX = ".debug_backup"

# 调试代码模板
DEBUG_CODE_TEMPLATE = '''
    # ===== START: MapSage DDP设备深度调试日志 =====
    print('\\n' + '='*60, flush=True)
    print('>>> MMEngine wrap_model DDP设备调试信息 <<<', flush=True)
    print('='*60, flush=True)
    
    try:
        # 检查模型参数设备分布
        param_devices = set()
        param_count = 0
        
        for name, param in model.named_parameters():
            param_devices.add(str(param.device))
            param_count += 1
            if param_count <= 5:  # 打印前5个参数的详细信息
                print(f'>>> 参数 {name}: 设备={param.device}, 形状={param.shape}', flush=True)
        
        print(f'>>> 总参数数量: {param_count}', flush=True)
        print(f'>>> 参数设备分布: {param_devices}', flush=True)
        
        # 检查模型本身的设备
        if hasattr(model, 'device'):
            print(f'>>> 模型设备属性: {model.device}', flush=True)
        
        # 检查第一个参数的设备（最常用的检查方法）
        first_param = next(model.parameters())
        print(f'>>> 第一个参数设备: {first_param.device}', flush=True)
        
        # 检查当前CUDA/GCU设备
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                print(f'>>> 当前CUDA设备: {torch.cuda.current_device()}', flush=True)
        except:
            pass
            
        try:
            import torch_gcu
            if torch_gcu.is_available():
                print(f'>>> 当前GCU设备: {torch_gcu.current_device()}', flush=True)
        except:
            pass
        
        # 检查环境变量
        local_rank = os.environ.get('LOCAL_RANK', 'None')
        rank = os.environ.get('RANK', 'None')
        world_size = os.environ.get('WORLD_SIZE', 'None')
        print(f'>>> 分布式环境: LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}', flush=True)
        
        # 警告检查
        if any('cpu' in device for device in param_devices):
            print('🚨 警告: 检测到模型参数仍在CPU上!', flush=True)
            print('🚨 这将导致DDP设备不匹配错误!', flush=True)
        else:
            print('✅ 模型参数已正确移动到加速器设备', flush=True)
            
    except Exception as e:
        print(f'>>> DDP包装前设备检查失败: {e}', flush=True)
        import traceback
        traceback.print_exc()
    
    print('='*60, flush=True)
    print('>>> DDP设备调试信息结束 <<<', flush=True)
    print('='*60 + '\\n', flush=True)
    # ===== END: MapSage DDP设备深度调试日志 =====
'''

class MMEngineDebugger:
    """MMEngine DDP调试器"""
    
    def __init__(self):
        self.runner_path = Path(MMENGINE_RUNNER_PATH)
        self.backup_path = Path(str(self.runner_path) + BACKUP_SUFFIX)
        
    def check_environment(self):
        """检查运行环境"""
        print("🔍 检查T20服务器环境...")
        
        # 检查是否在容器内
        if not os.path.exists("/.dockerenv"):
            print("⚠️ 警告: 似乎不在Docker容器内")
        
        # 检查MMEngine文件是否存在
        if not self.runner_path.exists():
            print(f"❌ 错误: MMEngine runner.py文件不存在: {self.runner_path}")
            return False
            
        # 检查文件权限
        if not os.access(self.runner_path, os.R_OK | os.W_OK):
            print(f"❌ 错误: 没有读写权限: {self.runner_path}")
            return False
            
        print("✅ 环境检查通过")
        return True
    
    def backup_original_file(self):
        """备份原始文件"""
        if self.backup_path.exists():
            print(f"📁 备份文件已存在: {self.backup_path}")
            return True
            
        try:
            shutil.copy2(self.runner_path, self.backup_path)
            print(f"✅ 已备份原始文件到: {self.backup_path}")
            return True
        except Exception as e:
            print(f"❌ 备份失败: {e}")
            return False
    
    def find_wrap_model_function(self, content):
        """查找wrap_model函数和DDP包装位置"""
        lines = content.split('\n')
        
        # 查找wrap_model函数
        wrap_model_start = -1
        for i, line in enumerate(lines):
            if 'def wrap_model(' in line or 'def wrap_model (' in line:
                wrap_model_start = i
                break
        
        if wrap_model_start == -1:
            print("❌ 未找到wrap_model函数")
            return None, None
        
        print(f"✅ 找到wrap_model函数，起始行: {wrap_model_start + 1}")
        
        # 在wrap_model函数内查找DDP包装代码
        ddp_line = -1
        for i in range(wrap_model_start, len(lines)):
            line = lines[i].strip()
            if ('MMDistributedDataParallel(' in line or 
                'DistributedDataParallel(' in line or
                'DDP(' in line):
                ddp_line = i
                break
            # 如果遇到下一个函数定义，停止搜索
            if i > wrap_model_start and line.startswith('def '):
                break
        
        if ddp_line == -1:
            print("❌ 在wrap_model函数中未找到DDP包装代码")
            return wrap_model_start, None
        
        print(f"✅ 找到DDP包装代码，行号: {ddp_line + 1}")
        return wrap_model_start, ddp_line
    
    def add_debug_code(self):
        """添加调试代码"""
        try:
            # 读取原始文件
            with open(self.runner_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找插入位置
            wrap_start, ddp_line = self.find_wrap_model_function(content)
            if ddp_line is None:
                return False
            
            lines = content.split('\n')
            
            # 检查是否已经添加了调试代码
            if 'MapSage DDP设备深度调试日志' in content:
                print("⚠️ 调试代码已存在，跳过添加")
                return True
            
            # 获取DDP行的缩进
            ddp_line_content = lines[ddp_line]
            indent = len(ddp_line_content) - len(ddp_line_content.lstrip())
            
            # 准备调试代码（添加适当缩进）
            debug_lines = []
            for line in DEBUG_CODE_TEMPLATE.strip().split('\n'):
                if line.strip():
                    debug_lines.append(' ' * indent + line)
                else:
                    debug_lines.append('')
            
            # 在DDP包装前插入调试代码
            lines.insert(ddp_line, '\n'.join(debug_lines))
            
            # 写入修改后的文件
            modified_content = '\n'.join(lines)
            with open(self.runner_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✅ 已在第{ddp_line + 1}行前添加调试代码")
            return True
            
        except Exception as e:
            print(f"❌ 添加调试代码失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def restore_original_file(self):
        """恢复原始文件"""
        if not self.backup_path.exists():
            print("❌ 备份文件不存在，无法恢复")
            return False
        
        try:
            shutil.copy2(self.backup_path, self.runner_path)
            print("✅ 已恢复原始文件")
            return True
        except Exception as e:
            print(f"❌ 恢复失败: {e}")
            return False
    
    def show_usage_instructions(self):
        """显示使用说明"""
        print("\n" + "="*60)
        print("🎯 MMEngine DDP调试代码已添加完成!")
        print("="*60)
        print("\n📋 下一步操作指南:")
        print("1. 运行您的8卡分布式训练命令:")
        print("   cd /workspace/code/MapSage_V5")
        print("   bash scripts/start_8card_training.sh")
        print("\n2. 观察输出中的调试信息:")
        print("   - 查找以 '>>> MMEngine wrap_model DDP设备调试信息 <<<' 开头的日志")
        print("   - 重点关注 '参数设备分布' 和 '第一个参数设备' 信息")
        print("   - 如果看到 '🚨 警告: 检测到模型参数仍在CPU上!'，这就是问题根源")
        print("\n3. 将完整的调试输出发送给开发团队进行分析")
        print("\n4. 调试完成后，可运行以下命令恢复原始文件:")
        print(f"   python3 {__file__} --restore")
        print("\n" + "="*60)

def main():
    """主函数"""
    debugger = MMEngineDebugger()
    
    # 解析命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--restore':
        print("🔄 恢复原始MMEngine文件...")
        if debugger.restore_original_file():
            print("✅ 恢复完成")
        else:
            print("❌ 恢复失败")
        return
    
    print("🚀 MMEngine DDP设备配置调试脚本")
    print("="*50)
    
    # 检查环境
    if not debugger.check_environment():
        sys.exit(1)
    
    # 备份原始文件
    if not debugger.backup_original_file():
        sys.exit(1)
    
    # 添加调试代码
    if not debugger.add_debug_code():
        print("❌ 添加调试代码失败")
        sys.exit(1)
    
    # 显示使用说明
    debugger.show_usage_instructions()

if __name__ == "__main__":
    main()