#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持GCU的训练启动脚本
在训练开始前正确配置GCU设备
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入GCU设备配置
from scripts.gcu_device_setup import setup_gcu_device

def main():
    parser = argparse.ArgumentParser(description='Train with GCU support')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度')
    
    args = parser.parse_args()
    
    print("=== 启动GCU训练 ===")
    
    # 1. 配置GCU设备
    print("1. 配置GCU设备...")
    device = setup_gcu_device()
    
    if device is None:
        print("✗ GCU设备配置失败，退出训练")
        sys.exit(1)
    
    # 2. 设置训练环境变量
    print("2. 设置训练环境变量...")
    os.environ['MMENGINE_DEVICE'] = 'xla'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用CUDA
    
    # 3. 启动训练
    print("3. 启动训练...")
    
    # 构建训练命令
    train_cmd = [
        sys.executable, 
        str(project_root / 'tools' / 'train.py'),
        args.config
    ]
    
    if args.work_dir:
        train_cmd.extend(['--work-dir', args.work_dir])
    
    if args.resume:
        train_cmd.append('--resume')
    
    if args.amp:
        train_cmd.append('--amp')
    
    print(f"执行命令: {' '.join(train_cmd)}")
    
    # 执行训练
    import subprocess
    try:
        result = subprocess.run(train_cmd, check=True)
        print("✓ 训练完成")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"✗ 训练失败: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("✗ 训练被用户中断")
        return 1

if __name__ == "__main__":
    sys.exit(main())