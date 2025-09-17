#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
燧原T20适配脚本
自动将项目中的CUDA相关代码适配为燧原T20设备

使用方法:
python scripts/adapt_to_enflame_t20.py --device_name gcu --api_name torch.gcu
"""

import os
import re
import argparse
from pathlib import Path

def adapt_file_for_enflame(file_path, dry_run=True):
    """
    适配单个文件中的CUDA相关代码为燧原T20设备
    基于T20集群环境配置手册的实际配置
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 替换规则 - 基于T20实际环境
        replacements = [
            # 设备相关 - 使用ptex.device('xla')
            (r"device\s*=\s*['\"]cuda['\"]?", "device = ptex.device('xla')"),
            (r"\.cuda\(\)", ".to(ptex.device('xla'))"),
            (r"\.to\(['\"]cuda['\"]\)", ".to(ptex.device('xla'))"),
            (r"torch\.device\(['\"]cuda['\"]\)", "ptex.device('xla')"),
            
            # CUDA函数替换为ptex.tops
            (r"torch\.cuda\.is_available\(\)", "True  # ptex设备默认可用"),
            (r"torch\.cuda\.device_count\(\)", "1  # T20设备数量"),
            (r"torch\.cuda\.manual_seed_all\(", "ptex.tops.manual_seed_all("),
            (r"torch\.cuda\.empty_cache\(\)", "# ptex.tops.empty_cache()  # 如果需要"),
            (r"torch\.cuda\.set_device\(", "# ptex.tops.set_device(  # 如果需要"),
            
            # 导入相关 - 添加ptex导入
            (r"^import torch$", "import torch\nimport ptex"),
            (r"^import torch\n", "import torch\nimport ptex\n"),
        ]
        
        changes_made = []
        for pattern, replacement in replacements:
            if re.search(pattern, content, re.MULTILINE):
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                changes_made.append(f"替换: {pattern} -> {replacement}")
        
        if content != original_content:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 已修改: {file_path}")
            else:
                print(f"🔍 需要修改: {file_path}")
            
            for change in changes_made:
                print(f"   - {change}")
            return True
        else:
            print(f"⏭️  无需修改: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ 处理文件失败 {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='适配燧原T20设备')
    parser.add_argument('--device_name', default='gcu', help='设备名称 (默认: gcu)')
    parser.add_argument('--api_name', default='torch.gcu', help='API名称 (默认: torch.gcu)')
    parser.add_argument('--dry_run', action='store_true', help='仅显示需要修改的文件，不实际修改')
    
    args = parser.parse_args()
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 需要适配的文件列表
    files_to_adapt = [
        'run_staged_distillation_experiment.py',
        'run_task_oriented_distillation.py', 
        'run_improved_distillation_experiment.py',
        'configs/train_distill_dinov3_v2_improved.py',
        'scripts/setup_distillation_environment.py',
        'scripts/validate_tta.py',
        'app.py'
    ]
    
    print(f"🚀 开始适配燧原T20设备")
    print(f"设备名称: {args.device_name}")
    print(f"API名称: {args.api_name}")
    print(f"项目根目录: {project_root}")
    print("-" * 50)
    
    modified_files = []
    
    for file_rel_path in files_to_adapt:
        file_path = project_root / file_rel_path
        
        if not file_path.exists():
            print(f"⚠️  文件不存在: {file_path}")
            continue
        
        print(f"🔍 检查文件: {file_rel_path}")
        
        if args.dry_run:
            # 干运行模式：只检查是否需要修改
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if any(pattern in content for pattern in ['torch.cuda', "'cuda'", '"cuda"', '.cuda()']):
                    print(f"  📝 需要修改")
                    modified_files.append(file_rel_path)
                else:
                    print(f"  ✅ 无需修改")
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        else:
            # 实际修改模式
            if adapt_file_for_enflame(file_path, args.dry_run):
                print(f"  ✅ 已修改")
                modified_files.append(file_rel_path)
            else:
                print(f"  ➡️  无需修改")
    
    print("-" * 50)
    
    if args.dry_run:
        print(f"📋 干运行完成，发现 {len(modified_files)} 个文件需要修改:")
        for file_path in modified_files:
            print(f"  - {file_path}")
        print("\n💡 运行 'python scripts/adapt_to_enflame_t20.py' 进行实际修改")
    else:
        print(f"🎉 适配完成！共修改了 {len(modified_files)} 个文件:")
        for file_path in modified_files:
            print(f"  - {file_path}")
        
        if modified_files:
            print("\n📝 接下来的步骤:")
            print("1. 确认权重文件路径正确")
            print("2. 运行验证脚本: python scripts/validate_tta.py")
            print("3. 检查mIoU是否达到84.96%左右")
        else:
            print("\n✨ 所有文件都已是最新状态，无需修改")

if __name__ == '__main__':
    main()