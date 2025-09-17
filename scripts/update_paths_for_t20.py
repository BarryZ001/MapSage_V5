#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新配置文件路径 - 适配T20服务器环境
将Kaggle路径替换为T20服务器路径
"""

import os
import re
from pathlib import Path

def update_paths_in_file(file_path, dry_run=True):
    """
    更新单个文件中的路径配置
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 路径替换规则
        path_replacements = [
            # LoveDA数据集路径
            (r"/kaggle/input/loveda", "/workspace/data/loveda"),
            
            # EarthVQA数据集路径
            (r"/kaggle/input/2024earthvqa/2024EarthVQA", "/workspace/data/EarthVQA"),
            
            # 权重文件路径
            (r"/kaggle/input/mapsage-stage02-checkpoint-6000/best_mIoU_iter_6000\.pth", "/workspace/weights/best_mIoU_iter_6000.pth"),
            (r"/kaggle/input/temp-8000tier/iter_8000\.pth", "/workspace/weights/iter_8000.pth"),
            
            # 预训练权重路径
            (r"/kaggle/input/mit-b2-imagenet-weights/mit-b2_in1k-20230209-4d95315b\.pth", "/workspace/weights/mit-b2_in1k-20230209-4d95315b.pth"),
            (r"/kaggle/input/dinov3-vitl16-pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff\.pth", "/workspace/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
            (r"/kaggle/input/dinov3-sat-weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff\.pth", "/workspace/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
        ]
        
        changes_made = []
        for pattern, replacement in path_replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"替换: {pattern} -> {replacement}")
        
        if content != original_content:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 已更新: {file_path}")
            else:
                print(f"🔍 需要更新: {file_path}")
            
            for change in changes_made:
                print(f"   - {change}")
            return True
        else:
            print(f"⏭️  无需更新: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ 处理文件失败 {file_path}: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='批量更新T20服务器路径配置')
    parser.add_argument('--dry-run', action='store_true', help='干运行模式，只显示将要进行的更改')
    parser.add_argument('--configs-only', action='store_true', help='只更新configs目录')
    parser.add_argument('--scripts-only', action='store_true', help='只更新scripts目录')
    
    args = parser.parse_args()
    
    print("🚀 开始批量更新T20服务器路径配置...")
    print("=" * 50)
    
    if args.dry_run:
        print("📋 干运行模式：只显示将要进行的更改")
    else:
        print("⚠️  实际修改模式：将直接修改文件")
    
    print()
    
    # 确定要处理的目录
    project_root = Path(__file__).parent.parent
    directories_to_process = []
    
    if args.configs_only:
        directories_to_process = [project_root / 'configs']
    elif args.scripts_only:
        directories_to_process = [project_root / 'scripts']
    else:
        directories_to_process = [project_root / 'configs', project_root / 'scripts']
    
    modified_files = []
    
    for directory in directories_to_process:
        if not directory.exists():
            print(f"⚠️  目录不存在: {directory}")
            continue
            
        print(f"📁 处理目录: {directory}")
        
        # 处理Python文件
        for py_file in directory.rglob('*.py'):
            if update_paths_in_file(py_file, args.dry_run):
                modified_files.append(str(py_file.relative_to(project_root)))
        
        # 处理Markdown文件
        for md_file in directory.rglob('*.md'):
            if update_paths_in_file(md_file, args.dry_run):
                modified_files.append(str(md_file.relative_to(project_root)))
        
        print()
    
    # 总结
    print("📊 更新总结")
    print("=" * 30)
    
    if modified_files:
        print(f"✅ 共处理了 {len(modified_files)} 个文件:")
        for file_path in modified_files:
            print(f"   - {file_path}")
    else:
        print("ℹ️  没有文件需要更新")
    
    if args.dry_run and modified_files:
        print("\n🚀 要执行实际更新，请运行:")
        if args.configs_only:
            print("   python scripts/update_paths_for_t20.py --configs-only")
        elif args.scripts_only:
            print("   python scripts/update_paths_for_t20.py --scripts-only")
        else:
            print("   python scripts/update_paths_for_t20.py")
    
    print("\n🎯 T20服务器路径配置:")
    print("   - 数据集: /workspace/data/")
    print("   - 权重: /workspace/weights/")
    print("   - 输出: /workspace/outputs/")
    print("   - 代码: /workspace/code/")

if __name__ == '__main__':
    main()