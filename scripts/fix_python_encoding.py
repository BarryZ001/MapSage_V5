#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复Python脚本编码声明问题

为所有缺少编码声明的Python脚本添加UTF-8编码头。
"""

import os
import re
from pathlib import Path

def fix_python_encoding(file_path: Path) -> bool:
    """为Python文件添加UTF-8编码声明"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 如果UTF-8读取失败，尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            print(f"Warning: Cannot read {file_path}")
            return False
    
    lines = content.split('\n')
    
    # 检查是否已有编码声明
    has_encoding = False
    for i, line in enumerate(lines[:3]):  # 检查前3行
        if re.search(r'coding[:=]\s*([-\w.]+)', line):
            has_encoding = True
            break
    
    if has_encoding:
        print(f"Skip {file_path}: already has encoding declaration")
        return True
    
    # 添加编码声明
    if lines[0].startswith('#!'):
        # 如果有shebang，在第二行添加编码声明
        lines.insert(1, '# -*- coding: utf-8 -*-')
    else:
        # 否则在第一行添加编码声明
        lines.insert(0, '# -*- coding: utf-8 -*-')
    
    # 写回文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Fixed {file_path}")
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def main():
    """主函数"""
    scripts_dir = Path(__file__).parent
    python_files = list(scripts_dir.glob('*.py'))
    
    print(f"Found {len(python_files)} Python files in scripts directory")
    
    fixed_count = 0
    for py_file in python_files:
        if fix_python_encoding(py_file):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()