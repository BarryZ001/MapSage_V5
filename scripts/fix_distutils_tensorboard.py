#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复distutils.version兼容性问题的补丁脚本
解决PyTorch TensorBoard在Python 3.10+中的distutils.version错误
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_packaging():
    """安装packaging包"""
    try:
        import packaging
        print("✅ packaging包已安装")
        return True
    except ImportError:
        print("📦 正在安装packaging包...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging'])
            print("✅ packaging包安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ packaging包安装失败: {e}")
            return False

def find_tensorboard_init():
    """查找TensorBoard的__init__.py文件"""
    import torch
    torch_path = Path(torch.__file__).parent
    tensorboard_init = torch_path / "utils" / "tensorboard" / "__init__.py"
    
    if tensorboard_init.exists():
        return tensorboard_init
    
    # 尝试其他可能的路径
    possible_paths = [
        "/usr/local/lib/python3.8/dist-packages/torch/utils/tensorboard/__init__.py",
        "/usr/local/lib/python3.9/dist-packages/torch/utils/tensorboard/__init__.py",
        "/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/__init__.py",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return Path(path)
    
    return None

def patch_tensorboard_init(tensorboard_init_path):
    """修补TensorBoard的__init__.py文件"""
    print(f"🔧 正在修补文件: {tensorboard_init_path}")
    
    # 备份原文件
    backup_path = tensorboard_init_path.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy2(tensorboard_init_path, backup_path)
        print(f"📋 已备份原文件到: {backup_path}")
    
    # 读取文件内容
    with open(tensorboard_init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修补过
    if 'from packaging.version import Version as LooseVersion' in content:
        print("✅ 文件已经修补过，无需重复修补")
        return True
    
    # 执行修补
    old_import = 'LooseVersion = distutils.version.LooseVersion'
    new_import = '''try:
    from packaging.version import Version as LooseVersion
except ImportError:
    try:
        from distutils.version import LooseVersion
    except ImportError:
        # 如果都不可用，创建一个简单的版本比较类
        class LooseVersion:
            def __init__(self, version):
                self.version = str(version)
            def __lt__(self, other):
                return self.version < str(other)
            def __le__(self, other):
                return self.version <= str(other)
            def __gt__(self, other):
                return self.version > str(other)
            def __ge__(self, other):
                return self.version >= str(other)
            def __eq__(self, other):
                return self.version == str(other)
            def __ne__(self, other):
                return self.version != str(other)'''
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        
        # 写回文件
        with open(tensorboard_init_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ TensorBoard __init__.py 修补完成")
        return True
    else:
        print("⚠️ 未找到需要修补的代码行")
        return False

def main():
    """主函数"""
    print("🔧 开始修复distutils.version兼容性问题...")
    
    # 1. 安装packaging包
    if not install_packaging():
        print("❌ 无法安装packaging包，修复失败")
        return False
    
    # 2. 查找TensorBoard的__init__.py文件
    tensorboard_init = find_tensorboard_init()
    if not tensorboard_init:
        print("❌ 未找到TensorBoard的__init__.py文件")
        return False
    
    print(f"📍 找到TensorBoard文件: {tensorboard_init}")
    
    # 3. 修补文件
    if patch_tensorboard_init(tensorboard_init):
        print("✅ distutils.version兼容性问题修复完成")
        print("🚀 现在可以重新运行训练脚本")
        return True
    else:
        print("❌ 修补失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)