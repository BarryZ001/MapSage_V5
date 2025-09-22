#!/usr/bin/env python3
"""
修复自定义模块导入问题的脚本
解决MMSegmentation训练时自定义模块导入失败的问题
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_path():
    """检查Python路径配置"""
    print("🔍 检查Python路径配置...")
    
    current_dir = Path.cwd()
    project_root = current_dir
    
    print(f"当前工作目录: {current_dir}")
    print(f"项目根目录: {project_root}")
    
    # 检查sys.path
    print("\n当前sys.path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # 检查PYTHONPATH环境变量
    pythonpath = os.environ.get('PYTHONPATH', '')
    print(f"\nPYTHONPATH环境变量: {pythonpath}")
    
    return project_root

def check_custom_modules():
    """检查自定义模块结构"""
    print("\n🔍 检查自定义模块结构...")
    
    project_root = Path.cwd()
    custom_modules_dir = project_root / 'mmseg_custom'
    
    if not custom_modules_dir.exists():
        print(f"❌ 自定义模块目录不存在: {custom_modules_dir}")
        return False
    
    print(f"✅ 自定义模块目录存在: {custom_modules_dir}")
    
    # 检查关键文件
    key_files = [
        '__init__.py',
        'datasets/__init__.py',
        'transforms/__init__.py',
        'models/__init__.py'
    ]
    
    missing_files = []
    for file_path in key_files:
        full_path = custom_modules_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(full_path)
    
    return len(missing_files) == 0

def create_missing_init_files():
    """创建缺失的__init__.py文件"""
    print("\n🔧 创建缺失的__init__.py文件...")
    
    project_root = Path.cwd()
    custom_modules_dir = project_root / 'mmseg_custom'
    
    # 确保目录存在
    directories = [
        custom_modules_dir,
        custom_modules_dir / 'datasets',
        custom_modules_dir / 'transforms',
        custom_modules_dir / 'models'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        init_file = directory / '__init__.py'
        
        if not init_file.exists():
            # 创建基本的__init__.py文件
            if directory.name == 'mmseg_custom':
                content = '''"""
自定义MMSegmentation模块包
"""

# 导入子模块以确保注册
try:
    from . import datasets
    from . import transforms
    from . import models
except ImportError as e:
    print(f"Warning: Failed to import some custom modules: {e}")

__version__ = "1.0.0"
'''
            else:
                content = f'''"""
{directory.name} 模块
"""

# 在这里导入具体的类和函数
# 例如: from .your_module import YourClass

__all__ = []
'''
            
            init_file.write_text(content, encoding='utf-8')
            print(f"✅ 创建 {init_file}")
        else:
            print(f"✅ 已存在 {init_file}")

def fix_pythonpath():
    """修复PYTHONPATH配置"""
    print("\n🔧 修复PYTHONPATH配置...")
    
    project_root = Path.cwd()
    
    # 检查当前PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    paths_to_add = [str(project_root)]
    
    if current_pythonpath:
        existing_paths = current_pythonpath.split(':')
        paths_to_add = [p for p in paths_to_add if p not in existing_paths]
    
    if paths_to_add:
        new_pythonpath = ':'.join(paths_to_add + ([current_pythonpath] if current_pythonpath else []))
        os.environ['PYTHONPATH'] = new_pythonpath
        print(f"✅ 更新PYTHONPATH: {new_pythonpath}")
        
        # 同时更新sys.path
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
                print(f"✅ 添加到sys.path: {path}")
    else:
        print("✅ PYTHONPATH已正确配置")

def test_custom_imports():
    """测试自定义模块导入"""
    print("\n🧪 测试自定义模块导入...")
    
    test_modules = [
        'mmseg_custom',
        'mmseg_custom.datasets',
        'mmseg_custom.transforms',
        'mmseg_custom.models'
    ]
    
    success_count = 0
    for module_name in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
    
    print(f"\n导入测试结果: {success_count}/{len(test_modules)} 成功")
    return success_count == len(test_modules)

def test_mmseg_imports():
    """测试MMSegmentation相关导入"""
    print("\n🧪 测试MMSegmentation导入...")
    
    try:
        import mmseg
        print(f"✅ mmseg (版本: {mmseg.__version__})")
    except ImportError as e:
        print(f"❌ mmseg: {e}")
        return False
    
    try:
        import mmseg.models
        print("✅ mmseg.models")
    except ImportError as e:
        print(f"❌ mmseg.models: {e}")
        return False
    
    try:
        import mmseg.datasets
        print("✅ mmseg.datasets")
    except ImportError as e:
        print(f"❌ mmseg.datasets: {e}")
        return False
    
    return True

def create_setup_script():
    """创建环境设置脚本"""
    print("\n📝 创建环境设置脚本...")
    
    project_root = Path.cwd()
    setup_script = project_root / 'setup_training_env.sh'
    
    content = f'''#!/bin/bash
# 训练环境设置脚本

# 设置项目根目录
export PROJECT_ROOT="{project_root}"

# 设置PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 设置GCU相关环境变量（如果使用GCU）
export ECCL_BACKEND=gloo
export ECCL_DEVICE_TYPE=GCU
export ECCL_DEBUG=0

# 打印环境信息
echo "🚀 训练环境已设置"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

# 验证Python模块导入
echo "🧪 验证模块导入..."
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')

try:
    import mmseg_custom
    print('✅ mmseg_custom')
except ImportError as e:
    print(f'❌ mmseg_custom: {{e}}')

try:
    import mmseg
    print(f'✅ mmseg (版本: {{mmseg.__version__}})')
except ImportError as e:
    print(f'❌ mmseg: {{e}}')
"

echo "✅ 环境设置完成"
'''
    
    setup_script.write_text(content, encoding='utf-8')
    setup_script.chmod(0o755)
    print(f"✅ 创建环境设置脚本: {setup_script}")

def main():
    """主函数"""
    print("🔧 开始修复自定义模块导入问题...")
    
    # 1. 检查Python路径
    project_root = check_python_path()
    
    # 2. 检查自定义模块结构
    modules_ok = check_custom_modules()
    
    # 3. 创建缺失的__init__.py文件
    create_missing_init_files()
    
    # 4. 修复PYTHONPATH
    fix_pythonpath()
    
    # 5. 测试自定义模块导入
    custom_imports_ok = test_custom_imports()
    
    # 6. 测试MMSegmentation导入
    mmseg_imports_ok = test_mmseg_imports()
    
    # 7. 创建环境设置脚本
    create_setup_script()
    
    # 总结
    print("\n" + "="*50)
    print("🎉 修复完成！")
    print("="*50)
    
    if custom_imports_ok and mmseg_imports_ok:
        print("✅ 所有模块导入测试通过")
        print("\n📋 后续步骤:")
        print("1. 在训练前运行: source setup_training_env.sh")
        print("2. 或者在Python脚本中确保项目根目录在sys.path中")
        print("3. 重新运行训练脚本")
    else:
        print("⚠️  部分模块导入仍有问题，请检查:")
        print("1. 确保所有必要的Python包已安装")
        print("2. 检查自定义模块的具体实现")
        print("3. 验证MMSegmentation版本兼容性")
    
    return 0 if (custom_imports_ok and mmseg_imports_ok) else 1

if __name__ == '__main__':
    sys.exit(main())