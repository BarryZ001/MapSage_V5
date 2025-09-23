#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECCL环境修复脚本
解决ECCL库路径和环境变量配置问题
"""

import os
import subprocess
import sys

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")

def check_eccl_files():
    """检查ECCL文件位置"""
    print_section("检查ECCL文件位置")
    
    eccl_locations = {
        'header': '/usr/include/eccl/eccl.h',
        'library': '/usr/lib/libeccl.so',
        'tops_header': '/opt/tops/include/tops/tops_eccl_ext.h'
    }
    
    found_files = {}
    for name, path in eccl_locations.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
            found_files[name] = path
        else:
            print(f"❌ {name}: {path} (不存在)")
    
    return found_files

def update_ldconfig():
    """更新ldconfig配置"""
    print_section("更新ldconfig配置")
    
    # 检查/usr/lib是否在ldconfig中
    try:
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
        if 'libeccl.so' in result.stdout:
            print("✅ libeccl.so已在ldconfig中")
            return True
        else:
            print("❌ libeccl.so不在ldconfig中")
    except Exception as e:
        print(f"⚠️ 检查ldconfig失败: {e}")
    
    # 尝试更新ldconfig
    try:
        print("🔄 更新ldconfig...")
        subprocess.run(['ldconfig'], check=True)
        
        # 再次检查
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
        if 'libeccl.so' in result.stdout:
            print("✅ ldconfig更新成功，libeccl.so现在可用")
            return True
        else:
            print("❌ ldconfig更新后仍未找到libeccl.so")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 更新ldconfig失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️ 更新ldconfig异常: {e}")
        return False

def generate_env_script():
    """生成环境变量设置脚本"""
    print_section("生成环境变量设置脚本")
    
    env_script = """#!/bin/bash
# ECCL环境变量设置脚本
# 使用方法: source scripts/setup_eccl_env.sh

echo "🔧 设置ECCL环境变量..."

# 设置ECCL根目录
export ECCL_ROOT=/usr
export TOPS_ECCL_ROOT=/opt/tops

# 设置库路径
export LD_LIBRARY_PATH=/usr/lib:/opt/tops/lib:$LD_LIBRARY_PATH

# 设置包含路径
export CPATH=/usr/include/eccl:/opt/tops/include:$CPATH

# 设置PKG_CONFIG_PATH（如果存在）
if [ -d "/usr/lib/pkgconfig" ]; then
    export PKG_CONFIG_PATH=/usr/lib/pkgconfig:$PKG_CONFIG_PATH
fi

# 验证设置
echo "✅ ECCL_ROOT: $ECCL_ROOT"
echo "✅ TOPS_ECCL_ROOT: $TOPS_ECCL_ROOT"
echo "✅ LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 检查库是否可用
if ldconfig -p | grep -q libeccl; then
    echo "✅ libeccl.so在系统库路径中"
else
    echo "⚠️ libeccl.so不在系统库路径中，可能需要运行: sudo ldconfig"
fi

# 检查文件是否存在
if [ -f "/usr/lib/libeccl.so" ]; then
    echo "✅ ECCL库文件存在: /usr/lib/libeccl.so"
else
    echo "❌ ECCL库文件不存在: /usr/lib/libeccl.so"
fi

if [ -f "/usr/include/eccl/eccl.h" ]; then
    echo "✅ ECCL头文件存在: /usr/include/eccl/eccl.h"
else
    echo "❌ ECCL头文件不存在: /usr/include/eccl/eccl.h"
fi

echo "🎯 ECCL环境设置完成！"
"""
    
    script_path = "/workspace/code/MapSage_V5/scripts/setup_eccl_env.sh"
    try:
        with open(script_path, 'w') as f:
            f.write(env_script)
        
        # 设置执行权限
        os.chmod(script_path, 0o755)
        print(f"✅ 环境脚本已生成: {script_path}")
        return script_path
        
    except Exception as e:
        print(f"❌ 生成环境脚本失败: {e}")
        return None

def test_eccl_import():
    """测试ECCL相关导入"""
    print_section("测试ECCL相关导入")
    
    # 设置环境变量
    os.environ['ECCL_ROOT'] = '/usr'
    os.environ['TOPS_ECCL_ROOT'] = '/opt/tops'
    
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = '/usr/lib:/opt/tops/lib'
    if current_ld_path:
        new_ld_path = f"{new_ld_path}:{current_ld_path}"
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    print(f"设置环境变量:")
    print(f"  ECCL_ROOT: {os.environ['ECCL_ROOT']}")
    print(f"  TOPS_ECCL_ROOT: {os.environ['TOPS_ECCL_ROOT']}")
    print(f"  LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    
    # 测试torch导入
    try:
        import torch
        print("✅ torch导入成功")
        
        # 测试torch_gcu导入
        try:
            import torch_gcu  # type: ignore
            print("✅ torch_gcu导入成功")
            
            # 测试分布式后端
            import torch.distributed as dist
            
            backends = ['gloo', 'nccl', 'mpi', 'eccl']
            available_backends = []
            
            for backend in backends:
                try:
                    if hasattr(dist, 'is_backend_available') and dist.is_backend_available(backend):  # type: ignore
                        available_backends.append(backend)
                        print(f"✅ {backend}: 可用")
                    else:
                        print(f"❌ {backend}: 不可用")
                except Exception as e:
                    print(f"⚠️ {backend}: 检查失败 - {e}")
            
            return available_backends
            
        except ImportError as e:
            print(f"❌ torch_gcu导入失败: {e}")
            return []
            
    except ImportError as e:
        print(f"❌ torch导入失败: {e}")
        return []

def create_test_script():
    """创建ECCL测试脚本"""
    print_section("创建ECCL测试脚本")
    
    test_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
ECCL后端测试脚本
在设置环境变量后测试ECCL后端
\"\"\"

import os
import sys
import torch
import torch.distributed as dist

def test_eccl_backend():
    \"\"\"测试ECCL后端初始化\"\"\"
    print("🧪 测试ECCL后端...")
    
    # 设置分布式参数
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29502')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
        
        print("🔧 尝试初始化ECCL后端...")
        dist.init_process_group(
            backend='eccl',
            init_method='tcp://127.0.0.1:29502',
            world_size=1,
            rank=0
        )
        
        print("✅ ECCL后端初始化成功！")
        
        # 测试基本操作
        tensor = torch.tensor([1.0])
        dist.all_reduce(tensor)
        print(f"✅ all_reduce测试成功: {tensor.item()}")
        
        # 清理
        dist.destroy_process_group()
        return True
        
    except Exception as e:
        print(f"❌ ECCL后端测试失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        return False

if __name__ == '__main__':
    success = test_eccl_backend()
    sys.exit(0 if success else 1)
"""
    
    script_path = "/workspace/code/MapSage_V5/scripts/test_eccl_backend.py"
    try:
        with open(script_path, 'w') as f:
            f.write(test_script)
        
        os.chmod(script_path, 0o755)
        print(f"✅ 测试脚本已生成: {script_path}")
        return script_path
        
    except Exception as e:
        print(f"❌ 生成测试脚本失败: {e}")
        return None

def main():
    """主函数"""
    print("🔧 ECCL环境修复工具")
    print("=" * 60)
    
    try:
        # 检查文件
        found_files = check_eccl_files()
        
        if not found_files:
            print("❌ 未找到ECCL文件，请确认ECCL已正确安装")
            return False
        
        # 更新ldconfig
        ldconfig_success = update_ldconfig()
        
        # 生成环境脚本
        env_script = generate_env_script()
        
        # 创建测试脚本
        test_script = create_test_script()
        
        # 测试导入
        available_backends = test_eccl_import()
        
        print_section("修复结果总结")
        
        if 'eccl' in available_backends:
            print("🎉 ECCL后端现在可用！")
            print("\n📋 使用说明:")
            print("1. 在新的shell中运行: source scripts/setup_eccl_env.sh")
            print("2. 然后测试: python scripts/test_eccl_backend.py")
            print("3. 或者直接运行分布式训练")
        else:
            print("⚠️ ECCL后端仍不可用，可能需要:")
            print("1. 重启shell或容器")
            print("2. 运行: sudo ldconfig")
            print("3. 检查torch_gcu版本兼容性")
            print("4. 联系燧原技术支持")
        
        print(f"\n📁 生成的文件:")
        if env_script:
            print(f"  - 环境设置脚本: {env_script}")
        if test_script:
            print(f"  - 测试脚本: {test_script}")
        
        return 'eccl' in available_backends
        
    except Exception as e:
        print(f"❌ 修复过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)