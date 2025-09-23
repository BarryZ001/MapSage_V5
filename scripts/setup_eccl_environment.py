#!/usr/bin/env python3
"""
ECCL环境设置脚本
基于已安装的ECCL 2.5.136版本进行环境配置
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def print_header(title):
    """打印标题"""
    print(f"\n🔧 {title}")
    print("=" * 50)

def check_container_environment():
    """检查是否在容器环境中"""
    print_header("环境检测")
    
    if os.path.exists('/.dockerenv'):
        print("✅ 检测到Docker容器环境")
        return True
    else:
        print("⚠️ 未检测到容器环境")
        return False

def verify_eccl_installation():
    """验证ECCL安装"""
    print_header("验证ECCL安装")
    
    # 检查关键文件
    eccl_files = {
        'library': '/usr/lib/libeccl.so',
        'header': '/usr/include/eccl/eccl.h',
        'all_reduce_perf': '/usr/local/bin/eccl_all_reduce_perf',
        'all_gather_perf': '/usr/local/bin/eccl_all_gather_perf',
        'broadcast_perf': '/usr/local/bin/eccl_broadcast_perf'
    }
    
    found_files = {}
    for name, path in eccl_files.items():
        if os.path.exists(path):
            print(f"✅ 找到 {name}: {path}")
            found_files[name] = path
        else:
            print(f"❌ 未找到 {name}: {path}")
    
    return found_files

def create_environment_config():
    """创建环境配置"""
    print_header("创建环境配置")
    
    # 环境变量配置
    env_config = """#!/bin/bash
# ECCL环境配置脚本
# 基于ECCL 2.5.136版本

echo "🚀 配置ECCL环境变量..."

# ECCL核心配置
export ECCL_DEBUG=0
export ECCL_LOG_LEVEL=INFO
export ECCL_SOCKET_IFNAME=eth0
export ECCL_IB_DISABLE=1
export ECCL_ASYNC_DISABLE=false
export ECCL_MAX_NCHANNELS=2

# 库路径配置
export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH}"

# 工具路径配置  
export PATH="/usr/local/bin:${PATH}"

# GCU设备配置
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=""

# 分布式训练配置
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0

echo "✅ ECCL环境变量配置完成"
echo "📋 关键配置:"
echo "   - ECCL库路径: /usr/lib/libeccl.so"
echo "   - ECCL头文件: /usr/include/eccl/eccl.h"
echo "   - 性能工具: /usr/local/bin/eccl_*_perf"
"""
    
    # 写入配置文件
    config_path = '/tmp/eccl_env_setup.sh'
    with open(config_path, 'w') as f:
        f.write(env_config)
    
    os.chmod(config_path, 0o755)
    print(f"✅ 环境配置脚本已创建: {config_path}")
    
    return config_path

def create_python_test_script():
    """创建Python测试脚本"""
    print_header("创建Python测试脚本")
    
    test_script = '''#!/usr/bin/env python3
"""
ECCL Python功能测试脚本
"""

import os
import sys
import subprocess

def test_environment_variables():
    """测试环境变量"""
    print("🌍 检查环境变量:")
    
    eccl_vars = [
        'ECCL_DEBUG', 'ECCL_LOG_LEVEL', 'ECCL_SOCKET_IFNAME',
        'LD_LIBRARY_PATH', 'TOPS_VISIBLE_DEVICES'
    ]
    
    for var in eccl_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")

def test_library_loading():
    """测试库加载"""
    print("\\n📚 测试库加载:")
    
    try:
        # 尝试使用ctypes加载ECCL库
        import ctypes
        lib_path = '/usr/lib/libeccl.so'
        if os.path.exists(lib_path):
            lib = ctypes.CDLL(lib_path)
            print(f"✅ 成功加载ECCL库: {lib_path}")
            return True
        else:
            print(f"❌ ECCL库文件不存在: {lib_path}")
            return False
    except Exception as e:
        print(f"❌ 加载ECCL库失败: {e}")
        return False

def test_performance_tools():
    """测试性能工具"""
    print("\\n🔧 测试性能工具:")
    
    tools = [
        'eccl_all_reduce_perf',
        'eccl_all_gather_perf', 
        'eccl_broadcast_perf'
    ]
    
    available_tools = []
    for tool in tools:
        tool_path = f'/usr/local/bin/{tool}'
        if os.path.exists(tool_path):
            print(f"✅ 找到工具: {tool}")
            available_tools.append(tool)
        else:
            print(f"❌ 未找到工具: {tool}")
    
    return available_tools

def test_torch_distributed():
    """测试PyTorch分布式功能"""
    print("\\n🔥 测试PyTorch分布式:")
    
    try:
        import torch
        import torch.distributed as dist
        
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ 分布式可用: {dist.is_available()}")
        
        # 检查可用后端
        backends = []
        for backend in ['gloo', 'nccl', 'mpi']:
            if dist.is_backend_available(backend):
                backends.append(backend)
        
        print(f"✅ 可用后端: {backends}")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 分布式检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 ECCL功能测试")
    print("=" * 40)
    
    # 运行所有测试
    test_environment_variables()
    lib_ok = test_library_loading()
    tools = test_performance_tools()
    torch_ok = test_torch_distributed()
    
    # 总结
    print("\\n📊 测试结果总结:")
    print("=" * 40)
    
    if lib_ok:
        print("✅ ECCL库加载成功")
    else:
        print("❌ ECCL库加载失败")
    
    if tools:
        print(f"✅ 找到 {len(tools)} 个性能工具")
    else:
        print("❌ 未找到性能工具")
    
    if torch_ok:
        print("✅ PyTorch分布式功能可用")
    else:
        print("❌ PyTorch分布式功能不可用")
    
    # 给出建议
    print("\\n💡 建议:")
    if lib_ok and torch_ok:
        print("✅ 环境配置良好，可以进行分布式训练")
        print("🚀 下一步: 运行 train_distributed_gcu_robust.py 进行测试")
    else:
        print("⚠️ 环境配置需要调整")
        print("📝 请确保已正确配置环境变量")

if __name__ == "__main__":
    main()
'''
    
    # 写入测试脚本
    test_path = '/tmp/test_eccl_functionality.py'
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    os.chmod(test_path, 0o755)
    print(f"✅ Python测试脚本已创建: {test_path}")
    
    return test_path

def create_distributed_test():
    """创建分布式训练测试脚本"""
    print_header("创建分布式训练测试")
    
    test_script = '''#!/usr/bin/env python3
"""
简单的分布式训练测试脚本
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # 初始化分布式进程组
    dist.init_process_group(
        backend='gloo',  # 使用gloo后端
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.default_pg_timeout
    )

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def run_distributed_test(rank, world_size):
    """运行分布式测试"""
    print(f"🚀 启动进程 {rank}/{world_size}")
    
    try:
        # 设置分布式
        setup_distributed(rank, world_size)
        
        # 创建测试张量
        tensor = torch.ones(2, 2) * rank
        print(f"进程 {rank} 初始张量:\\n{tensor}")
        
        # 执行all_reduce操作
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"进程 {rank} all_reduce后:\\n{tensor}")
        
        # 执行broadcast操作
        if rank == 0:
            broadcast_tensor = torch.tensor([1.0, 2.0, 3.0])
        else:
            broadcast_tensor = torch.zeros(3)
        
        dist.broadcast(broadcast_tensor, src=0)
        print(f"进程 {rank} broadcast后: {broadcast_tensor}")
        
        print(f"✅ 进程 {rank} 测试完成")
        
    except Exception as e:
        print(f"❌ 进程 {rank} 测试失败: {e}")
    finally:
        cleanup()

def main():
    """主函数"""
    print("🧪 分布式训练测试")
    print("=" * 30)
    
    world_size = 2  # 使用2个进程进行测试
    
    try:
        # 启动多进程
        mp.spawn(
            run_distributed_test,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("✅ 分布式测试完成")
    except Exception as e:
        print(f"❌ 分布式测试失败: {e}")

if __name__ == "__main__":
    main()
'''
    
    # 写入测试脚本
    test_path = '/tmp/test_distributed_training.py'
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    os.chmod(test_path, 0o755)
    print(f"✅ 分布式测试脚本已创建: {test_path}")
    
    return test_path

def main():
    """主函数"""
    print("🚀 ECCL环境设置脚本")
    print("基于ECCL 2.5.136版本")
    print("=" * 50)
    
    # 1. 检查环境
    is_container = check_container_environment()
    
    # 2. 验证ECCL安装
    found_files = verify_eccl_installation()
    
    if not found_files:
        print("❌ 未找到ECCL文件，请确认ECCL已正确安装")
        sys.exit(1)
    
    # 3. 创建配置文件
    config_path = create_environment_config()
    
    # 4. 创建测试脚本
    test_path = create_python_test_script()
    distributed_test_path = create_distributed_test()
    
    # 5. 输出使用说明
    print_header("使用说明")
    print("📋 后续步骤:")
    print(f"1. 应用环境配置: source {config_path}")
    print(f"2. 运行功能测试: python {test_path}")
    print(f"3. 运行分布式测试: python {distributed_test_path}")
    print("4. 使用增强的分布式训练脚本: train_distributed_gcu_robust.py")
    
    print("\n📁 重要文件:")
    for name, path in found_files.items():
        print(f"   - {name}: {path}")
    
    print(f"\n🔧 配置文件: {config_path}")
    print(f"🧪 测试脚本: {test_path}")
    print(f"🚀 分布式测试: {distributed_test_path}")
    
    print("\n✅ ECCL环境设置完成！")

if __name__ == "__main__":
    main()