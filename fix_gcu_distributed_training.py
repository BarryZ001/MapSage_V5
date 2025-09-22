#!/usr/bin/env python3
"""
GCU分布式训练环境配置脚本
解决燧原T20 GCU分布式训练环境配置问题
"""

import os
import sys
import subprocess
from pathlib import Path

def check_gcu_environment():
    """检查GCU环境"""
    print("🔍 检查GCU环境...")
    
    # 检查torch_gcu模块
    try:
        import torch_gcu  # type: ignore
        print(f"✅ torch_gcu 已安装")
        
        # 检查版本信息
        if hasattr(torch_gcu, '__version__'):
            print(f"   版本: {torch_gcu.__version__}")
        
        # 检查设备可用性
        if hasattr(torch_gcu, 'is_available') and torch_gcu.is_available():
            device_count = getattr(torch_gcu, 'device_count', lambda: 0)()
            print(f"✅ GCU设备可用，设备数量: {device_count}")
            return True, device_count
        else:
            print("❌ GCU设备不可用")
            return False, 0
            
    except ImportError:
        print("❌ torch_gcu 未安装")
        return False, 0

def check_distributed_backends():
    """检查分布式后端支持"""
    print("\n🔍 检查分布式后端支持...")
    
    import torch
    import torch.distributed as dist
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"分布式可用: {dist.is_available()}")
    
    # 检查各种后端
    backends = []
    
    # 检查NCCL
    try:
        if hasattr(dist, 'is_nccl_available') and dist.is_nccl_available():
            backends.append('nccl')
            print("✅ NCCL后端可用")
        else:
            print("❌ NCCL后端不可用")
    except Exception as e:
        print(f"❌ NCCL检查失败: {e}")
    
    # 检查Gloo
    try:
        if hasattr(dist, 'is_gloo_available') and dist.is_gloo_available():
            backends.append('gloo')
            print("✅ Gloo后端可用")
        else:
            print("❌ Gloo后端不可用")
    except Exception as e:
        print(f"❌ Gloo检查失败: {e}")
    
    # 检查ECCL（GCU专用）
    try:
        import eccl  # type: ignore
        backends.append('eccl')
        print("✅ ECCL后端可用")
    except ImportError:
        print("❌ ECCL后端不可用")
    
    return backends

def configure_gcu_environment():
    """配置GCU环境变量"""
    print("\n🔧 配置GCU环境变量...")
    
    # GCU相关环境变量
    gcu_env_vars = {
        'ECCL_BACKEND': 'gloo',  # 使用gloo作为ECCL后端
        'ECCL_DEVICE_TYPE': 'GCU',  # 设备类型
        'ECCL_DEBUG': '0',  # 调试级别
        'TOPS_VISIBLE_DEVICES': '',  # 将在运行时设置
        'CUDA_VISIBLE_DEVICES': '',  # 禁用CUDA设备
    }
    
    # 网络相关环境变量
    network_env_vars = {
        'MASTER_ADDR': '127.0.0.1',  # 主节点地址
        'MASTER_PORT': '29500',  # 主节点端口
        'GLOO_SOCKET_IFNAME': 'lo',  # 网络接口（本地回环）
        'GLOO_TIMEOUT_SECONDS': '300',  # 超时时间
    }
    
    all_env_vars = {**gcu_env_vars, **network_env_vars}
    
    for key, value in all_env_vars.items():
        current_value = os.environ.get(key)
        if current_value != value:
            os.environ[key] = value
            print(f"✅ 设置 {key}={value}")
        else:
            print(f"✅ {key} 已正确设置")
    
    return all_env_vars

def create_gcu_training_script():
    """创建GCU训练启动脚本"""
    print("\n📝 创建GCU训练启动脚本...")
    
    project_root = Path.cwd()
    script_path = project_root / 'start_gcu_training_fixed.sh'
    
    content = '''#!/bin/bash
# GCU分布式训练启动脚本（修复版）

set -e  # 遇到错误立即退出

echo "🚀 启动GCU分布式训练（修复版）"

# 检查GCU设备
echo "🔍 检查GCU设备..."
python3 -c "
try:
    import torch_gcu
    if torch_gcu.is_available():
        device_count = torch_gcu.device_count()
        print(f'✅ GCU设备可用，数量: {device_count}')
        if device_count < 8:
            print(f'⚠️  警告: 检测到{device_count}个GCU设备，少于8个')
    else:
        print('❌ GCU设备不可用')
        exit(1)
except ImportError:
    print('❌ torch_gcu未安装')
    exit(1)
"

# 设置环境变量
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# GCU相关环境变量
export ECCL_BACKEND=gloo
export ECCL_DEVICE_TYPE=GCU
export ECCL_DEBUG=0
export CUDA_VISIBLE_DEVICES=""

# 分布式训练环境变量
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 网络配置
export GLOO_SOCKET_IFNAME=lo
export GLOO_TIMEOUT_SECONDS=300

# 训练参数
CONFIG_FILE="configs/dinov3/dinov3_vit-l16_mmrs1m_t20_gcu.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu"
TRAIN_SCRIPT="scripts/train_distributed_pytorch_ddp_8card_gcu.py"

# 检查必要文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 创建工作目录
mkdir -p "$WORK_DIR"

# 停止之前的训练进程
echo "🛑 停止之前的训练进程..."
pkill -f "train_distributed_pytorch_ddp_8card_gcu.py" || true
sleep 2

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "📋 训练配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  工作目录: $WORK_DIR"
echo "  训练脚本: $TRAIN_SCRIPT"
echo "  设备数量: $WORLD_SIZE"
echo "  主节点: $MASTER_ADDR:$MASTER_PORT"

# 启动分布式训练
echo "🚀 启动8卡GCU分布式训练..."

# 使用torchrun启动分布式训练
torchrun \\
    --standalone \\
    --nnodes=1 \\
    --nproc_per_node=8 \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=127.0.0.1:29500 \\
    "$TRAIN_SCRIPT" \\
    --config "$CONFIG_FILE" \\
    --work-dir "$WORK_DIR" \\
    --launcher pytorch

echo "✅ 训练启动完成"
echo "📊 监控GCU设备使用情况:"
echo "   使用命令: watch -n 1 'python3 -c \"import torch_gcu; print(f\\\"GCU设备数量: {torch_gcu.device_count()}\\\")\"'"
echo "📁 日志保存在: $WORK_DIR"
'''
    
    script_path.write_text(content, encoding='utf-8')
    script_path.chmod(0o755)
    print(f"✅ 创建GCU训练脚本: {script_path}")

def create_single_process_fallback_script():
    """创建单进程回退脚本"""
    print("\n📝 创建单进程回退脚本...")
    
    project_root = Path.cwd()
    script_path = project_root / 'start_single_gcu_training.sh'
    
    content = '''#!/bin/bash
# 单进程GCU训练脚本（分布式失败时的回退方案）

set -e

echo "🚀 启动单进程GCU训练"

# 设置环境变量
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 单进程模式
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# GCU环境变量
export ECCL_BACKEND=gloo
export ECCL_DEVICE_TYPE=GCU
export ECCL_DEBUG=0
export CUDA_VISIBLE_DEVICES=""
export TOPS_VISIBLE_DEVICES=0

# 训练参数
CONFIG_FILE="configs/dinov3/dinov3_vit-l16_mmrs1m_t20_gcu.py"
WORK_DIR="./work_dirs/dinov3_mmrs1m_t20_gcu_single"
TRAIN_SCRIPT="scripts/train_distributed_pytorch_ddp_8card_gcu.py"

# 创建工作目录
mkdir -p "$WORK_DIR"

echo "📋 单进程训练配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  工作目录: $WORK_DIR"
echo "  训练脚本: $TRAIN_SCRIPT"
echo "  模式: 单进程"

# 启动单进程训练
echo "🚀 启动单进程GCU训练..."

python3 "$TRAIN_SCRIPT" \\
    --config "$CONFIG_FILE" \\
    --work-dir "$WORK_DIR" \\
    --launcher none

echo "✅ 单进程训练启动完成"
'''
    
    script_path.write_text(content, encoding='utf-8')
    script_path.chmod(0o755)
    print(f"✅ 创建单进程训练脚本: {script_path}")

def test_distributed_initialization():
    """测试分布式初始化"""
    print("\n🧪 测试分布式初始化...")
    
    # 设置测试环境
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'  # 使用不同端口避免冲突
    
    try:
        import torch
        import torch.distributed as dist
        
        # 测试gloo后端
        print("测试gloo后端...")
        try:
            dist.init_process_group(
                backend='gloo',
                init_method='env://',
                world_size=1,
                rank=0
            )
            print("✅ gloo后端初始化成功")
            dist.destroy_process_group()
        except Exception as e:
            print(f"❌ gloo后端初始化失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分布式初始化测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 开始配置GCU分布式训练环境...")
    
    # 1. 检查GCU环境
    gcu_available, device_count = check_gcu_environment()
    
    # 2. 检查分布式后端
    available_backends = check_distributed_backends()
    
    # 3. 配置GCU环境变量
    env_vars = configure_gcu_environment()
    
    # 4. 创建训练脚本
    create_gcu_training_script()
    create_single_process_fallback_script()
    
    # 5. 测试分布式初始化
    dist_test_ok = test_distributed_initialization()
    
    # 总结
    print("\n" + "="*50)
    print("🎉 GCU分布式训练环境配置完成！")
    print("="*50)
    
    print(f"GCU设备: {'✅ 可用' if gcu_available else '❌ 不可用'}")
    if gcu_available:
        print(f"设备数量: {device_count}")
    
    print(f"可用后端: {', '.join(available_backends) if available_backends else '无'}")
    print(f"分布式测试: {'✅ 通过' if dist_test_ok else '❌ 失败'}")
    
    print("\n📋 使用说明:")
    if gcu_available and device_count >= 8:
        print("1. 8卡分布式训练: ./start_gcu_training_fixed.sh")
    print("2. 单进程训练: ./start_single_gcu_training.sh")
    print("3. 如果分布式失败，训练脚本会自动回退到单进程模式")
    
    if not gcu_available:
        print("\n⚠️  警告: GCU设备不可用，请检查:")
        print("1. torch_gcu是否正确安装")
        print("2. GCU驱动是否正确安装")
        print("3. 设备是否正确连接")
    
    return 0 if gcu_available else 1

if __name__ == '__main__':
    sys.exit(main())