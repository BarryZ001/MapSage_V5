#!/bin/bash
# -*- coding: utf-8 -*-

# T20环境PyTorch修复脚本
# 修复torch.distributed模块缺失问题

set -e

echo "🔧 T20环境PyTorch修复脚本"
echo "================================"

# 检查当前环境
echo "📋 检查当前环境..."
echo "Python版本: $(python3 --version)"
echo "pip版本: $(pip3 --version)"
echo "当前用户: $(whoami)"
echo "当前目录: $(pwd)"

# 检查现有PyTorch安装
echo ""
echo "🔍 检查现有PyTorch安装..."
python3 -c "import torch; print('PyTorch版本:', torch.__version__)" 2>/dev/null || echo "❌ PyTorch未正确安装"

# 检查torch.distributed
echo ""
echo "🔍 检查torch.distributed模块..."
python3 -c "import torch.distributed as dist; print('✅ torch.distributed可用')" 2>/dev/null || echo "❌ torch.distributed不可用"

# 方案1: 重新安装PyTorch
echo ""
echo "🔧 方案1: 重新安装PyTorch..."
echo "卸载现有PyTorch..."
pip3 uninstall -y torch torchvision torchaudio || true

echo "安装PyTorch CPU版本..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 验证安装
echo ""
echo "✅ 验证PyTorch安装..."
python3 -c "
import torch
import torch.distributed as dist
print('PyTorch版本:', torch.__version__)
print('torch.distributed可用:', dist.is_available())
print('支持的后端:', [backend for backend in ['gloo', 'nccl', 'mpi'] if getattr(dist, f'is_{backend}_available', lambda: False)()])
"

# 检查torch_gcu
echo ""
echo "🔍 检查torch_gcu模块..."
python3 -c "
try:
    import torch_gcu
    print('✅ torch_gcu版本:', getattr(torch_gcu, '__version__', '未知'))
    print('✅ 可用GCU设备数:', torch_gcu.device_count())
except ImportError as e:
    print('❌ torch_gcu导入失败:', e)
    print('💡 可能需要重新安装torch_gcu')
"

# 检查torchrun
echo ""
echo "🔍 检查torchrun命令..."
if command -v torchrun >/dev/null 2>&1; then
    echo "✅ torchrun命令可用: $(which torchrun)"
else
    echo "❌ torchrun命令不可用"
    echo "💡 尝试使用: python3 -m torch.distributed.run"
fi

# 测试分布式功能
echo ""
echo "🧪 测试分布式功能..."
python3 -c "
import torch
import torch.distributed as dist
import os

# 设置测试环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

try:
    # 初始化进程组
    dist.init_process_group(backend='gloo', world_size=1, rank=0)
    print('✅ 分布式进程组初始化成功')
    dist.destroy_process_group()
    print('✅ 分布式进程组清理成功')
except Exception as e:
    print('❌ 分布式测试失败:', e)
"

echo ""
echo "🏁 PyTorch修复完成！"
echo "================================"
echo ""
echo "📝 使用建议:"
echo "1. 如果torch_gcu有问题，请联系管理员重新安装"
echo "2. 使用以下命令测试训练:"
echo "   python3 scripts/diagnose_t20_pytorch.py"
echo "3. 如果还有问题，尝试重启容器"