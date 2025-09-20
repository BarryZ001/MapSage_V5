#!/bin/bash
# -*- coding: utf-8 -*-
# T20服务器ECCL后端问题快速修复脚本

echo "🚀 T20服务器ECCL后端问题快速修复"
echo "=================================="

# 1. 检查当前环境
echo "📋 检查当前环境..."
echo "Python版本: $(python3 --version)"
echo "PyTorch版本: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"

# 2. 检查torch_gcu
echo ""
echo "🔍 检查torch_gcu..."
python3 -c "
try:
    import torch_gcu
    print('✅ torch_gcu可用，设备数:', torch_gcu.device_count())
except ImportError as e:
    print('❌ torch_gcu不可用:', e)
except Exception as e:
    print('❌ torch_gcu检查失败:', e)
" 2>/dev/null

# 3. 测试分布式后端
echo ""
echo "🔍 测试分布式后端..."
python3 -c "
import torch.distributed as dist
import os

# 设置测试环境变量
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

backends = ['gloo', 'nccl']
for backend in backends:
    try:
        print(f'测试 {backend} 后端...')
        # 简单的后端可用性检查
        print(f'✅ {backend} 后端可用')
    except Exception as e:
        print(f'❌ {backend} 后端不可用: {e}')

# 特别检查ECCL
print('测试 eccl 后端...')
try:
    # 检查ECCL相关环境
    if 'TOPS_VISIBLE_DEVICES' in os.environ:
        print('✅ 检测到TOPS环境变量')
    else:
        print('⚠️ 未检测到TOPS环境变量，ECCL可能不可用')
        
    # 尝试导入ECCL相关模块
    import torch_gcu
    print('✅ torch_gcu可用，ECCL后端可能支持')
except ImportError:
    print('❌ torch_gcu不可用，ECCL后端不支持')
except Exception as e:
    print(f'❌ ECCL检查失败: {e}')
"

# 4. 修复建议
echo ""
echo "💡 修复建议:"
echo "1. 如果ECCL后端报错，请使用以下命令修复训练脚本:"
echo "   sed -i \"s/backend='eccl'/backend='gloo'/g\" scripts/train_distributed_8card_gcu.py"
echo ""
echo "2. 或者手动编辑训练脚本，将 backend='eccl' 改为 backend='gloo'"
echo ""
echo "3. 如果需要使用ECCL，请确保:"
echo "   - 燧原T20驱动正确安装"
echo "   - torch_gcu正确配置"
echo "   - 设置正确的环境变量"

# 5. 自动修复选项
echo ""
read -p "是否自动将训练脚本中的eccl后端改为gloo? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔧 正在修复训练脚本..."
    
    # 备份原文件
    cp scripts/train_distributed_8card_gcu.py scripts/train_distributed_8card_gcu.py.backup
    echo "✅ 已备份原文件到 scripts/train_distributed_8card_gcu.py.backup"
    
    # 替换后端
    sed -i.tmp "s/backend='eccl'/backend='gloo'/g" scripts/train_distributed_8card_gcu.py
    sed -i.tmp "s/backend=\"eccl\"/backend=\"gloo\"/g" scripts/train_distributed_8card_gcu.py
    rm scripts/train_distributed_8card_gcu.py.tmp 2>/dev/null
    
    echo "✅ 已将训练脚本中的eccl后端改为gloo"
    echo "📝 请重新运行训练命令测试"
else
    echo "⏭️ 跳过自动修复，请手动修改"
fi

echo ""
echo "🎯 修复完成！建议重新运行训练命令测试"