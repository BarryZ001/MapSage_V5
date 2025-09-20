#!/bin/bash

# T20环境MMCV和MMSegmentation安装脚本
# 基于用户建议的正确版本配置

echo "🔧 T20环境MMCV和MMSegmentation安装脚本"
echo "================================================"

# 检查是否在T20容器环境中
if [ -f "/.dockerenv" ] || [ -n "$CONTAINER_ID" ]; then
    echo "✅ 检测到T20容器环境"
else
    echo "⚠️  未检测到容器环境，请确认在正确的T20环境中运行"
fi

echo ""
echo "🔍 检查当前PyTorch环境..."

# 检查PyTorch版本
python3 -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A')

try:
    import torch_gcu
    print('torch_gcu可用:', torch_gcu.is_available())
except Exception as e:
    print('torch_gcu状态:', e)
"

echo ""
echo "🧹 清理现有MMCV相关包..."

# 卸载可能存在的冲突包
pip3 uninstall -y mmcv mmcv-full mmcv-lite mmsegmentation || echo "无现有包需要卸载"

# 清理pip缓存
pip3 cache purge

echo ""
echo "📦 安装mmcv-full (针对cu102和torch1.10优化)..."

# 根据用户建议安装mmcv-full
# 注意：这里的cu102和torch1.10是与PyTorch版本匹配的关键信息
pip3 install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html

if [ $? -eq 0 ]; then
    echo "✅ mmcv-full安装成功"
else
    echo "❌ mmcv-full安装失败"
    echo "💡 尝试备用安装方法..."
    
    # 备用方法：直接从PyPI安装
    pip3 install mmcv-full==1.6.0
    
    if [ $? -eq 0 ]; then
        echo "✅ mmcv-full备用安装成功"
    else
        echo "❌ mmcv-full安装失败，请检查网络连接"
        exit 1
    fi
fi

echo ""
echo "📦 安装mmsegmentation..."

# 安装mmsegmentation
pip3 install mmsegmentation==0.29.1

if [ $? -eq 0 ]; then
    echo "✅ mmsegmentation安装成功"
else
    echo "❌ mmsegmentation安装失败"
    exit 1
fi

echo ""
echo "🔍 验证安装结果..."

# 验证MMCV安装
python3 -c "
try:
    import mmcv
    print('✅ MMCV版本:', mmcv.__version__)
    print('✅ MMCV安装路径:', mmcv.__file__)
    
    # 检查MMCV编译信息
    from mmcv.utils import collect_env
    env_info = collect_env()
    print('✅ MMCV编译信息:')
    for key in ['MMCV', 'MMCV Compiler', 'MMCV CUDA Ops']:
        if key in env_info:
            print(f'   {key}: {env_info[key]}')
            
except Exception as e:
    print('❌ MMCV验证失败:', e)
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ MMCV验证失败"
    exit 1
fi

# 验证MMSegmentation安装
python3 -c "
try:
    import mmseg
    print('✅ MMSegmentation版本:', mmseg.__version__)
    print('✅ MMSegmentation安装路径:', mmseg.__file__)
    
    # 检查关键组件
    from mmseg.apis import init_segmentor
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    print('✅ MMSegmentation关键组件导入成功')
    
except Exception as e:
    print('❌ MMSegmentation验证失败:', e)
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ MMSegmentation验证失败"
    exit 1
fi

echo ""
echo "🔍 检查预训练权重文件..."

# 检查预训练权重路径
PRETRAINED_WEIGHTS="/workspace/weights/pretrained/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

if [ -f "$PRETRAINED_WEIGHTS" ]; then
    echo "✅ 找到预训练权重文件: $PRETRAINED_WEIGHTS"
    
    # 检查文件大小
    file_size=$(ls -lh "$PRETRAINED_WEIGHTS" | awk '{print $5}')
    echo "   文件大小: $file_size"
    
    # 验证权重文件是否可读
    python3 -c "
import torch
try:
    weights = torch.load('$PRETRAINED_WEIGHTS', map_location='cpu')
    print('✅ 权重文件可正常加载')
    if isinstance(weights, dict):
        print('   权重键数量:', len(weights.keys()))
        print('   主要键:', list(weights.keys())[:5])
except Exception as e:
    print('❌ 权重文件加载失败:', e)
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "✅ 预训练权重验证成功"
    else
        echo "❌ 预训练权重验证失败"
    fi
else
    echo "❌ 未找到预训练权重文件: $PRETRAINED_WEIGHTS"
    echo "💡 请确保权重文件位于正确路径"
fi

echo ""
echo "🧪 测试torch-gcu与MMCV兼容性..."

python3 -c "
try:
    import torch
    import torch_gcu
    import mmcv
    import mmseg
    
    print('✅ 所有模块导入成功')
    
    # 测试基本兼容性
    if torch_gcu.is_available():
        print('✅ torch-gcu可用')
        device = torch.device('gcu:0')
        x = torch.randn(1, 3, 224, 224).to(device)
        print('✅ GCU张量操作正常')
    else:
        print('⚠️  torch-gcu不可用，但模块导入正常')
        
except Exception as e:
    print('❌ 兼容性测试失败:', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ torch-gcu与MMCV兼容性测试通过"
else
    echo "❌ 兼容性测试失败"
fi

echo ""
echo "================================================"
echo "🎉 MMCV和MMSegmentation安装完成！"
echo ""
echo "📋 安装总结:"
echo "  - mmcv-full: 1.6.0 (针对cu102/torch1.10优化)"
echo "  - mmsegmentation: 0.29.1"
echo "  - 预训练权重: $PRETRAINED_WEIGHTS"
echo ""
echo "💡 下一步建议:"
echo "  1. 运行训练脚本验证环境"
echo "  2. 检查配置文件中的权重路径"
echo "  3. 启动8卡GCU分布式训练"
echo ""