#!/bin/bash

# T20服务器MMCV兼容性修复脚本
# 解决MMCV==2.2.0与MMSegmentation不兼容的问题

echo "🔧 T20服务器MMCV兼容性修复脚本"
echo "解决MMCV版本与MMSegmentation的兼容性问题..."

# 检查当前环境
echo "📋 检查当前环境状态..."
python3 -c "
try:
    import mmcv
    print(f'当前MMCV版本: {mmcv.__version__}')
except ImportError:
    print('MMCV未安装')
    
try:
    import mmseg
    print(f'当前MMSegmentation版本: {mmseg.__version__}')
except ImportError:
    print('MMSegmentation未安装')
"

# 卸载现有的MMCV相关包
echo "🗑️  卸载现有MMCV包..."
pip3 uninstall -y mmcv mmcv-full mmcv-lite || echo "无现有MMCV包需要卸载"

# 清理pip缓存
echo "🧹 清理pip缓存..."
pip3 cache purge

# 安装兼容版本的MMCV
echo "📦 安装兼容版本的MMCV..."
# MMSegmentation 0.30.0 需要 mmcv>=2.0.0rc4
pip3 install --no-cache-dir "mmcv>=2.0.0rc4,<2.2.0"

# 验证安装
echo "✅ 验证MMCV安装..."
python3 -c "
try:
    import mmcv
    print(f'✅ MMCV安装成功，版本: {mmcv.__version__}')
    
    # 检查版本兼容性
    from packaging import version
    mmcv_version = version.parse(mmcv.__version__)
    mmcv_min = version.parse('2.0.0rc4')
    mmcv_max = version.parse('2.2.0')
    
    if mmcv_min <= mmcv_version < mmcv_max:
        print('✅ MMCV版本兼容MMSegmentation要求 (>=2.0.0rc4, <2.2.0)')
    else:
        print(f'❌ MMCV版本{mmcv.__version__}可能不兼容MMSegmentation要求')
        
except Exception as e:
    print(f'❌ MMCV验证失败: {e}')
    exit(1)
"

# 验证MMSegmentation导入
echo "🧪 验证MMSegmentation导入..."
python3 -c "
try:
    import mmcv
    import mmengine
    import mmseg
    print('✅ 所有模块导入成功')
    print(f'MMCV版本: {mmcv.__version__}')
    print(f'MMEngine版本: {mmengine.__version__}')
    print(f'MMSegmentation版本: {mmseg.__version__}')
except Exception as e:
    print(f'❌ 模块导入失败: {e}')
    exit(1)
"

echo "🎉 MMCV兼容性修复完成！"
echo "现在可以正常使用MMSegmentation进行训练了。"