#!/bin/bash

# 修复T20服务器上的mmcv版本兼容性问题
# mmseg要求: 1.3.13 <= mmcv < 1.8.0

echo "🔧 修复T20服务器mmcv版本兼容性问题..."

# 检查当前mmcv版本
echo "📋 检查当前mmcv版本..."
python3 -c "
try:
    import mmcv
    print(f'当前mmcv版本: {mmcv.__version__}')
except ImportError:
    print('mmcv未安装')
"

# 卸载所有mmcv相关包
echo "🗑️  卸载现有mmcv包..."
pip3 uninstall mmcv mmcv-full mmcv-lite -y

# 清理pip缓存
echo "🧹 清理pip缓存..."
pip3 cache purge

# 安装兼容版本的mmcv-full
echo "📦 安装兼容版本mmcv-full==1.7.2..."
pip3 install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html

# 验证安装和兼容性
echo "✅ 验证安装结果..."
python3 -c "
import mmcv
from packaging import version

print(f'✅ mmcv安装成功，版本: {mmcv.__version__}')

# 检查版本兼容性
mmcv_version = version.parse(mmcv.__version__)
mmcv_min = version.parse('1.3.13')
mmcv_max = version.parse('1.8.0')

if mmcv_min <= mmcv_version < mmcv_max:
    print('✅ mmcv版本兼容mmseg要求 (1.3.13 <= version < 1.8.0)')
else:
    print(f'❌ mmcv版本{mmcv.__version__}不兼容mmseg要求')
    print(f'   要求: 1.3.13 <= version < 1.8.0')
    exit(1)
"

# 测试mmseg导入
echo "🧪 测试mmseg导入..."
python3 -c "
try:
    import mmseg
    print('✅ mmseg导入成功')
except Exception as e:
    print(f'❌ mmseg导入失败: {e}')
    exit(1)
"

echo "🎉 mmcv版本修复完成！现在可以正常使用mmseg了。"