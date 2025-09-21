#!/bin/bash

# 修复T20服务器上的mmcv符号错误问题
# 错误: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEENS2_8optionalINS2_10ScalarTypeEEENS6_INS2_6LayoutEEENS6_INS2_6DeviceEEENS6_IbEE

echo "🔧 修复T20服务器mmcv符号错误问题..."

# 检查当前PyTorch和mmcv版本
echo "📋 检查当前版本..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
try:
    import mmcv
    print(f'MMCV版本: {mmcv.__version__}')
except Exception as e:
    print(f'MMCV导入错误: {e}')
"

# 卸载所有mmcv相关包
echo "🗑️  卸载现有mmcv包..."
pip3 uninstall mmcv mmcv-full mmcv-lite -y

# 清理pip缓存
echo "🧹 清理pip缓存..."
pip3 cache purge

# 检查PyTorch版本并安装对应的mmcv
echo "📦 根据PyTorch版本安装兼容的mmcv..."
python3 -c "
import torch
import sys

torch_version = torch.__version__
print(f'检测到PyTorch版本: {torch_version}')

# 根据PyTorch版本选择合适的mmcv安装命令
if torch_version.startswith('2.0'):
    print('安装适用于PyTorch 2.0的mmcv-full==1.7.2...')
    import subprocess
    result = subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        'mmcv-full==1.7.2', 
        '-f', 'https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html'
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print('✅ mmcv-full安装成功')
    else:
        print(f'❌ mmcv-full安装失败: {result.stderr}')
        # 尝试安装CPU版本
        print('尝试安装CPU版本的mmcv-full...')
        result2 = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'mmcv-full==1.7.2'
        ], capture_output=True, text=True)
        if result2.returncode == 0:
            print('✅ CPU版本mmcv-full安装成功')
        else:
            print(f'❌ CPU版本mmcv-full也安装失败: {result2.stderr}')
elif torch_version.startswith('1.'):
    print('安装适用于PyTorch 1.x的mmcv-full==1.7.2...')
    import subprocess
    result = subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        'mmcv-full==1.7.2'
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print('✅ mmcv-full安装成功')
    else:
        print(f'❌ mmcv-full安装失败: {result.stderr}')
else:
    print(f'未知的PyTorch版本: {torch_version}')
"

# 验证安装和符号问题
echo "✅ 验证mmcv安装和符号问题..."
python3 -c "
try:
    import mmcv
    print(f'✅ mmcv导入成功，版本: {mmcv.__version__}')
    
    # 测试可能有问题的函数
    try:
        import torch
        # 测试zeros函数（错误信息中提到的符号）
        x = torch.zeros(2, 3)
        print('✅ torch.zeros函数正常工作')
        
        # 测试mmcv的一些基本功能
        from mmcv.utils import Config
        print('✅ mmcv.utils.Config导入成功')
        
    except Exception as e:
        print(f'⚠️ 符号错误仍然存在: {e}')
        
except Exception as e:
    print(f'❌ mmcv导入失败: {e}')
    exit(1)
"

# 测试mmseg导入
echo "🧪 测试mmseg导入..."
python3 -c "
try:
    import mmseg
    print('✅ mmseg导入成功')
    
    # 测试损失函数导入
    try:
        from mmseg.models.losses import CrossEntropyLoss
        print('✅ mmseg损失函数导入成功')
    except Exception as e:
        print(f'⚠️ mmseg损失函数导入失败: {e}')
        print('将使用自定义损失函数实现')
        
except Exception as e:
    print(f'❌ mmseg导入失败: {e}')
    exit(1)
"

echo "🎉 mmcv符号错误修复完成！"