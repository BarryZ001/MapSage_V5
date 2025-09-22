#!/bin/bash
# 训练环境设置脚本

# 设置项目根目录
export PROJECT_ROOT="/Users/barryzhang/myDev3/MapSage_V5"

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
    print(f'❌ mmseg_custom: {e}')

try:
    import mmseg
    print(f'✅ mmseg (版本: {mmseg.__version__})')
except ImportError as e:
    print(f'❌ mmseg: {e}')
"

echo "✅ 环境设置完成"
