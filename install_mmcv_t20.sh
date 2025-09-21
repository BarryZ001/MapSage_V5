#!/bin/bash

# T20服务器安装mmcv脚本
# 适用于燧原T20 GCU环境

echo "开始安装兼容版本的mmcv for T20 GCU环境..."

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "已激活虚拟环境"
fi

# 卸载现有的不兼容版本
echo "卸载现有的mmcv版本..."
pip uninstall mmcv mmcv-full -y

# 安装兼容的mmcv-full版本 (1.7.2 符合 1.3.13 <= version < 1.8.0 的要求)
echo "安装兼容的mmcv-full==1.7.2..."
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html

# 验证安装
echo "验证mmcv安装..."
python -c "import mmcv; print(f'mmcv version: {mmcv.__version__}')"

# 检查版本兼容性
echo "检查mmseg兼容性..."
python -c "
import mmcv
from packaging import version
mmcv_version = version.parse(mmcv.__version__)
mmcv_min = version.parse('1.3.13')
mmcv_max = version.parse('1.8.0')
if mmcv_min <= mmcv_version < mmcv_max:
    print('✅ mmcv版本兼容mmseg要求')
else:
    print(f'❌ mmcv版本{mmcv.__version__}不兼容，需要1.3.13 <= version < 1.8.0')
"

echo "mmcv安装完成！"