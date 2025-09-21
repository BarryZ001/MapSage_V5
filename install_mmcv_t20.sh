#!/bin/bash

# T20服务器安装mmcv脚本
# 适用于燧原T20 GCU环境

echo "开始安装mmcv for T20 GCU环境..."

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "已激活虚拟环境"
fi

# 安装mmcv-full
echo "安装mmcv-full..."
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html

# 验证安装
echo "验证mmcv安装..."
python -c "import mmcv; print(f'mmcv version: {mmcv.__version__}')"

# 如果mmcv安装失败，尝试从源码安装
if [ $? -ne 0 ]; then
    echo "mmcv-full安装失败，尝试安装mmcv-lite..."
    pip install mmcv==2.0.1
    
    # 再次验证
    python -c "import mmcv; print(f'mmcv version: {mmcv.__version__}')"
fi

echo "mmcv安装完成！"