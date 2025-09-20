#!/bin/bash

# T20服务器Horovod安装脚本
# 适用于燧原T20 GCU环境

set -e

echo "=== T20服务器Horovod安装脚本 ==="
echo "开始安装Horovod及其依赖..."

# 检查当前环境
echo "检查当前Python环境..."
python3 --version
pip3 --version

# 检查MPI环境
echo "检查MPI环境..."
which mpirun || echo "Warning: mpirun not found in PATH"
mpirun --version || echo "Warning: Cannot get mpirun version"

# 安装系统依赖
echo "安装系统依赖..."
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    libgfortran5

# 尝试安装NCCL（如果可用）
echo "尝试安装NCCL库..."
apt-get install -y libnccl2 libnccl-dev || echo "Warning: NCCL packages not available, continuing without NCCL"

# 设置环境变量
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_GPU_OPERATIONS=NCCL

# 检查NCCL
echo "检查NCCL库..."
ldconfig -p | grep nccl || echo "Warning: NCCL not found in library path"

# 安装Horovod
echo "安装Horovod..."
pip3 install --no-cache-dir horovod[pytorch]

# 验证安装
echo "验证Horovod安装..."
python3 -c "import horovod.torch as hvd; print('Horovod version:', hvd.__version__)"

# 检查Horovod配置
echo "检查Horovod配置..."
horovodrun --check-build

echo "=== Horovod安装完成 ==="
echo "可以使用以下命令测试："
echo "horovodrun -np 2 python3 -c \"import horovod.torch as hvd; hvd.init(); print(f'Rank {hvd.rank()}/{hvd.size()}')\""