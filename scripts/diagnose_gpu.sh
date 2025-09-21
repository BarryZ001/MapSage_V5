#!/bin/bash
# GPU环境诊断脚本
# 用于检查T20服务器的GPU驱动和CUDA环境

echo "🔍 开始GPU环境诊断..."
echo "=================================="

# 1. 检查NVIDIA驱动
echo "1️⃣ 检查NVIDIA驱动:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ nvidia-smi 可用"
else
    echo "❌ nvidia-smi 不可用 - NVIDIA驱动可能未安装"
fi
echo ""

# 2. 检查CUDA版本
echo "2️⃣ 检查CUDA版本:"
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✅ CUDA编译器可用"
else
    echo "❌ nvcc 不可用 - CUDA可能未安装"
fi
echo ""

# 3. 检查CUDA环境变量
echo "3️⃣ 检查CUDA环境变量:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH中的CUDA: $(echo $PATH | grep -o '/usr/local/cuda[^:]*' || echo '未找到')"
echo ""

# 4. 检查GPU设备
echo "4️⃣ 检查GPU设备:"
if [ -d "/dev" ]; then
    ls -la /dev/nvidia* 2>/dev/null || echo "❌ 未找到NVIDIA设备文件"
fi
echo ""

# 5. 检查Python CUDA支持
echo "5️⃣ 检查Python CUDA支持:"
python3 -c "
import sys
print(f'Python版本: {sys.version}')

try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA版本: {torch.version.cuda}')
        print(f'GPU数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('❌ PyTorch无法访问CUDA')
except ImportError:
    print('❌ PyTorch未安装')

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    print(f'NVML检测到GPU数量: {device_count}')
except ImportError:
    print('⚠️  nvidia-ml-py3未安装')
except Exception as e:
    print(f'❌ NVML错误: {e}')
"
echo ""

# 6. 检查Docker GPU支持（如果在容器中）
echo "6️⃣ 检查Docker GPU支持:"
if [ -f /.dockerenv ]; then
    echo "✅ 运行在Docker容器中"
    if command -v nvidia-container-cli &> /dev/null; then
        echo "✅ nvidia-container-toolkit可用"
    else
        echo "❌ nvidia-container-toolkit不可用"
    fi
else
    echo "ℹ️  不在Docker容器中运行"
fi
echo ""

# 7. 检查系统信息
echo "7️⃣ 系统信息:"
echo "内核版本: $(uname -r)"
echo "发行版: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2 2>/dev/null || echo '未知')"
echo "架构: $(uname -m)"
echo ""

echo "=================================="
echo "🔍 GPU环境诊断完成"
echo ""
echo "💡 常见解决方案:"
echo "1. 如果nvidia-smi不可用，需要安装NVIDIA驱动"
echo "2. 如果CUDA不可用，需要安装CUDA toolkit"
echo "3. 如果PyTorch无法访问CUDA，需要重新安装支持CUDA的PyTorch"
echo "4. 如果在Docker中，确保使用--gpus all参数启动容器"