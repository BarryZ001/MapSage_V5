# ECCL容器环境使用指南

## 概述

基于您提供的信息，ECCL 2.5.136已在容器环境中正确安装。本指南将帮助您在容器中配置和使用ECCL进行分布式训练。

## 已安装的ECCL组件

根据 `dpkg -L tops-eccl` 的输出，以下组件已安装：

### 核心库文件
- **ECCL库**: `/usr/lib/libeccl.so`
- **ECCL头文件**: `/usr/include/eccl/eccl.h`

### 性能测试工具
- `eccl_all_gather_perf`
- `eccl_all_reduce_perf`
- `eccl_all_to_all_perf`
- `eccl_broadcast_perf`
- `eccl_gather_perf`
- `eccl_reduce_perf`
- `eccl_reduce_scatter_perf`
- `eccl_scatter_perf`
- `eccl_send_recv_perf`

所有工具位于 `/usr/local/bin/` 目录下。

## 容器环境配置

### 1. 环境变量设置

在容器中创建并运行以下环境配置脚本：

```bash
#!/bin/bash
# 保存为 /tmp/eccl_env.sh

# ECCL核心配置
export ECCL_DEBUG=0
export ECCL_LOG_LEVEL=INFO
export ECCL_SOCKET_IFNAME=eth0
export ECCL_IB_DISABLE=1
export ECCL_ASYNC_DISABLE=false
export ECCL_MAX_NCHANNELS=2

# 库路径配置
export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH}"

# 工具路径配置
export PATH="/usr/local/bin:${PATH}"

# GCU设备配置
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=""

# 分布式训练配置
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "✅ ECCL环境变量配置完成"
```

应用配置：
```bash
source /tmp/eccl_env.sh
```

### 2. 验证ECCL安装

在容器中运行以下命令验证安装：

```bash
# 检查库文件
ls -la /usr/lib/libeccl.so

# 检查头文件
ls -la /usr/include/eccl/eccl.h

# 检查性能工具
ls -la /usr/local/bin/eccl_*_perf

# 测试库依赖
ldd /usr/lib/libeccl.so
```

### 3. 测试ECCL功能

#### 基本功能测试
```bash
# 测试all_reduce性能工具
/usr/local/bin/eccl_all_reduce_perf --help

# 测试broadcast性能工具
/usr/local/bin/eccl_broadcast_perf --help
```

#### Python集成测试

创建测试脚本 `/tmp/test_eccl.py`：

```python
#!/usr/bin/env python3
import os
import ctypes

def test_eccl_library():
    """测试ECCL库加载"""
    try:
        lib_path = '/usr/lib/libeccl.so'
        if os.path.exists(lib_path):
            lib = ctypes.CDLL(lib_path)
            print(f"✅ 成功加载ECCL库: {lib_path}")
            return True
        else:
            print(f"❌ ECCL库文件不存在: {lib_path}")
            return False
    except Exception as e:
        print(f"❌ 加载ECCL库失败: {e}")
        return False

def test_environment():
    """测试环境变量"""
    eccl_vars = ['ECCL_DEBUG', 'ECCL_LOG_LEVEL', 'LD_LIBRARY_PATH']
    for var in eccl_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

if __name__ == "__main__":
    print("🧪 ECCL功能测试")
    test_environment()
    test_eccl_library()
```

运行测试：
```bash
python /tmp/test_eccl.py
```

## 分布式训练配置

### 使用增强的分布式训练脚本

我们已经创建了 `train_distributed_gcu_robust.py` 脚本，专门用于解决ECCL分布式训练中的TCP连接问题。

#### 在容器中运行分布式训练

1. **单机多卡训练**：
```bash
# 在容器中设置环境
source /tmp/eccl_env.sh

# 运行分布式训练
python train_distributed_gcu_robust.py \
    configs/your_config.py \
    --launcher pytorch \
    --backend gloo \
    --max-retries 5 \
    --retry-delay 10
```

2. **多机训练**：
```bash
# 主节点 (rank 0)
python train_distributed_gcu_robust.py \
    configs/your_config.py \
    --launcher pytorch \
    --backend gloo \
    --master-addr <MASTER_IP> \
    --master-port 29500

# 其他节点
python train_distributed_gcu_robust.py \
    configs/your_config.py \
    --launcher pytorch \
    --backend gloo \
    --master-addr <MASTER_IP> \
    --master-port 29500
```

### 关键配置说明

#### ECCL后端配置
- `ECCL_SOCKET_IFNAME=eth0`: 指定网络接口
- `ECCL_IB_DISABLE=1`: 禁用InfiniBand（如果不可用）
- `ECCL_DEBUG=0`: 设置调试级别
- `ECCL_LOG_LEVEL=INFO`: 设置日志级别

#### 网络配置
- `MASTER_ADDR`: 主节点IP地址
- `MASTER_PORT`: 通信端口（默认29500）
- 确保容器间网络连通性

## 故障排除

### 常见问题及解决方案

#### 1. TCP连接问题
```
Connection closed by peer
```

**解决方案**：
- 使用 `train_distributed_gcu_robust.py` 脚本
- 增加重试次数和延迟
- 检查网络配置

#### 2. 库加载问题
```
libeccl.so: cannot open shared object file
```

**解决方案**：
```bash
export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH}"
ldconfig  # 如果有权限
```

#### 3. 设备不可见
```
No GCU devices found
```

**解决方案**：
```bash
export TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 检查设备状态
lspci | grep -i enflame  # 或相应的设备检查命令
```

#### 4. 进程组初始化失败

**解决方案**：
- 确保所有节点使用相同的 `MASTER_ADDR` 和 `MASTER_PORT`
- 检查防火墙设置
- 增加超时时间

## 性能优化建议

### 1. ECCL参数调优
```bash
export ECCL_MAX_NCHANNELS=4  # 根据网络带宽调整
export ECCL_ASYNC_DISABLE=false  # 启用异步通信
```

### 2. 网络优化
- 使用高速网络接口
- 调整TCP缓冲区大小
- 优化网络拓扑

### 3. 内存优化
- 合理设置batch size
- 启用梯度累积
- 使用混合精度训练

## 监控和调试

### 1. 启用详细日志
```bash
export ECCL_DEBUG=1
export ECCL_LOG_LEVEL=DEBUG
```

### 2. 性能监控
```bash
# 使用性能工具测试
/usr/local/bin/eccl_all_reduce_perf -b 1M -e 1G -i 100

# 监控网络使用
iftop -i eth0
```

### 3. 系统资源监控
```bash
# CPU和内存使用
htop

# GPU使用（如果适用）
nvidia-smi  # 或相应的GCU监控命令
```

## 总结

ECCL 2.5.136已在您的容器环境中正确安装。通过正确配置环境变量和使用我们提供的增强分布式训练脚本，您应该能够成功运行分布式训练任务。

如果遇到问题，请：
1. 检查环境变量配置
2. 验证网络连通性
3. 查看详细日志输出
4. 使用性能工具进行诊断

关键文件位置：
- ECCL库：`/usr/lib/libeccl.so`
- ECCL头文件：`/usr/include/eccl/eccl.h`
- 性能工具：`/usr/local/bin/eccl_*_perf`
- 增强训练脚本：`train_distributed_gcu_robust.py`