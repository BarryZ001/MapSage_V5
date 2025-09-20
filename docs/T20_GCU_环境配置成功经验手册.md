# T20 GCU环境配置成功经验手册

## 📋 概述

本文档记录了在燧原T20服务器上成功配置torch-gcu深度学习训练环境的完整过程，包括遇到的所有问题、解决方案和最佳实践。适用于DINOv3等深度学习模型的训练环境配置。

## 🏗️ 基础环境信息

### 硬件环境
- **服务器**: 燧原T20集群
- **计算卡**: 燧原GCU (Graphics Compute Unit)
- **容器**: dinov3_trainer (基于TopsRider软件栈)

### 软件环境
- **操作系统**: Linux (容器内)
- **Python版本**: 3.8.10
- **PyTorch版本**: 1.10.0
- **torch-gcu版本**: 1.10.0-2.5.136
- **TopsRider软件栈**: 已预装

## 🚨 常见问题与解决方案

### 问题1: torch.gcu.device_count() 返回 "GCU not available"

**现象**:
```python
>>> import torch
>>> torch.gcu.device_count()
RuntimeError: GCU not available
```

**根本原因**: 环境变量配置不完整，缺少关键的PYTHONPATH设置

**解决方案**:
```bash
# 设置完整的环境变量
export PATH=/opt/tops/bin:$PATH
export LD_LIBRARY_PATH=/opt/tops/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:$PYTHONPATH
```

### 问题2: 设备类型错误 - 使用错误的设备创建方式

**现象**:
```python
>>> device = torch.device("cuda:0")
RuntimeError: Expected one of cpu, cuda, xla, ... device type at start of device string
```

**根本原因**: 燧原T20使用XLA设备，不是CUDA设备

**正确的设备创建方式**:
```python
# ❌ 错误方式
device = torch.device("cuda:0")
device = torch.device("gcu:0")

# ✅ 正确方式
device = torch.device("xla:0")
```

### 问题3: ptex模块安装路径问题

**现象**:
```bash
pip install /workspace/ptex-1.3.20-py3-none-any.whl
ERROR: Could not find a version that satisfies the requirement
```

**根本原因**: ptex wheel包路径不正确

**解决方案**:
```bash
# 1. 搜索正确的ptex路径
find /usr/local/topsrider -name '*ptex*.whl' -type f

# 2. 使用找到的正确路径安装
pip install /usr/local/topsrider/ai_development_toolkit/distributed/ptex-1.3.20-py3-none-any.whl
```

### 问题4: torch_gcu模块导入但设备不可用

**现象**:
```python
>>> import torch_gcu
>>> torch_gcu.is_available()
True
>>> torch.gcu.device_count()
RuntimeError: GCU not available
```

**根本原因**: torch_gcu和torch.gcu是不同的接口，需要使用正确的API

**解决方案**:
```python
# ✅ 正确的检查方式
import torch_gcu
print("torch_gcu可用:", torch_gcu.is_available())

# ✅ 正确的设备创建
device = torch.device("xla:0")
```

## ✅ 完整配置流程

### 步骤1: 环境变量配置

在容器内执行以下命令设置环境变量：

```bash
docker exec -it dinov3_trainer bash -c "
export PATH=/opt/tops/bin:\$PATH &&
export LD_LIBRARY_PATH=/opt/tops/lib:\$LD_LIBRARY_PATH &&
export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:\$PYTHONPATH &&
echo 'PATH='$PATH &&
echo 'LD_LIBRARY_PATH='$LD_LIBRARY_PATH &&
echo 'PYTHONPATH='$PYTHONPATH
"
```

### 步骤2: 验证torch-gcu基础功能

```bash
docker exec -it dinov3_trainer bash -c "
export PATH=/opt/tops/bin:\$PATH &&
export LD_LIBRARY_PATH=/opt/tops/lib:\$LD_LIBRARY_PATH &&
export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:\$PYTHONPATH &&
python3 -c 'import torch_gcu; print("torch_gcu可用:", torch_gcu.is_available())'
"
```

**预期输出**:
```
torch_gcu可用: True
```

### 步骤3: 安装ptex模块

```bash
docker exec -it dinov3_trainer bash -c "
export PATH=/opt/tops/bin:\$PATH &&
export LD_LIBRARY_PATH=/opt/tops/lib:\$LD_LIBRARY_PATH &&
export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:\$PYTHONPATH &&
pip install /usr/local/topsrider/ai_development_toolkit/distributed/ptex-1.3.20-py3-none-any.whl
"
```

### 步骤4: 完整环境验证

```bash
docker exec -it dinov3_trainer bash -c "
export PATH=/opt/tops/bin:\$PATH &&
export LD_LIBRARY_PATH=/opt/tops/lib:\$LD_LIBRARY_PATH &&
export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:\$PYTHONPATH &&
python3 -c '
import torch
import torch_gcu
import ptex

print(\"1. torch_gcu可用:\", torch_gcu.is_available())
print(\"2. ptex模块导入成功\")

# 创建XLA设备
device = torch.device(\"xla:0\")
print(\"3. XLA设备:\", device)

# 创建张量并移动到XLA设备
tensor = torch.randn(2, 3)
print(\"4. CPU张量:\", tensor)

tensor_xla = tensor.to(device)
print(\"5. XLA张量:\", tensor_xla)
print(\"6. XLA张量设备:\", tensor_xla.device)

# 在XLA设备上进行计算
result = tensor_xla * 2 + 1
print(\"7. XLA计算结果:\", result)

# 测试矩阵运算
matrix1 = torch.randn(3, 4).to(device)
matrix2 = torch.randn(4, 2).to(device)
matmul_result = torch.matmul(matrix1, matrix2)
print(\"8. 矩阵乘法结果:\", matmul_result)

print(\"\\n=== torch-gcu环境配置完全成功！ ===\")
'
"
```

**预期输出示例**:
```
1. torch_gcu可用: True
2. ptex模块导入成功
3. XLA设备: xla:0
4. CPU张量: tensor([[-1.4660, -0.6265,  0.6715], [-0.0447,  0.6810, -1.1726]])
5. XLA张量: tensor([[-1.4660, -0.6265,  0.6715], [-0.0447,  0.6810, -1.1726]], device='xla:0')
6. XLA张量设备: xla:0
7. XLA计算结果: tensor([[-1.9319, -0.2531,  2.3430], [ 0.9106,  2.3619, -1.3452]], device='xla:0')
8. 矩阵乘法结果: tensor([[ 0.5804, -4.9953], [ 2.4586, -1.1838], [ 1.9963, -1.3844]], device='xla:0')

=== torch-gcu环境配置完全成功！ ===
```

## 🔧 关键配置要点

### 1. 环境变量配置

**必须设置的三个环境变量**:
```bash
export PATH=/opt/tops/bin:$PATH
export LD_LIBRARY_PATH=/opt/tops/lib:$LD_LIBRARY_PATH  
export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:$PYTHONPATH
```

**重要说明**:
- `PATH`: 添加TopsRider工具链路径
- `LD_LIBRARY_PATH`: 添加动态链接库路径
- `PYTHONPATH`: **最关键** - 添加torch-gcu模块路径

### 2. 设备使用规范

**燧原T20 GCU设备使用规范**:
```python
# ✅ 正确的设备创建
device = torch.device("xla:0")

# ✅ 张量移动到GCU设备
tensor_gcu = tensor.to(device)

# ✅ 检查设备类型
print(tensor_gcu.device)  # 输出: xla:0
```

### 3. 模块导入顺序

**推荐的导入顺序**:
```python
import torch
import torch_gcu  # 必须在torch之后导入
import ptex       # 可选，用于分布式训练
```

## 🚀 训练环境配置

### DINOv3训练配置适配

**配置文件关键修改** (`configs/train_dinov3_loveda_t20_gcu.py`):

```python
# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,  # GCU环境关闭cudnn
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo')  # 使用gloo后端适配GCU
)

# 批次大小适配GCU内存
train_dataloader = dict(
    batch_size=4,  # 适配T20 GCU内存限制
    num_workers=4,
    persistent_workers=True,
    # ...
)
```

### 训练启动脚本适配

**启动脚本关键配置**:
```bash
# 设置环境变量
export PATH=/opt/tops/bin:$PATH
export LD_LIBRARY_PATH=/opt/tops/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/tops/lib/python3.8/site-packages:$PYTHONPATH

# 使用torch.distributed.launch启动
python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    tools/train.py \
    configs/train_dinov3_loveda_t20_gcu.py \
    --launcher pytorch
```

## 🔍 故障排查指南

### 1. 环境变量检查

```bash
# 检查环境变量是否正确设置
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"  
echo "PYTHONPATH: $PYTHONPATH"

# 验证关键路径存在
ls -la /opt/tops/bin/
ls -la /opt/tops/lib/
ls -la /opt/tops/lib/python3.8/site-packages/
```

### 2. 模块导入检查

```python
# 逐步检查模块导入
try:
    import torch
    print("✅ torch导入成功")
except ImportError as e:
    print("❌ torch导入失败:", e)

try:
    import torch_gcu
    print("✅ torch_gcu导入成功")
    print("torch_gcu可用:", torch_gcu.is_available())
except ImportError as e:
    print("❌ torch_gcu导入失败:", e)

try:
    import ptex
    print("✅ ptex导入成功")
except ImportError as e:
    print("❌ ptex导入失败:", e)
```

### 3. 设备创建检查

```python
# 测试设备创建
try:
    device = torch.device("xla:0")
    print("✅ XLA设备创建成功:", device)
    
    # 测试张量操作
    tensor = torch.randn(2, 2).to(device)
    print("✅ 张量移动到XLA设备成功")
    print("张量设备:", tensor.device)
    
except Exception as e:
    print("❌ XLA设备操作失败:", e)
```

## 📝 最佳实践总结

### 1. 环境配置最佳实践

- **永远先设置环境变量**: 在任何torch-gcu操作之前
- **使用完整路径**: 避免相对路径导致的问题
- **验证每个步骤**: 逐步验证而不是一次性配置

### 2. 代码编写最佳实践

- **使用XLA设备**: `torch.device("xla:0")`而不是CUDA
- **检查设备可用性**: 使用`torch_gcu.is_available()`
- **渐进式测试**: 从简单张量操作开始测试

### 3. 调试最佳实践

- **保存日志**: 记录每次配置的输出结果
- **分步验证**: 不要跳过任何验证步骤
- **环境隔离**: 在干净的容器环境中测试

## 🎯 成功标志

当看到以下输出时，说明环境配置完全成功：

```
torch_gcu可用: True
XLA设备: xla:0
XLA张量设备: xla:0
XLA计算结果: tensor([[...]], device='xla:0')
矩阵乘法结果: tensor([[...]], device='xla:0')

=== torch-gcu环境配置完全成功！ ===
环境已准备就绪，可以开始DINOv3训练！
```

## 📚 相关文档

- [T20集群TopsRider软件栈环境配置成功手册.md](./T20集群TopsRider软件栈环境配置成功手册.md)
- [T20环境问题修复指南.md](./T20环境问题修复指南.md)
- [燧原T20适配指导.md](./燧原T20适配指导.md)

## 🏷️ 版本信息

- **文档版本**: v1.0
- **创建日期**: 2025-01-19
- **适用环境**: 燧原T20 + TopsRider软件栈
- **验证状态**: ✅ 已验证通过

---

**注意**: 本文档基于实际配置成功经验编写，所有命令和配置都已经过验证。如遇到问题，请按照故障排查指南逐步检查。