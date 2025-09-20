# T20服务器Horovod安装与配置指南

## 问题诊断

根据错误日志分析，T20服务器上出现以下问题：

```
RuntimeError: Horovod is not available. Please install with: pip install horovod[pytorch]
```

这表明Horovod未正确安装或配置。

## 解决方案

### 方案一：安装Horovod（推荐）

#### 1. 使用自动安装脚本

```bash
# 在T20服务器上执行
cd /workspace/code/MapSage_V5
bash scripts/install_horovod_t20.sh
```

#### 2. 手动安装步骤

```bash
# 1. 更新系统包
apt-get update

# 2. 安装系统依赖
apt-get install -y \
    build-essential \
    cmake \
    libnccl2 \
    libnccl-dev \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    libgfortran5

# 3. 设置环境变量
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_GPU_OPERATIONS=NCCL

# 4. 安装Horovod
pip3 install --no-cache-dir horovod[pytorch]

# 5. 验证安装
python3 -c "import horovod.torch as hvd; print('Horovod version:', hvd.__version__)"
horovodrun --check-build
```

#### 3. 验证MPI环境

```bash
# 检查MPI安装
which mpirun
mpirun --version

# 测试MPI通信
mpirun -np 2 python3 -c "import horovod.torch as hvd; hvd.init(); print(f'Rank {hvd.rank()}/{hvd.size()}')"
```

#### 4. 重新运行Horovod训练

```bash
# 使用官方启动脚本
bash scripts/run_horovod_training.sh \
    --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --work-dir ./work_dirs/horovod_training \
    --slots-per-node 8 \
    --total-processes 8 \
    --amp

# 或直接使用mpirun
mpirun --hostfile /etc/mpi/hostfile --allow-run-as-root -mca btl ^openib -np 8 \
    python3 scripts/train_distributed_horovod_8card_gcu.py \
    configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --work-dir ./work_dirs/horovod_training \
    --amp
```

### 方案二：使用PyTorch DDP（备选方案）

如果Horovod安装困难，可以使用原生PyTorch DDP：

#### 1. 使用PyTorch DDP启动脚本

```bash
# 单节点8卡训练
bash scripts/run_pytorch_ddp_training.sh \
    --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --work-dir ./work_dirs/pytorch_ddp_training \
    --amp

# 多节点训练（节点0）
bash scripts/run_pytorch_ddp_training.sh \
    --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr 192.168.1.100 \
    --amp

# 多节点训练（节点1）
bash scripts/run_pytorch_ddp_training.sh \
    --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr 192.168.1.100 \
    --amp
```

#### 2. 直接使用torch.distributed.launch

```bash
python3 -m torch.distributed.launch \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train_distributed_pytorch_ddp_8card_gcu.py \
    configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --work-dir ./work_dirs/pytorch_ddp_training \
    --amp
```

## 常见问题排查

### 1. NCCL相关错误

```bash
# 检查NCCL库
ldconfig -p | grep nccl

# 如果缺少NCCL，安装：
apt-get install -y libnccl2 libnccl-dev
```

### 2. MPI相关错误

```bash
# 检查OpenMPI安装
dpkg -l | grep openmpi

# 重新安装OpenMPI
apt-get remove --purge openmpi-*
apt-get install -y libopenmpi-dev openmpi-bin openmpi-common
```

### 3. 权限问题

```bash
# 如果遇到权限错误，确保以root身份运行
sudo bash scripts/install_horovod_t20.sh
```

### 4. 环境变量设置

将以下环境变量添加到 `~/.bashrc` 或 `~/.profile`：

```bash
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_GPU_OPERATIONS=NCCL
```

## 性能对比

| 方案 | 优势 | 劣势 |
|------|------|------|
| Horovod | 更好的多节点扩展性，支持更多后端 | 安装复杂，依赖较多 |
| PyTorch DDP | 安装简单，PyTorch原生支持 | 多节点配置稍复杂 |

## 推荐使用顺序

1. **首选**：修复Horovod安装问题，使用现有的Horovod方案
2. **备选**：如果Horovod问题难以解决，切换到PyTorch DDP方案
3. **临时**：单卡训练进行功能验证

## 联系支持

如果遇到问题，请提供以下信息：
- 错误日志完整内容
- 系统环境信息（`uname -a`, `python3 --version`, `pip3 list`）
- GCU驱动版本信息