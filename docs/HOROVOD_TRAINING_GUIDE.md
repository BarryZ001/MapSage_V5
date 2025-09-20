# Horovod分布式训练指南

## 概述

本指南介绍如何使用基于OpenMPI+Horovod架构的分布式训练脚本，该架构符合TopsDL官方推荐。

## 架构对比

### 当前torchrun方案 vs TopsDL官方推荐方案

| 特性 | torchrun + DDP | OpenMPI + Horovod (官方推荐) |
|------|----------------|------------------------------|
| 通信后端 | NCCL/Gloo | MPI |
| 启动方式 | torchrun | mpirun |
| 设备标识 | cuda:N | dtu/xla:N |
| 配置文件 | hostfile (torchrun格式) | hostfile (MPI格式) |
| 梯度同步 | DDP AllReduce | Horovod AllReduce |
| 官方支持 | 社区方案 | TopsDL官方推荐 |

## 文件结构

```
scripts/
├── train_distributed_horovod_8card_gcu.py  # Horovod训练脚本
├── run_horovod_training.sh                 # 启动脚本
└── train_distributed_8card_gcu.py          # 原torchrun脚本
```

## 使用方法

### 1. 基本使用

```bash
# 使用默认配置（8卡训练）
./scripts/run_horovod_training.sh --config configs/mapsage/mapsage_gcu.py

# 指定工作目录
./scripts/run_horovod_training.sh \
    --config configs/mapsage/mapsage_gcu.py \
    --work-dir ./work_dirs/mapsage_horovod
```

### 2. 高级选项

```bash
# 启用自动混合精度和学习率缩放
./scripts/run_horovod_training.sh \
    --config configs/mapsage/mapsage_gcu.py \
    --work-dir ./work_dirs/mapsage_horovod \
    --amp \
    --auto-scale-lr

# 恢复训练
./scripts/run_horovod_training.sh \
    --config configs/mapsage/mapsage_gcu.py \
    --resume ./work_dirs/mapsage_horovod/epoch_10.pth

# 自定义进程数和slots
./scripts/run_horovod_training.sh \
    --config configs/mapsage/mapsage_gcu.py \
    --slots-per-node 4 \
    --total-processes 16
```

### 3. 多机训练

对于多机训练，需要配置`/etc/volcano/worker.host`文件：

```bash
# 示例worker.host内容
worker-node-1
worker-node-2
worker-node-3
```

启动脚本会自动读取该文件并生成相应的MPI hostfile。

## 技术特点

### 1. 设备兼容性

- **自动设备检测**: 支持CUDA和XLA设备
- **强制CPU初始化**: 避免MMEngine设备不匹配问题
- **手动设备迁移**: 确保模型参数正确迁移到目标设备

### 2. Horovod集成

- **参数广播**: 确保所有进程模型参数一致
- **梯度平均**: 使用Horovod的AllReduce进行梯度同步
- **学习率缩放**: 根据进程数自动调整学习率

### 3. 配置优化

- **数据采样**: 自动配置DistributedSampler
- **批次大小**: 可选的自动批次大小调整
- **日志控制**: 只在rank 0进程保存日志和检查点

## 官方文档参考

根据TopsDL用户使用手册，官方推荐的分布式训练命令格式：

```bash
mkdir -p /etc/mpi && \
echo "localhost slots=8" > /etc/mpi/hostfile && \
cat /etc/volcano/worker.host | sed "s/$/& slots=8/g" >> /etc/mpi/hostfile && \
mpirun --hostfile /etc/mpi/hostfile --allow-run-as-root -mca btl ^openib -np 32 \
python3 /workspace/algorithm/resnet50/train.py \
--data_dir=/workspace/dataset/imagenet \
--output_dir=/workspace/persistent-model \
--device=dtu --dataset=imagenet
```

我们的脚本完全遵循这一格式。

## 故障排除

### 1. 常见问题

**问题**: `ModuleNotFoundError: No module named 'horovod'`
**解决**: 安装Horovod
```bash
pip install horovod[pytorch]
```

**问题**: MPI相关错误
**解决**: 确保OpenMPI已正确安装
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# CentOS/RHEL
sudo yum install openmpi openmpi-devel
```

**问题**: 设备不匹配错误
**解决**: 检查设备配置，确保XLA或CUDA环境正确设置

### 2. 调试模式

启用详细日志：
```bash
export HOROVOD_LOG_LEVEL=DEBUG
export OMPI_MCA_plm_rsh_args="-v"
./scripts/run_horovod_training.sh --config configs/mapsage/mapsage_gcu.py
```

## 性能对比

建议在相同配置下对比两种方案的性能：

1. **torchrun方案**: `scripts/train_distributed_8card_gcu.py`
2. **Horovod方案**: `scripts/train_distributed_horovod_8card_gcu.py`

对比指标：
- 训练速度（samples/sec）
- 内存使用
- 收敛性能
- 稳定性

## 注意事项

1. **环境依赖**: 确保Horovod和OpenMPI正确安装
2. **设备配置**: 根据实际硬件调整slots和进程数
3. **网络配置**: 多机训练需要确保节点间网络连通
4. **资源管理**: 注意内存和计算资源的合理分配

## 参考文档

- [TopsDL训推一体化平台用户使用手册](./【原文】训推一体化平台TopsDL用户使用手册.md)
- [TopsDL白皮书](./【原文】训推一体化平台TopsDL白皮书.md)
- [Horovod官方文档](https://horovod.readthedocs.io/)
- [OpenMPI官方文档](https://www.open-mpi.org/doc/)