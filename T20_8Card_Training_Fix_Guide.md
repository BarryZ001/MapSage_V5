# T20服务器8卡分布式训练修复指南

## 问题描述

当前训练只使用了1张GCU卡（卡0），其他7张卡处于Sleep状态，没有参与分布式训练。

## 问题原因分析

1. **分布式启动方式不正确**: 当前可能使用单进程启动，没有正确启动8个进程
2. **环境变量配置不完整**: 缺少必要的分布式训练环境变量
3. **torchrun参数设置错误**: 没有正确配置`--nproc_per_node=8`

## 解决方案

### 1. 停止当前训练

```bash
# 在T20服务器上执行
pkill -f "train_distributed_8card_gcu.py"
```

### 2. 使用正确的启动脚本

已创建新的启动脚本：`start_8card_training_correct.sh`

```bash
# 在T20服务器上执行
cd /installer
./start_8card_training_correct.sh
```

### 3. 关键修复点

#### 3.1 正确的torchrun命令
```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    scripts/train_distributed_8card_gcu.py \
    configs/train_dinov3_mmrs1m_t20_gcu_8card.py
```

#### 3.2 必要的环境变量
```bash
export WORLD_SIZE=8
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
export ECCL_BACKEND=eccl
export ECCL_DEVICE_TYPE=gcu
```

#### 3.3 验证8卡启动
启动后应该看到：
- 8个Python进程（1个主进程 + 7个子进程）
- 所有8张GCU卡的利用率 > 0%
- 每个进程对应一个LOCAL_RANK (0-7)

## 验证步骤

### 1. 检查进程数量
```bash
ps aux | grep train_distributed_8card_gcu.py | wc -l
# 应该显示8个进程
```

### 2. 检查GCU设备状态
```bash
efsmi
# 所有8张卡的DEV-Util应该 > 0%
```

### 3. 检查训练日志
```bash
tail -f work_dirs/dinov3_mmrs1m_t20_gcu_8card_correct/logs/train.log
# 应该看到8个rank的初始化信息
```

## 预期结果

正确启动后，`efsmi`输出应该显示：
```
* 0      T20    | ... | 32G  Disable | 85.0%  W28001150301 *
* 1      T20    | ... | 32G  Disable | 85.0%  W28001110508 *
* 2      T20    | ... | 32G  Disable | 85.0%  W28001170208 *
* 3      T20    | ... | 32G  Disable | 85.0%  W28001180500 *
* 4      T20    | ... | 32G  Disable | 85.0%  W28001010806 *
* 5      T20    | ... | 32G  Disable | 85.0%  W28001250206 *
* 6      T20    | ... | 32G  Disable | 85.0%  W28001060610 *
* 7      T20    | ... | 32G  Disable | 85.0%  W28001150105 *
```

## 性能提升预期

- **训练速度**: 8卡并行，理论上提升7-8倍
- **批次大小**: 从2提升到16 (2 × 8卡)
- **收敛速度**: 更大批次大小通常带来更好的收敛

## 故障排除

### 如果仍然只有1张卡工作

1. **检查ECCL后端**:
```bash
python -c "import torch.distributed as dist; print('ECCL available:', 'eccl' in dist.Backend.__members__)"
```

2. **检查GCU设备数量**:
```bash
python -c "import torch_gcu; print('GCU count:', torch_gcu.device_count())"
```

3. **检查环境变量**:
```bash
env | grep -E "(WORLD_SIZE|RANK|LOCAL_RANK|MASTER_|ECCL_)"
```

### 如果出现设备错误

1. **重启容器**:
```bash
# 退出容器后重新进入
exit
# 重新启动容器
```

2. **清理进程**:
```bash
pkill -f python
pkill -f train_distributed
```

## 监控命令

```bash
# 实时监控GCU使用情况
watch -n 1 efsmi

# 监控进程状态
watch -n 1 "ps aux | grep train_distributed_8card_gcu.py"

# 监控训练日志
tail -f work_dirs/dinov3_mmrs1m_t20_gcu_8card_correct/logs/train.log
```

## 注意事项

1. **内存使用**: 8卡训练会显著增加内存使用，确保系统内存充足
2. **网络带宽**: 分布式训练需要进程间通信，确保网络稳定
3. **数据加载**: 增加`num_workers`以匹配8卡的数据需求

---

**创建时间**: 2025-09-21  
**适用环境**: 燧原T20 GCU × 8卡  
**状态**: 待验证