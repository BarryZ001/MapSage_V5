# T20集群DINOv3 8卡分布式训练启动指导

## 📋 概述

本指导文档说明如何在T20集群上启动DINOv3+MMRS-1M的8卡分布式训练。

## 🎯 训练配置

- **模型**: DINOv3-ViT-L/16 backbone + VisionTransformerUpHead
- **数据集**: MMRS-1M多模态遥感数据集
- **设备**: 8卡燧原T20 GCU
- **分布式后端**: eccl (燧原专用)
- **预计训练时间**: 5-7天

## 🚀 启动步骤

### 1. 环境准备

```bash
# 登录T20服务器
ssh -p 60026 root@117.156.108.234

# 进入工作目录
cd /workspace/MapSage_V5

# 拉取最新代码
git pull origin main
```

### 2. 检查环境依赖

```bash
# 检查torch_gcu是否可用
python3 -c "import torch_gcu; print(f'GCU设备数: {torch_gcu.device_count()}')"

# 检查分布式后端支持
python3 -c "import torch.distributed as dist; print('分布式支持正常')"

# 检查数据集路径
ls -la /workspace/data/mmrs1m/
```

### 3. 启动8卡分布式训练

有两种启动方式可选：

#### 方式1: 后台运行（推荐）
```bash
# 后台启动，日志输出到文件
bash scripts/start_8card_training.sh

# 监控训练进度
tail -f ./work_dirs/dinov3_mmrs1m_t20_gcu_8card/logs/train_rank_0.log
```

#### 方式2: 前台运行（调试用）
```bash
# 前台启动，直接显示输出
bash scripts/start_8card_training_interactive.sh
```

### 4. 监控训练状态

```bash
# 查看所有训练进程
ps aux | grep train_distributed_8card_gcu

# 查看GPU使用情况
nvidia-smi  # 或对应的GCU监控命令

# 查看训练日志
ls -la ./work_dirs/dinov3_mmrs1m_t20_gcu_8card/logs/
tail -f ./work_dirs/dinov3_mmrs1m_t20_gcu_8card/logs/train_rank_0.log
```

### 5. 停止训练（如需要）

```bash
# 停止所有训练进程
pkill -f train_distributed_8card_gcu.py

# 或使用脚本停止
bash scripts/stop_distributed_training.sh
```

## 📁 关键文件说明

- **配置文件**: `configs/train_dinov3_mmrs1m_t20_gcu_8card.py`
- **训练脚本**: `scripts/train_distributed_8card_gcu.py`
- **启动脚本**: `scripts/start_8card_training.sh`
- **工作目录**: `./work_dirs/dinov3_mmrs1m_t20_gcu_8card/`

## ⚙️ 关键配置参数

### 分布式配置
- **后端**: eccl (燧原T20专用)
- **设备数**: 8个GCU
- **批次大小**: 2 per GPU × 8 GPUs = 16 (total)

### 训练参数
- **学习率**: 1e-4 (8卡训练优化)
- **最大迭代数**: 80,000
- **验证间隔**: 2000 iterations
- **图像尺寸**: 512×512

## 🚨 常见问题排查

### 1. torch_gcu导入失败
```bash
# 检查torch_gcu安装
pip list | grep torch

# 重新安装torch_gcu（如需要）
pip install torch_gcu
```

### 2. 分布式初始化失败
```bash
# 检查环境变量
echo $WORLD_SIZE
echo $MASTER_ADDR
echo $MASTER_PORT

# 检查端口是否被占用
netstat -tulpn | grep 29500
```

### 3. 数据集路径错误
```bash
# 检查数据集路径
ls -la /workspace/data/mmrs1m/
du -sh /workspace/data/mmrs1m/

# 修改配置文件中的数据路径（如需要）
vim configs/train_dinov3_mmrs1m_t20_gcu_8card.py
```

### 4. 内存不足
```bash
# 检查系统内存
free -h

# 减小批次大小（修改配置文件）
# batch_size = 1  # 从2改为1
```

## 📊 预期输出

### 训练日志示例
```
✅ torch_gcu导入成功，可用设备数: 8
🌍 分布式训练参数:
  - WORLD_SIZE: 8
  - RANK: 0
  - LOCAL_RANK: 0
🔧 初始化分布式进程组:
  - Backend: eccl
  - Init method: env://
✅ 分布式进程组初始化成功
🔧 设置当前进程GCU设备: 0
📁 工作目录: ./work_dirs/dinov3_mmrs1m_t20_gcu_8card
🚀 启动训练 - Rank 0/8
```

### 文件结构
```
work_dirs/dinov3_mmrs1m_t20_gcu_8card/
├── logs/
│   ├── train_rank_0.log
│   ├── train_rank_1.log
│   └── ...
├── checkpoints/
│   ├── latest.pth
│   └── best_mIoU_iter_*.pth
└── train_dinov3_mmrs1m_t20_gcu_8card.py
```

## 📞 支持资源

- **燧原文档中心**: https://support.enflame-tech.com/documents/
- **PyTorch使用指南**: 查看燧原官方PyTorch适配文档
- **调试相关**: 查看燧原官方调试指南

## 🔄 更新代码

如果训练过程中发现问题需要修复：

```bash
# 停止当前训练
pkill -f train_distributed_8card_gcu.py

# 拉取最新代码
git pull origin main

# 重新启动训练
bash scripts/start_8card_training.sh
```

---

**注意**: 如果遇到任何问题，请记录详细的错误日志，以便进行针对性修复。