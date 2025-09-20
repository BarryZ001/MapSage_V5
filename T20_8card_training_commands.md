# T20服务器8卡GCU训练命令指南

## 🚀 快速启动8卡训练

### 1. 环境准备
```bash
# 在T20服务器容器中执行
cd /workspace/code/MapSage_V5

# 修复DNS问题（如果需要）
python3 scripts/fix_dns_rdtypes_issue.py

# 验证环境
bash scripts/verify_t20_training_env.sh
```

### 2. 8卡分布式训练命令

#### 方法1: 使用启动脚本（推荐）
```bash
# 使用DINOv3 + LoveDA数据集
bash scripts/start_8card_training.sh configs/train_dinov3_loveda_t20_gcu.py

# 使用DINOv3 + MMRS1M数据集
bash scripts/start_8card_training.sh configs/train_dinov3_mmrs1m_t20_gcu_8card.py
```

#### 方法2: 直接使用torchrun
```bash
# DINOv3 + LoveDA (8卡)
torchrun --nproc_per_node=8 --master_port=29500 \
    scripts/train_distributed_8card_gcu.py \
    configs/train_dinov3_loveda_t20_gcu.py \
    --launcher pytorch

# DINOv3 + MMRS1M (8卡)
torchrun --nproc_per_node=8 --master_port=29500 \
    scripts/train_distributed_8card_gcu.py \
    configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --launcher pytorch
```

#### 方法3: 使用python -m torch.distributed.launch
```bash
# DINOv3 + LoveDA (8卡)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train_distributed_8card_gcu.py \
    configs/train_dinov3_loveda_t20_gcu.py \
    --launcher pytorch

# DINOv3 + MMRS1M (8卡)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train_distributed_8card_gcu.py \
    configs/train_dinov3_mmrs1m_t20_gcu_8card.py \
    --launcher pytorch
```

### 3. 单卡测试命令
```bash
# 单卡测试 - DINOv3 + LoveDA
python scripts/train.py configs/train_dinov3_loveda_t20_gcu.py

# 单卡测试 - DINOv3 + MMRS1M
python scripts/train.py configs/train_dinov3_mmrs1m_t20_gcu.py
```

### 4. 训练监控
```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f work_dirs/*/latest.log

# 查看训练进程
ps aux | grep python
```

### 5. 常用配置文件说明

| 配置文件 | 数据集 | 用途 |
|---------|--------|------|
| `train_dinov3_loveda_t20_gcu.py` | LoveDA | DINOv3骨干网络训练 |
| `train_dinov3_mmrs1m_t20_gcu.py` | MMRS1M | DINOv3骨干网络训练 |
| `train_dinov3_mmrs1m_t20_gcu_8card.py` | MMRS1M | 8卡分布式训练专用 |

### 6. 预训练权重路径
```bash
# 确保预训练权重存在
ls -la /workspace/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

### 7. 故障排除

#### 如果遇到DNS错误
```bash
python3 scripts/fix_dns_rdtypes_issue.py
```

#### 如果遇到torch_gcu设备错误
```bash
# 验证脚本已修复，重新验证环境
bash scripts/verify_t20_training_env.sh
```

#### 如果训练中断
```bash
# 查看最新的checkpoint
ls -la work_dirs/*/latest.pth

# 从checkpoint恢复训练
python scripts/train.py configs/your_config.py --resume-from work_dirs/*/latest.pth
```

### 8. 性能优化建议

1. **批次大小调整**: 根据显存使用情况调整batch_size
2. **学习率调整**: 8卡训练时学习率通常需要相应调整
3. **数据加载**: 确保数据加载不成为瓶颈
4. **混合精度**: 如果支持，可以启用混合精度训练

### 9. 训练完成后

```bash
# 查看训练结果
ls -la work_dirs/*/

# 运行验证
python scripts/validate.py configs/your_config.py work_dirs/*/latest.pth
```

## 📝 注意事项

1. 确保所有8张GCU卡都可用
2. 预训练权重路径正确
3. 数据集路径配置正确
4. 有足够的磁盘空间保存checkpoints
5. 网络连接稳定（如果需要下载数据）

## 🔧 环境要求

- Python 3.8+
- PyTorch 1.10.0+
- torch_gcu (燧原GCU支持)
- MMCV 1.6.0
- MMSegmentation 0.29.1
- 8张GCU卡可用