# T20服务器DINOv3+MMRS-1M训练部署指南

## 📋 概述

本指南详细说明如何在T20服务器上部署和启动DINOv3+MMRS-1M阶段一基础模型训练。

## 🎯 训练目标

- **模型**: DINOv3-ViT-L/16 backbone + VisionTransformerUpHead
- **数据集**: MMRS-1M多模态遥感数据集
- **任务**: 多模态图像分类和分割基础模型训练
- **预计时间**: 5-7天（8卡A100）
- **目标**: 为后续指令跟随和多模态理解奠定基础

## 🚀 部署步骤

### 1. 环境准备

```bash
# 登录T20服务器
ssh username@t20-server

# 激活conda环境
conda activate mmseg

# 检查GPU状态
nvidia-smi
```

### 2. 代码部署

```bash
# 创建项目目录
mkdir -p /workspace/MapSage_V5
cd /workspace/MapSage_V5

# 上传代码（从本地）
# 使用scp或rsync上传整个项目
scp -r /Users/barryzhang/myDev3/MapSage_V5/* username@t20-server:/workspace/MapSage_V5/

# 或者使用git clone（如果代码已推送到仓库）
# git clone <repository_url> .
```

### 3. 数据集验证

```bash
# 检查MMRS-1M数据集
ls -la /workspace/data/mmrs1m/

# 应该包含以下目录：
# - caption/
# - classification/
# - detection/
# - json/
# - RSVG/
# - VQA/

# 检查数据集大小
du -sh /workspace/data/mmrs1m/
```

### 4. 预训练权重准备

```bash
# 创建权重目录
mkdir -p /workspace/weights/

# 下载DINOv3预训练权重（如果不存在）
wget -O /workspace/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

# 或使用遥感预训练权重（推荐）
# 需要从相关论文或项目获取
```

### 5. 环境验证

```bash
# 运行环境验证脚本
cd /workspace/MapSage_V5
python scripts/validate_training_env.py

# 确保以下检查通过：
# ✅ Python版本
# ✅ PyTorch + CUDA
# ✅ MMSegmentation
# ✅ 自定义模块导入
# ✅ 数据路径存在
# ✅ GPU可用
```

### 6. 启动训练

```bash
# 方法1: 使用训练脚本（推荐）
bash scripts/train_dinov3_mmrs1m.sh

# 方法2: 直接使用MMSegmentation训练命令
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py \
    configs/train_dinov3_mmrs1m.py \
    --work-dir ./work_dirs/dinov3_mmrs1m_stage1 \
    --launcher pytorch \
    --seed 42 \
    --deterministic
```

## 📊 训练监控

### 实时监控

```bash
# 查看训练日志
tail -f work_dirs/dinov3_mmrs1m_stage1/20240101_120000.log

# 监控GPU使用情况
watch -n 1 nvidia-smi

# 查看训练进度
ls -la work_dirs/dinov3_mmrs1m_stage1/
```

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=work_dirs/dinov3_mmrs1m_stage1/tf_logs --port=6006

# 在浏览器中访问：http://t20-server:6006
```

## 🔧 配置说明

### 关键训练参数

- **批次大小**: 8 (per GPU) × 8 (GPUs) × 2 (accumulation) = 128 (effective)
- **学习率**: 1e-4 (base) with warmup
- **训练步数**: 80,000 iterations
- **验证间隔**: 2000 iterations
- **图像尺寸**: 512×512
- **优化器**: AdamW with weight decay 0.05

### 数据配置

- **数据根目录**: `/workspace/data/mmrs1m`
- **任务类型**: classification (阶段一)
- **模态**: optical (主要)
- **指令格式**: 启用

### 模型配置

- **Backbone**: DINOv3-ViT-L/16
- **嵌入维度**: 1024
- **层数**: 24
- **注意力头数**: 16
- **预训练权重**: DINOv2官方权重

## 📈 预期结果

### 训练指标

- **分类准确率**: >85% (MMRS-1M分类任务)
- **收敛时间**: 3-4天
- **内存使用**: ~40GB per GPU
- **训练速度**: ~2-3 iterations/second

### 输出文件

```
work_dirs/dinov3_mmrs1m_stage1/
├── train_dinov3_mmrs1m.py          # 训练配置
├── latest.pth                      # 最新权重
├── best_mIoU_iter_*.pth           # 最佳权重
├── 20240101_120000.log            # 训练日志
└── tf_logs/                       # TensorBoard日志
```

## 🚨 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   # 在配置文件中修改 batch_size = 4
   ```

2. **数据加载错误**
   ```bash
   # 检查数据路径
   ls -la /workspace/data/mmrs1m/
   # 检查权限
   chmod -R 755 /workspace/data/mmrs1m/
   ```

3. **分布式训练失败**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep 29500
   # 更换端口
   export MASTER_PORT=29501
   ```

4. **权重加载失败**
   ```bash
   # 检查权重文件
   ls -la /workspace/weights/
   # 重新下载权重
   wget -O /workspace/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth <url>
   ```

### 性能优化

1. **数据加载优化**
   - 增加 `num_workers=8`
   - 启用 `persistent_workers=True`
   - 使用SSD存储数据

2. **训练加速**
   - 启用混合精度训练 `fp16=True`
   - 使用编译优化 `compile_cfg`
   - 调整批次大小和累积步数

## 📝 检查清单

训练启动前确认：

- [ ] T20服务器环境正常
- [ ] 8张A100 GPU可用
- [ ] MMRS-1M数据集完整
- [ ] 预训练权重已下载
- [ ] 代码部署完成
- [ ] 环境验证通过
- [ ] 训练配置正确
- [ ] 监控工具就绪

## 🎯 下一步计划

阶段一训练完成后：

1. **模型评估**: 在验证集上评估性能
2. **权重保存**: 保存最佳模型权重
3. **阶段二准备**: 准备指令跟随训练数据
4. **多模态扩展**: 集成SAR和红外模态
5. **性能优化**: 根据结果调整超参数

## 📞 联系信息

如遇问题，请联系：
- 技术支持：[技术团队]
- 服务器管理：[运维团队]
- 项目负责人：[项目经理]

---

**注意**: 本指南基于T20服务器环境配置，其他环境可能需要相应调整。