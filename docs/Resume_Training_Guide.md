# MapSage V5 恢复训练指南

## 概述

本指南详细说明如何在Kaggle环境中从检查点恢复MapSage V5模型的训练。这对于在有时间限制的Kaggle会话中继续长时间训练至关重要。

## 检查点文件说明

### 关键文件类型

1. **`iter_XXXX.pth`** (例如: `iter_8000.pth`)
   - **内容**: 完整的训练状态快照
   - **包含**: 模型权重 + 优化器状态 + 学习率调度器状态 + 训练迭代计数
   - **用途**: 无缝恢复训练，就像从未中断过一样
   - **重要性**: ⭐⭐⭐⭐⭐ (恢复训练必需)

2. **`best_mIoU_iter_XXXX.pth`** (例如: `best_mIoU_iter_8000.pth`)
   - **内容**: 验证集上表现最佳的模型权重
   - **包含**: 仅模型权重，不包含训练状态
   - **用途**: 模型评估和部署
   - **重要性**: ⭐⭐⭐⭐ (保存最佳成果)

3. **`latest.pth`**
   - **内容**: 指向最新检查点的符号链接
   - **用途**: 快速访问最新状态

## 保存检查点的最佳实践

### 在Kaggle中保存

1. **使用Commit功能**:
   ```
   点击右上角 "Save Version" → 选择 "Save & Run All (Commit)"
   ```

2. **验证保存内容**:
   - 确保 `/kaggle/working/MapSage_V5/work_dirs/` 目录被完整保存
   - 检查所有 `.pth` 文件都在输出中

3. **获取输出路径**:
   - Commit完成后，在 "Data" → "Output" 中找到文件
   - 记录完整的Kaggle输入路径 (例如: `/kaggle/input/your-commit-name/`)

## 恢复训练步骤

### 第一步: 准备新的Kaggle会话

1. **挂载必要数据**:
   - GitHub仓库 (包含代码)
   - EarthVQA数据集
   - **上一次训练的输出** (包含检查点)

2. **安装环境**:
   ```bash
   !pip install -U openmim
   !mim install mmengine
   !mim install "mmcv>=2.0.0"
   !pip install "mmsegmentation>=1.0.0"
   ```

### 第二步: 确认检查点路径

1. **查找检查点文件**:
   ```bash
   !find /kaggle/input -name "iter_*.pth" -type f
   ```

2. **验证文件完整性**:
   ```bash
   !ls -la /kaggle/input/your-commit-output/MapSage_V5/work_dirs/train_*/
   ```

### 第三步: 更新配置文件

如果检查点路径与预设不同，需要修改 `configs/resume_earthvqa_kaggle.py`:

```python
# 更新为实际的检查点路径
load_from = '/kaggle/input/your-actual-path/iter_8000.pth'
resume = True
```

### 第四步: 启动恢复训练

```bash
!python scripts/train.py configs/resume_earthvqa_kaggle.py
```

## 验证恢复成功

### 检查日志输出

恢复成功的标志:
```
load checkpoint from /kaggle/input/.../iter_8000.pth
resumed epoch: X, iter: 8000
resume optimizer: <class 'torch.optim.adamw.AdamW'>
resume lr_scheduler: <class 'mmengine.optim.scheduler.lr_scheduler.PolyLR'>
```

### 验证训练继续

- 训练应该从第8001次迭代开始
- 学习率应该延续之前的调度
- 验证应该在第12000次迭代进行 (下一个4000的倍数)

## 常见问题和解决方案

### 问题1: 找不到检查点文件

**症状**: `FileNotFoundError: No such file or directory`

**解决方案**:
1. 检查Kaggle输入路径是否正确挂载
2. 验证文件路径拼写
3. 确认上一次Commit是否成功

### 问题2: 检查点损坏

**症状**: `RuntimeError: Error(s) in loading state_dict`

**解决方案**:
1. 尝试使用 `best_mIoU_iter_8000.pth` 并设置 `resume = False`
2. 检查模型配置是否与训练时一致

### 问题3: 内存不足

**症状**: `CUDA out of memory`

**解决方案**:
1. 减少 `batch_size` 从4到2
2. 启用梯度检查点: `gradient_checkpointing=True`

## 最佳实践建议

### 训练策略

1. **定期保存**: 每4000次迭代自动保存
2. **多重备份**: 保存多个检查点以防损坏
3. **监控验证**: 关注mIoU趋势，避免过拟合

### 时间管理

1. **预留时间**: 为保存和提交预留至少10分钟
2. **分阶段训练**: 将40000次迭代分为多个会话
3. **及时提交**: 不要等到会话即将结束才保存

### 文件管理

1. **命名规范**: 使用有意义的Commit名称
2. **版本控制**: 记录每次训练的配置变更
3. **清理空间**: 删除不必要的中间文件

## 示例命令序列

```bash
# 1. 查找检查点
!find /kaggle/input -name "iter_8000.pth" -type f

# 2. 验证路径
!ls -la /kaggle/input/temp-8000tier/

# 3. 启动恢复训练
!python scripts/train.py configs/resume_earthvqa_kaggle.py

# 4. 监控训练进度
!tail -f /kaggle/working/MapSage_V5/work_dirs/resume_earthvqa_kaggle_*/vis_data/scalars.json
```

## 总结

恢复训练是深度学习项目中的关键技能。通过正确使用检查点机制，您可以:

- ✅ 在有限的Kaggle会话中完成长时间训练
- ✅ 避免因意外中断而丢失训练进度
- ✅ 灵活调整训练策略和超参数
- ✅ 确保训练的连续性和一致性

记住：**`iter_8000.pth` 是您恢复训练的关键文件，`resume = True` 是启用恢复模式的关键参数。**