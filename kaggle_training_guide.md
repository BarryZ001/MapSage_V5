# Kaggle P100 知识蒸馏训练指南

## 数据集路径配置

### 已配置的Kaggle数据集路径：

1. **LoveDA数据集**：`/kaggle/input/loveda`
   - 训练集：`Train/Rural/images_png` 和 `Train/Rural/masks_png`
   - 验证集：`Val/Rural/images_png` 和 `Val/Rural/masks_png`

2. **DINOv3预训练权重**：`/kaggle/input/dinov3-vitl16-pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`

3. **mIoU=84.96的训练权重**：`/kaggle/input/mapsage-stage02-checkpoint-6000/best_mIoU_iter_6000.pth`

## 可用的配置文件

### 1. 完整知识蒸馏配置 (推荐)
```bash
# 使用DINOv3作为教师模型，SegFormer-B0作为学生模型
python tools/train.py configs/train_distill_dinov3_true_kd.py
```
**特点**：
- 真正的知识蒸馏架构
- DINOv3教师模型完全冻结
- 多层特征对齐
- 批量大小：2（适合P100 16GB）
- 训练15000次迭代

### 2. 简化知识蒸馏配置
```bash
# 基于已训练权重的继续训练
python tools/train.py configs/train_distill_dinov3_simple_kd.py
```
**特点**：
- 从mIoU=84.96权重开始
- 标准MMSeg架构
- 批量大小：3
- 训练18000次迭代

### 3. P100优化配置
```bash
# 专门为P100 GPU优化的配置
python tools/train.py configs/train_distill_dinov3_kaggle_p100.py
```
**特点**：
- 最大化P100性能
- 批量大小：4
- 训练20000次迭代
- 更高学习率

## 内存优化设置

所有配置都包含以下P100优化：
- 裁剪尺寸：512×512（降低内存使用）
- 混合精度训练（FP16）
- 梯度裁剪
- 优化的数据加载器

## 训练监控

- 日志间隔：每50次迭代
- 验证间隔：每1500-2000次迭代
- 检查点保存：每1500-2000次迭代
- 最佳模型保存：基于mIoU指标

## 预期结果

- 基线mIoU：84.96%
- 目标：通过知识蒸馏提升边界检测精度
- 模型大小：显著减少（SegFormer-B0 vs DINOv3）
- 推理速度：大幅提升

## 故障排除

1. **内存不足**：减少批量大小或裁剪尺寸
2. **权重加载失败**：检查Kaggle数据集是否正确挂载
3. **训练不收敛**：调整学习率或蒸馏损失权重

## 后续步骤

训练完成后，最佳模型将保存在 `/kaggle/working/work_dirs/` 目录下，可用于：
- 模型推理和评估
- 运营管理系统集成
- 进一步的模型优化