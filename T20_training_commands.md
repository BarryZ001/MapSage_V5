# T20服务端训练命令指导

## 数据集路径配置

根据您提供的服务端数据集路径信息，已更新配置文件：

### 数据集路径映射
- **LoveDA数据集**: `/workspace/data/loveda`
- **MMRS1M数据集**: `/workspace/data/mmrs1m/data`

### 已更新的配置文件
- `configs/train_dinov3_mmrs1m_t20_gcu.py` - MMRS1M训练配置（已更新）
- `configs/train_dinov3_loveda_t20_gcu.py` - LoveDA训练配置（已配置正确路径）

## T20服务端训练步骤

### 1. 拉取最新代码
```bash
cd /workspace/MapSage_V5
git pull origin main
```

### 2. 验证数据集路径
```bash
# 检查LoveDA数据集
ls -la /workspace/data/loveda/

# 检查MMRS1M数据集
ls -la /workspace/data/mmrs1m/data/
```

### 3. 运行MMRS1M训练（阶段一）
```bash
cd /workspace/MapSage_V5
python scripts/train.py configs/train_dinov3_mmrs1m_t20_gcu.py
```

### 4. 运行LoveDA微调训练（阶段二）
```bash
cd /workspace/MapSage_V5
python scripts/train.py configs/train_dinov3_loveda_t20_gcu.py
```

## 修复内容总结

### 1. 数据结构兼容性修复
- 修复了 `LoadImageFromFile` 和 `LoadAnnotations` 类的数据结构兼容性问题
- 支持传统格式（`img_info`、`ann_info`）和新格式（`img_path`、`seg_map_path`）
- 添加了智能格式检测和错误处理

### 2. 数据集路径配置
- 更新了训练配置文件中的数据集路径为服务端路径
- 保留了本地开发路径配置用于本地调试

### 3. 测试验证
- 创建并运行了数据加载测试脚本，验证修复效果
- 所有测试用例均通过，确保数据加载正常工作

## 注意事项

1. **环境检查**: 确保T20服务端环境已正确配置MMSegmentation和相关依赖
2. **权重文件**: 确保预训练权重文件在正确路径 `/workspace/weights/`
3. **工作目录**: 训练过程中会在 `./work_dirs/` 下创建相应的工作目录
4. **监控训练**: 可通过TensorBoard查看训练进度和指标

## 故障排除

如果遇到路径相关错误：
1. 检查数据集路径是否存在
2. 确认权限设置正确
3. 验证配置文件中的路径设置

如果遇到数据加载错误：
1. 运行测试脚本验证数据结构
2. 检查数据文件格式和完整性
3. 确认transform pipeline配置正确