# T20服务器Git冲突解决指南

## 问题描述

在T20服务器上执行 `git pull origin main` 时遇到以下错误：

```bash
error: The following untracked working tree files would be overwritten by merge:
        scripts/start_distributed_manual.sh
        scripts/start_distributed_training.sh
        scripts/start_distributed_training_simple.sh
        scripts/stop_distributed_training.sh
        scripts/train_distributed_gcu.py
Please move or remove them before you merge.
Aborting
```

## 解决方案

### 方案一：使用自动化脚本（推荐）

1. **执行解决冲突脚本**：
   ```bash
   cd ~/mapsage_project/code/MapSage_V5
   ./scripts/resolve_git_conflicts_t20.sh
   ```

2. **执行git pull**：
   ```bash
   git pull origin main
   ```

### 方案二：手动解决

1. **备份冲突文件**：
   ```bash
   cd ~/mapsage_project/code/MapSage_V5
   
   # 创建备份目录
   mkdir -p backup_conflicting_files_$(date +%Y%m%d_%H%M%S)
   BACKUP_DIR="backup_conflicting_files_$(date +%Y%m%d_%H%M%S)"
   
   # 备份文件
   cp scripts/start_distributed_manual.sh $BACKUP_DIR/
   cp scripts/start_distributed_training.sh $BACKUP_DIR/
   cp scripts/start_distributed_training_simple.sh $BACKUP_DIR/
   cp scripts/stop_distributed_training.sh $BACKUP_DIR/
   cp scripts/train_distributed_gcu.py $BACKUP_DIR/
   ```

2. **移除冲突文件**：
   ```bash
   rm -f scripts/start_distributed_manual.sh
   rm -f scripts/start_distributed_training.sh
   rm -f scripts/start_distributed_training_simple.sh
   rm -f scripts/stop_distributed_training.sh
   rm -f scripts/train_distributed_gcu.py
   ```

3. **执行git pull**：
   ```bash
   git pull origin main
   ```

## 冲突文件说明

### 被覆盖的文件功能：

1. **start_distributed_manual.sh** - 手动启动分布式训练脚本
2. **start_distributed_training.sh** - 自动启动分布式训练脚本
3. **start_distributed_training_simple.sh** - 简化版分布式训练启动脚本
4. **stop_distributed_training.sh** - 停止分布式训练脚本
5. **train_distributed_gcu.py** - GCU分布式训练Python脚本

### 更新后的改进：

- 修复了PyTorch DDP脚本中的类型错误
- 改进了GCU设备检查逻辑
- 优化了配置文件处理方式
- 增强了错误处理机制

## 验证更新

更新完成后，验证脚本是否正常工作：

```bash
# 检查新的PyTorch DDP脚本
python3 -m py_compile scripts/train_distributed_pytorch_ddp_8card_gcu.py
python3 scripts/train_distributed_pytorch_ddp_8card_gcu.py --help

# 检查启动脚本
bash scripts/run_pytorch_ddp_training.sh --help
```

## 恢复自定义修改

如果之前对这些文件有自定义修改，可以：

1. **比较备份文件与新文件**：
   ```bash
   diff backup_conflicting_files_*/start_distributed_training.sh scripts/start_distributed_training.sh
   ```

2. **手动合并需要的修改**：
   - 检查备份文件中的自定义配置
   - 将必要的修改应用到新文件中

## 注意事项

- 备份文件保存在 `backup_conflicting_files_YYYYMMDD_HHMMSS/` 目录中
- 新版本的脚本包含了重要的bug修复，建议使用新版本
- 如有特殊配置需求，请基于新版本进行修改

## 后续操作

更新完成后，可以继续进行分布式训练：

```bash
# 使用新的PyTorch DDP方案
bash scripts/run_pytorch_ddp_training.sh --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py

# 或使用Horovod方案（如果已安装）
bash scripts/run_horovod_training.sh --config configs/train_dinov3_mmrs1m_t20_gcu_8card.py
```