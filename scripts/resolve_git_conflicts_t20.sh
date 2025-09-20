#!/bin/bash
# scripts/resolve_git_conflicts_t20.sh - 解决T20服务器git pull冲突

set -e

echo "🔧 解决T20服务器git pull冲突"
echo "📅 时间: $(date)"

# 冲突的文件列表
CONFLICTING_FILES=(
    "scripts/start_distributed_manual.sh"
    "scripts/start_distributed_training.sh"
    "scripts/start_distributed_training_simple.sh"
    "scripts/stop_distributed_training.sh"
    "scripts/train_distributed_gcu.py"
)

# 创建备份目录
BACKUP_DIR="./backup_conflicting_files_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📁 创建备份目录: $BACKUP_DIR"

# 备份冲突文件
echo "💾 备份冲突文件..."
for file in "${CONFLICTING_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - 备份: $file"
        cp "$file" "$BACKUP_DIR/"
    else
        echo "  - 文件不存在: $file"
    fi
done

# 移除冲突文件
echo "🗑️ 移除冲突文件..."
for file in "${CONFLICTING_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - 移除: $file"
        rm -f "$file"
    fi
done

echo "✅ 冲突文件已备份并移除"
echo "📁 备份位置: $BACKUP_DIR"
echo ""
echo "🔄 现在可以执行 git pull origin main"
echo ""
echo "📋 执行步骤:"
echo "1. git pull origin main"
echo "2. 检查更新后的文件"
echo "3. 如需要，从备份中恢复自定义修改"
echo ""
echo "🔍 备份文件列表:"
ls -la "$BACKUP_DIR/"