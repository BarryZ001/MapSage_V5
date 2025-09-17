#!/bin/bash
# 解决T20服务器git pull冲突的脚本

echo "🔧 开始解决T20服务器git pull冲突..."

# 1. 备份本地修改的文件
echo "📦 备份本地修改的文件..."
mkdir -p backup_$(date +%Y%m%d_%H%M%S)
cp scripts/validate_tta.py backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "validate_tta.py不存在或已备份"

# 2. 备份未跟踪的文件
echo "📦 备份未跟踪的文件..."
cp docs/T20服务器环境配置.md backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp docs/T20集群TopsRider软件栈环境配置成功手册.md backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp docs/权重文件准备指导.md backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp docs/燧原T20适配指导.md backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp docs/阶段0执行指导.md backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp docs/阶段0验证清单.md backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp scripts/adapt_to_enflame_t20.py backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp scripts/quick_adapt_t20.sh backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"
cp scripts/update_paths_for_t20.py backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || echo "文件不存在"

# 3. 重置本地修改
echo "🔄 重置本地修改..."
git checkout -- scripts/validate_tta.py 2>/dev/null || echo "validate_tta.py重置完成"

# 4. 移除未跟踪的冲突文件
echo "🗑️ 移除未跟踪的冲突文件..."
rm -f docs/T20服务器环境配置.md
rm -f docs/T20集群TopsRider软件栈环境配置成功手册.md
rm -f docs/权重文件准备指导.md
rm -f docs/燧原T20适配指导.md
rm -f docs/阶段0执行指导.md
rm -f docs/阶段0验证清单.md
rm -f scripts/adapt_to_enflame_t20.py
rm -f scripts/quick_adapt_t20.sh
rm -f scripts/update_paths_for_t20.py

# 5. 重新拉取代码
echo "⬇️ 重新拉取最新代码..."
git pull

if [ $? -eq 0 ]; then
    echo "✅ git pull成功完成！"
    echo "📁 本地修改已备份到: backup_$(date +%Y%m%d_%H%M%S)/"
    echo "🎉 现在可以开始DINOv3+MMRS-1M训练了！"
    
    # 6. 显示新增的重要文件
    echo "\n📋 新增的重要文件:"
    echo "  - configs/train_dinov3_mmrs1m.py (DINOv3训练配置)"
    echo "  - scripts/train_dinov3_mmrs1m.sh (训练启动脚本)"
    echo "  - scripts/validate_training_env.py (环境验证脚本)"
    echo "  - docs/T20_DINOv3_Training_Guide.md (部署指南)"
    echo "  - mmseg_custom/ (自定义模块目录)"
    
    # 7. 给脚本添加执行权限
    chmod +x scripts/train_dinov3_mmrs1m.sh
    echo "\n🔧 已为训练脚本添加执行权限"
    
    # 8. 运行环境验证
    echo "\n🔍 运行环境验证..."
    python scripts/validate_training_env.py
    
else
    echo "❌ git pull失败，请检查错误信息"
    exit 1
fi

echo "\n=== 🎯 下一步操作指南 ==="
echo "1. 检查环境验证结果"
echo "2. 确保MMRS-1M数据集在 /workspace/data/mmrs1m/"
echo "3. 确保DINOv3预训练权重在 /workspace/weights/"
echo "4. 运行训练: bash scripts/train_dinov3_mmrs1m.sh"
echo "================================="