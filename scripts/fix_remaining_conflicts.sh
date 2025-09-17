#!/bin/bash

# T20服务器剩余冲突解决脚本
# 解决configs/v87/v87_tta_final.py文件冲突

echo "🔧 解决剩余的git冲突..."

# 删除冲突的未跟踪文件
echo "🗑️ 删除冲突文件: configs/v87/v87_tta_final.py"
rm -f configs/v87/v87_tta_final.py

# 确保v87目录存在但为空
if [ -d "configs/v87" ]; then
    echo "📁 清理v87目录..."
    rm -rf configs/v87/*
fi

# 重新执行git pull
echo "⬇️ 重新拉取最新代码..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ Git pull成功完成！"
    echo "📋 检查当前状态..."
    git status
    
    echo ""
    echo "🎉 冲突解决完成！现在可以继续训练了。"
    echo "💡 下一步操作:"
    echo "   1. 验证训练环境: python scripts/validate_training_env.py"
    echo "   2. 启动训练: bash scripts/train_dinov3_mmrs1m.sh"
else
    echo "❌ Git pull仍然失败，请检查错误信息"
    echo "📋 当前git状态:"
    git status
    echo ""
    echo "🔍 如果还有其他冲突文件，请手动删除后重新运行此脚本"
fi

echo "="*60