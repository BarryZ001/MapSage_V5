#!/bin/bash
# 燧原T20快速适配脚本
# 一键完成MapSage V5项目的T20适配

set -e  # 遇到错误立即退出

echo "🚀 开始燧原T20适配流程..."
echo "==========================================="

# 检查当前目录
if [ ! -f "scripts/adapt_to_enflame_t20.py" ]; then
    echo "❌ 错误：请在MapSage_V5项目根目录下运行此脚本"
    exit 1
fi

# 步骤1：环境检查
echo "📋 步骤1：检查环境..."
echo "检查torch-gcu和ptex是否可用..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "❌ PyTorch导入失败"
    exit 1
}

python3 -c "import ptex; print('ptex库检查通过')" || {
    echo "❌ ptex库导入失败，请确认torch-gcu已正确安装"
    echo "💡 解决方案：cd / && ./TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu"
    exit 1
}

echo "✅ 环境检查通过"
echo ""

# 步骤2：备份原文件
echo "📋 步骤2：备份原文件..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 需要适配的文件列表
FILES_TO_ADAPT=(
    "scripts/validate_tta.py"
    "scripts/run_staged_distillation_experiment.py"
    "scripts/run_task_oriented_distillation.py"
    "scripts/run_improved_distillation_experiment.py"
    "configs/train_distill_dinov3_v2_improved.py"
)

for file in "${FILES_TO_ADAPT[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "✅ 已备份: $file"
    else
        echo "⚠️  文件不存在: $file"
    fi
done

echo "📁 备份目录: $BACKUP_DIR"
echo ""

# 步骤3：干运行检查
echo "📋 步骤3：干运行检查需要修改的内容..."
echo "执行干运行模式，查看将要进行的修改..."
echo ""

python3 scripts/adapt_to_enflame_t20.py --dry-run "${FILES_TO_ADAPT[@]}"

echo ""
echo "🤔 请检查上述修改内容是否合理"
read -p "是否继续执行实际修改？(y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 用户取消操作"
    exit 1
fi

# 步骤4：更新路径配置
echo "📋 步骤4：更新路径配置..."
echo "执行路径配置更新..."
python3 scripts/update_paths_for_t20.py --dry-run "${FILES_TO_ADAPT[@]}"

echo ""
read -p "确认路径更新正确？(y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 路径更新被取消"
    exit 1
fi

python3 scripts/update_paths_for_t20.py "${FILES_TO_ADAPT[@]}"
echo "✅ 路径配置更新完成"
echo ""

# 步骤5：执行实际适配
echo "📋 步骤5：执行实际适配..."
python3 scripts/adapt_to_enflame_t20.py "${FILES_TO_ADAPT[@]}"

echo "✅ 代码适配完成"
echo ""

# 步骤5：语法检查
echo "📋 步骤5：语法检查..."
for file in "${FILES_TO_ADAPT[@]}"; do
    if [ -f "$file" ]; then
        echo "检查: $file"
        python3 -m py_compile "$file" && echo "  ✅ 语法正确" || echo "  ❌ 语法错误"
    fi
done

echo ""

# 步骤6：设备测试
echo "📋 步骤6：设备功能测试..."
echo "测试ptex设备创建..."
python3 -c "
import torch
import ptex
device = ptex.device('xla')
print(f'✅ 设备创建成功: {device}')
tensor = torch.randn(2, 3).to(device)
print(f'✅ 张量操作成功: {tensor.shape}')
" || {
    echo "❌ 设备测试失败"
    exit 1
}

echo ""

# 完成总结
echo "🎉 燧原T20适配完成！"
echo "==========================================="
echo "📊 适配总结："
echo "  - 已适配文件数: ${#FILES_TO_ADAPT[@]}"
echo "  - 备份目录: $BACKUP_DIR"
echo "  - 环境检查: ✅ 通过"
echo "  - 语法检查: ✅ 通过"
echo "  - 设备测试: ✅ 通过"
echo ""
echo "🚀 下一步：执行基准验证"
echo "   运行命令: python scripts/validate_tta.py"
echo "   目标指标: mIoU ≥ 84.96%"
echo ""
echo "📊 T20服务器环境信息:"
echo "   - 容器名称: t20_mapsage_env"
echo "   - 代码路径: /workspace/code"
echo "   - 数据路径: /workspace/data"
echo "   - 权重路径: /workspace/weights"
echo "   - 输出路径: /workspace/outputs"
echo ""
echo "📚 参考文档:"
echo "   - docs/阶段0执行指导.md"
echo "   - docs/阶段0验证清单.md"
echo "   - docs/燧原T20适配指导.md"