#!/bin/bash

# 燧原T20容器重启指导脚本
# 用于解决torch-gcu框架需要重启容器才能生效的问题

echo "🔄 燧原T20容器重启指导"
echo "================================================"

echo "📋 当前状态检查:"
echo "容器名称: $(hostname)"
echo "当前用户: $(whoami)"
echo "工作目录: $(pwd)"

echo "
🔧 torch-gcu状态检查:"
python3 -c "
try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    if hasattr(torch, 'gcu'):
        print('✅ torch-gcu框架已加载')
    else:
        print('❌ torch-gcu框架未加载，需要重启容器')
except Exception as e:
    print(f'❌ PyTorch检查失败: {e}')
"

echo "
📝 重启容器步骤:"
echo "1. 退出当前容器:"
echo "   exit"
echo "
2. 在主机上重新进入容器:"
echo "   docker exec -it t20_mapsage_env /bin/bash"
echo "
3. 进入项目目录:"
echo "   cd /workspace/code/MapSage_V5"
echo "
4. 重新运行修复脚本:"
echo "   bash scripts/fix_t20_environment.sh"

echo "
🎯 预期结果:"
echo "重启后应该看到:"
echo "✅ torch-gcu框架可用"
echo "✅ ptex模块可用"
echo "✅ XLA设备可用"

echo "
⚠️  注意事项:"
echo "- 重启容器不会丢失数据（使用了数据卷挂载）"
echo "- 环境变量会重新加载"
echo "- TopsRider软件栈会重新初始化"

echo "
🚀 准备重启？按任意键继续，或Ctrl+C取消"
read -n 1 -s

echo "\n正在退出容器..."
echo "请在主机上执行: docker exec -it t20_mapsage_env /bin/bash"
exit