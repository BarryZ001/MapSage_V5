#!/bin/bash

# T20服务器DNS rdtypes错误修复脚本
echo "🔧 T20服务器DNS rdtypes错误修复"
echo "================================================"

# 进入容器并运行修复脚本
echo "📦 进入dinov3_trainer容器并运行DNS修复..."

docker exec -it dinov3_trainer bash -c "
cd /workspace/code/MapSage_V5 &&
python3 scripts/fix_dns_rdtypes_issue.py
"

echo ""
echo "✅ DNS修复脚本执行完成"
echo ""
echo "💡 接下来请："
echo "1. 验证修复结果"
echo "2. 重新启动8卡训练"
echo "3. 监控训练日志"