#!/bin/bash
# T20服务器文件上传脚本
# 服务器信息: ssh -p 60025 root@117.156.108.234
# 目标目录: /root/mapsage_project/code/MapSage_V5/

echo "🚀 开始上传修改的文件到T20服务器..."
echo "服务器: root@117.156.108.234:60025"
echo "目标目录: /root/mapsage_project/code/MapSage_V5/"
echo "="*50

# 上传文档文件
echo "📁 上传文档文件..."
scp -P 60025 "docs/T20服务器环境配置.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/
scp -P 60025 "docs/权重文件准备指导.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/
scp -P 60025 "docs/燧原T20适配指导.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/
scp -P 60025 "docs/阶段0执行指导.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/
scp -P 60025 "docs/阶段0验证清单.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/
scp -P 60025 "docs/T20集群TopsRider软件栈环境配置成功手册.md" root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/docs/

echo "✅ 文档文件上传完成"
echo ""

# 上传脚本文件
echo "🔧 上传脚本文件..."
scp -P 60025 scripts/adapt_to_enflame_t20.py root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/
scp -P 60025 scripts/quick_adapt_t20.sh root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/
scp -P 60025 scripts/update_paths_for_t20.py root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/
scp -P 60025 scripts/validate_tta.py root@117.156.108.234:/root/mapsage_project/code/MapSage_V5/scripts/

echo "✅ 脚本文件上传完成"
echo ""

# 设置脚本执行权限
echo "🔐 设置脚本执行权限..."
ssh -p 60025 root@117.156.108.234 "chmod +x /root/mapsage_project/code/MapSage_V5/scripts/quick_adapt_t20.sh"
ssh -p 60025 root@117.156.108.234 "chmod +x /root/mapsage_project/code/MapSage_V5/scripts/update_paths_for_t20.py"
ssh -p 60025 root@117.156.108.234 "chmod +x /root/mapsage_project/code/MapSage_V5/scripts/adapt_to_enflame_t20.py"

echo "✅ 权限设置完成"
echo ""

# 验证上传结果
echo "🔍 验证上传结果..."
ssh -p 60025 root@117.156.108.234 "ls -la /root/mapsage_project/code/MapSage_V5/docs/ | grep -E '(T20|权重|燧原|阶段0)'"
echo ""
ssh -p 60025 root@117.156.108.234 "ls -la /root/mapsage_project/code/MapSage_V5/scripts/ | grep -E '(adapt_to_enflame|quick_adapt|update_paths)'"

echo ""
echo "🎉 所有文件上传完成！"
echo "下一步可以在T20服务器上执行:"
echo "  cd /root/mapsage_project/code/MapSage_V5"
echo "  ./scripts/quick_adapt_t20.sh"
echo ""
echo "或者按照阶段0执行指导逐步进行:"
echo "  cat docs/阶段0执行指导.md"