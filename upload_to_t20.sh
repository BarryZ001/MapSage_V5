#!/bin/bash
# T20服务器文件上传脚本 - 更新版本
# 服务器信息: ssh -p 60026 root@117.156.108.234 (注意端口已更新为60026)
# 目标目录: /workspace/code/MapSage_V5/ (Docker容器内路径)

echo "🚀 开始上传修改的文件到T20服务器..."
echo "服务器: root@117.156.108.234:60026"
echo "目标目录: /workspace/code/MapSage_V5/ (Docker容器内)"
echo "="*50

# 上传修复后的训练脚本
echo "🔧 上传修复后的训练脚本..."
sshpass -p 'enflame@123' scp -P 60026 scripts/train_distributed_8card_gcu.py root@117.156.108.234:/tmp/
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker cp /tmp/train_distributed_8card_gcu.py dinov3_trainer:/workspace/code/MapSage_V5/scripts/"

echo "✅ 训练脚本上传完成"
echo ""

# 上传配置文件
echo "📄 上传配置文件..."
sshpass -p 'enflame@123' scp -P 60026 configs/train_dinov3_mmrs1m_t20_gcu_8card.py root@117.156.108.234:/tmp/
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker cp /tmp/train_dinov3_mmrs1m_t20_gcu_8card.py dinov3_trainer:/workspace/code/MapSage_V5/configs/"

echo "✅ 配置文件上传完成"
echo ""

# 上传启动脚本
echo "🚀 上传启动脚本..."
sshpass -p 'enflame@123' scp -P 60026 scripts/start_8card_training.sh root@117.156.108.234:/tmp/
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker cp /tmp/start_8card_training.sh dinov3_trainer:/workspace/code/MapSage_V5/scripts/"

echo "✅ 启动脚本上传完成"
echo ""

# 上传文档文件
echo "📁 上传文档文件..."
sshpass -p 'enflame@123' scp -P 60026 "docs/T20_DINOv3_Training_Guide.md" root@117.156.108.234:/tmp/
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker cp /tmp/T20_DINOv3_Training_Guide.md dinov3_trainer:/workspace/code/MapSage_V5/docs/"

# 上传其他重要脚本
echo "🔧 上传其他脚本文件..."
sshpass -p 'enflame@123' scp -P 60026 scripts/diagnose_torch_gcu.py root@117.156.108.234:/tmp/
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker cp /tmp/diagnose_torch_gcu.py dinov3_trainer:/workspace/code/MapSage_V5/scripts/"

echo "✅ 其他脚本文件上传完成"
echo ""

# 设置脚本执行权限
echo "🔐 设置脚本执行权限..."
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker exec dinov3_trainer chmod +x /workspace/code/MapSage_V5/scripts/start_8card_training.sh"
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker exec dinov3_trainer chmod +x /workspace/code/MapSage_V5/scripts/train_distributed_8card_gcu.py"
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker exec dinov3_trainer chmod +x /workspace/code/MapSage_V5/scripts/diagnose_torch_gcu.py"

echo "✅ 权限设置完成"
echo ""

# 验证上传结果
echo "🔍 验证上传结果..."
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker exec dinov3_trainer ls -la /workspace/code/MapSage_V5/scripts/ | grep -E '(train_distributed|start_8card|diagnose)'"
echo ""
sshpass -p 'enflame@123' ssh -p 60026 root@117.156.108.234 "docker exec dinov3_trainer ls -la /workspace/code/MapSage_V5/configs/ | grep train_dinov3"

echo ""
echo "🎉 所有文件上传完成！"
echo "下一步可以在T20服务器的Docker容器内执行:"
echo "  docker exec -it dinov3_trainer bash"
echo "  cd /workspace/code/MapSage_V5"
echo "  torchrun --nproc_per_node=8 --master_port=29500 scripts/train_distributed_8card_gcu.py configs/train_dinov3_mmrs1m_t20_gcu_8card.py"
echo "  cd /root/mapsage_project/code/MapSage_V5"
echo "  ./scripts/quick_adapt_t20.sh"
echo ""
echo "或者按照阶段0执行指导逐步进行:"
echo "  cat docs/阶段0执行指导.md"