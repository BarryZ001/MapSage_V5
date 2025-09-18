#!/bin/bash

# T20容器创建脚本 - 带有正确的目录挂载
# 基于T20环境配置手册的最佳实践

set -e

echo "=== T20 MapSage容器创建脚本 ==="
echo "本脚本将创建带有正确目录挂载的T20容器"
echo

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Docker是否可用
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker未安装或不可用${NC}"
    exit 1
fi

echo -e "${BLUE}步骤1: 检查并创建必要的目录${NC}"

# 在T20服务器上需要创建的目录
REQUIRED_DIRS=(
    "/root/mapsage_project/code"
    "/data/datasets"
    "/root/mapsage_project/weights"
    "/root/mapsage_project/outputs"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "创建目录: $dir"
        mkdir -p "$dir"
    else
        echo "✅ 目录已存在: $dir"
    fi
done

echo -e "${GREEN}✓ 目录检查完成${NC}"
echo

echo -e "${BLUE}步骤2: 停止并删除现有容器（如果存在）${NC}"
docker stop t20_mapsage_env 2>/dev/null || echo "容器未运行或不存在"
docker rm t20_mapsage_env 2>/dev/null || echo "容器不存在"
echo -e "${GREEN}✓ 旧容器已清理${NC}"
echo

echo -e "${BLUE}步骤3: 创建新的T20容器${NC}"
echo "正在创建带有正确目录挂载的容器..."

docker run -dit \
  --name t20_mapsage_env \
  --privileged \
  --ipc=host \
  --network=host \
  -v /root/mapsage_project/code:/workspace/code \
  -v /data/datasets:/workspace/data \
  -v /root/mapsage_project/weights:/workspace/weights \
  -v /root/mapsage_project/outputs:/workspace/outputs \
  ubuntu:20.04

echo -e "${GREEN}✓ T20容器创建成功${NC}"
echo

echo -e "${BLUE}步骤4: 验证容器状态${NC}"
if docker ps | grep -q t20_mapsage_env; then
    echo -e "${GREEN}✅ 容器运行正常${NC}"
    echo
    echo -e "${YELLOW}容器信息:${NC}"
    docker ps | grep t20_mapsage_env
    echo
    echo -e "${BLUE}目录挂载:${NC}"
    echo "  宿主机 -> 容器"
    echo "  /root/mapsage_project/code -> /workspace/code"
    echo "  /data/datasets -> /workspace/data"
    echo "  /root/mapsage_project/weights -> /workspace/weights"
    echo "  /root/mapsage_project/outputs -> /workspace/outputs"
    echo
    echo -e "${GREEN}🎉 容器创建完成！${NC}"
    echo -e "${YELLOW}下一步: 请将MapSage_V5代码上传到 /root/mapsage_project/code/ 目录${NC}"
    echo -e "${YELLOW}然后执行: docker exec -it t20_mapsage_env /bin/bash${NC}"
else
    echo -e "${RED}❌ 容器创建失败${NC}"
    exit 1
fi