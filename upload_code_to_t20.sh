#!/bin/bash

# 将MapSage_V5代码上传到T20服务器的脚本
# 使用rsync进行高效同步

set -e

echo "=== MapSage_V5代码上传到T20服务器 ==="
echo

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# T20服务器配置
T20_HOST="117.156.108.234"
T20_PORT="60025"
T20_USER="root"
T20_TARGET_DIR="/root/mapsage_project/code"

# 本地项目路径
LOCAL_PROJECT_DIR="/Users/barryzhang/myDev3/MapSage_V5"

echo -e "${BLUE}配置信息:${NC}"
echo "  本地项目: $LOCAL_PROJECT_DIR"
echo "  T20服务器: $T20_USER@$T20_HOST:$T20_PORT"
echo "  目标目录: $T20_TARGET_DIR"
echo

# 检查本地项目目录
if [ ! -d "$LOCAL_PROJECT_DIR" ]; then
    echo -e "${RED}错误: 本地项目目录不存在: $LOCAL_PROJECT_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}步骤1: 在T20服务器上创建目标目录${NC}"
ssh -p $T20_PORT $T20_USER@$T20_HOST "mkdir -p $T20_TARGET_DIR"
echo -e "${GREEN}✓ 目标目录已创建${NC}"

echo -e "${BLUE}步骤2: 同步代码到T20服务器${NC}"
echo "正在上传代码..."

# 使用rsync同步代码，排除不必要的文件
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='node_modules/' \
  --exclude='work_dirs/' \
  --exclude='experiments/' \
  --exclude='datasets/' \
  --exclude='checkpoints/' \
  --exclude='images/' \
  --exclude='*.log' \
  -e "ssh -p $T20_PORT" \
  "$LOCAL_PROJECT_DIR/" \
  "$T20_USER@$T20_HOST:$T20_TARGET_DIR/MapSage_V5/"

echo -e "${GREEN}✓ 代码上传完成${NC}"

echo -e "${BLUE}步骤3: 验证上传结果${NC}"
echo "检查T20服务器上的文件..."

ssh -p $T20_PORT $T20_USER@$T20_HOST "
echo '目标目录内容:'
ls -la $T20_TARGET_DIR/MapSage_V5/
echo
echo '关键文件检查:'
[ -f '$T20_TARGET_DIR/MapSage_V5/configs/train_dinov3_mmrs1m_t20_gcu.py' ] && echo '✅ T20配置文件存在' || echo '❌ T20配置文件缺失'
[ -f '$T20_TARGET_DIR/MapSage_V5/scripts/start_training_t20.sh' ] && echo '✅ T20训练脚本存在' || echo '❌ T20训练脚本缺失'
[ -d '$T20_TARGET_DIR/MapSage_V5/mmseg_custom' ] && echo '✅ 自定义模块存在' || echo '❌ 自定义模块缺失'
"

echo
echo -e "${GREEN}🎉 代码上传完成！${NC}"
echo -e "${YELLOW}下一步操作:${NC}"
echo "1. 进入T20容器: docker exec -it t20_mapsage_env /bin/bash"
echo "2. 进入项目目录: cd /workspace/code/MapSage_V5"
echo "3. 安装TopsRider环境: 按照环境配置手册执行"
echo "4. 开始训练: bash scripts/start_training_t20.sh"