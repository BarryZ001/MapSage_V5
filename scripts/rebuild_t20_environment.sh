#!/bin/bash

# T20环境完全重建脚本
# 基于诊断结果的根本性解决方案
# 解决TopsRider软件栈安装失败问题

set -e

echo "=== T20环境完全重建脚本 ==="
echo "本脚本将重新创建干净的容器环境并正确安装TopsRider软件栈"
echo

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查是否在宿主机上运行
if [ -f /.dockerenv ]; then
    echo -e "${RED}错误: 此脚本必须在宿主机上运行，不能在容器内运行${NC}"
    echo "请退出容器后在宿主机上执行此脚本"
    exit 1
fi

# 检查Docker是否可用
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker未安装或不可用${NC}"
    exit 1
fi

# 检查TopsRider安装包
TOPSRIDER_INSTALLER="/root/TopsRider_t2x_2.5.136_deb_amd64.run"
if [ ! -f "$TOPSRIDER_INSTALLER" ]; then
    echo -e "${RED}错误: TopsRider安装包不存在: $TOPSRIDER_INSTALLER${NC}"
    echo "请确保TopsRider安装包位于正确路径"
    exit 1
fi

echo -e "${BLUE}步骤1: 停止并删除现有容器${NC}"
echo "正在停止容器 t20_mapsage_env..."
docker stop t20_mapsage_env 2>/dev/null || echo "容器未运行或不存在"

echo "正在删除容器 t20_mapsage_env..."
docker rm t20_mapsage_env 2>/dev/null || echo "容器不存在"

echo -e "${GREEN}✓ 旧容器已清理${NC}"
echo

echo -e "${BLUE}步骤2: 创建新的干净容器${NC}"
echo "正在创建新容器..."
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

echo -e "${GREEN}✓ 新容器已创建${NC}"
echo

echo -e "${BLUE}步骤3: 复制TopsRider安装包到容器${NC}"
docker cp "$TOPSRIDER_INSTALLER" t20_mapsage_env:/
echo -e "${GREEN}✓ TopsRider安装包已复制${NC}"
echo

echo -e "${BLUE}步骤4: 在容器内执行基础环境配置${NC}"
docker exec t20_mapsage_env bash -c "
set -e
echo '更新系统包...'
apt-get update
apt-get install -y python3-pip vim git wget curl

echo '设置Python3为默认python...'
update-alternatives --install /usr/bin/python python /usr/bin/python3 1

echo '✓ 基础环境配置完成'
"

echo -e "${GREEN}✓ 基础环境配置完成${NC}"
echo

echo -e "${BLUE}步骤5: 安装TopsRider软件栈${NC}"
echo -e "${YELLOW}注意: 这是关键步骤，必须在安装其他Python包之前完成${NC}"

docker exec t20_mapsage_env bash -c "
set -e
echo '设置TopsRider安装包权限...'
chmod +x /TopsRider_t2x_2.5.136_deb_amd64.run

echo '执行TopsRider主安装...'
/TopsRider_t2x_2.5.136_deb_amd64.run -y

echo '执行torch-gcu组件安装...'
/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu

echo '配置环境变量...'
echo 'export PATH=\"/opt/tops/bin:\$PATH\"' >> /root/.bashrc
echo 'export LD_LIBRARY_PATH=\"/opt/tops/lib:\$LD_LIBRARY_PATH\"' >> /root/.bashrc
echo 'export PYTHONPATH=\"/opt/tops/lib/python3.8/site-packages:\$PYTHONPATH\"' >> /root/.bashrc

# 立即应用环境变量
export PATH=\"/opt/tops/bin:\$PATH\"
export LD_LIBRARY_PATH=\"/opt/tops/lib:\$LD_LIBRARY_PATH\"
export PYTHONPATH=\"/opt/tops/lib/python3.8/site-packages:\$PYTHONPATH\"

echo '执行ldconfig更新动态链接库缓存...'
ldconfig

echo '✓ TopsRider软件栈安装完成'
"

echo -e "${GREEN}✓ TopsRider软件栈安装完成${NC}"
echo

echo -e "${BLUE}步骤6: 验证torch-gcu安装${NC}"
echo "正在验证torch-gcu是否正确安装..."

TORCH_GCU_CHECK=$(docker exec t20_mapsage_env bash -c "
source /root/.bashrc
python -c 'import torch; print(torch.gcu.is_available())' 2>/dev/null || echo 'False'
")

if [ "$TORCH_GCU_CHECK" = "True" ]; then
    echo -e "${GREEN}✓ torch-gcu验证成功: $TORCH_GCU_CHECK${NC}"
else
    echo -e "${RED}✗ torch-gcu验证失败: $TORCH_GCU_CHECK${NC}"
    echo "请检查TopsRider安装是否有错误"
    exit 1
fi
echo

echo -e "${BLUE}步骤7: 准备项目requirements.txt${NC}"
echo "正在修复requirements.txt中的版本冲突..."

docker exec t20_mapsage_env bash -c "
cd /workspace/code/MapSage_V5
if [ -f requirements.txt ]; then
    # 备份原文件
    cp requirements.txt requirements.txt.backup
    
    # 修复已知的版本问题
    sed -i 's/torch==2.0.1+cu118/torch/g' requirements.txt
    sed -i 's/torchvision==0.15.2+cu118/torchvision/g' requirements.txt
    sed -i 's/torchaudio==2.0.2+cu118/torchaudio/g' requirements.txt
    
    echo '✓ requirements.txt已修复'
else
    echo '警告: requirements.txt不存在'
fi
"

echo -e "${GREEN}✓ requirements.txt准备完成${NC}"
echo

echo -e "${BLUE}步骤8: 安装项目依赖${NC}"
echo -e "${YELLOW}注意: 现在安装其他依赖，torch-gcu已经是主要PyTorch安装${NC}"

docker exec t20_mapsage_env bash -c "
source /root/.bashrc
cd /workspace/code/MapSage_V5

if [ -f requirements.txt ]; then
    echo '安装项目依赖...'
    pip3 install -r requirements.txt
    echo '✓ 项目依赖安装完成'
else
    echo '跳过依赖安装: requirements.txt不存在'
fi
"

echo -e "${GREEN}✓ 项目依赖安装完成${NC}"
echo

echo -e "${BLUE}步骤9: 最终验证${NC}"
echo "正在执行完整的环境验证..."

docker exec t20_mapsage_env bash -c "
source /root/.bashrc
cd /workspace/code/MapSage_V5

echo '=== 最终环境验证 ==='
echo '1. Python版本:'
python --version

echo '2. torch-gcu可用性:'
python -c 'import torch; print(\"torch.gcu.is_available():\", torch.gcu.is_available())'

echo '3. GCU设备数量:'
python -c 'import torch; print(\"GCU设备数量:\", torch.gcu.device_count())' 2>/dev/null || echo '无法获取GCU设备数量'

echo '4. 关键文件检查:'
ls -la /opt/tops/bin/tops-smi 2>/dev/null && echo '✓ tops-smi存在' || echo '✗ tops-smi缺失'
ls -la /opt/tops/lib/libtops.so 2>/dev/null && echo '✓ libtops.so存在' || echo '✗ libtops.so缺失'

echo '5. 环境变量:'
echo \"PATH: \$PATH\"
echo \"LD_LIBRARY_PATH: \$LD_LIBRARY_PATH\"
echo \"PYTHONPATH: \$PYTHONPATH\"

echo '=== 验证完成 ==='
"

echo
echo -e "${GREEN}=== T20环境重建完成 ===${NC}"
echo
echo -e "${BLUE}后续步骤:${NC}"
echo "1. 进入容器: docker exec -it t20_mapsage_env bash"
echo "2. 激活环境: source /root/.bashrc"
echo "3. 进入项目: cd /workspace/code/MapSage_V5"
echo "4. 运行验证脚本: python scripts/validate_official_installation.py"
echo
echo -e "${YELLOW}重要提示:${NC}"
echo "- 每次进入容器都要执行 'source /root/.bashrc' 来加载环境变量"
echo "- 如果仍有问题，请检查TopsRider安装包版本和系统兼容性"
echo "- 建议运行完整的验证脚本确认所有组件正常工作"
echo