# T20服务器TopsRider安装故障排除指南

## 基于安装日志的问题诊断

### 1. GCU硬件检测失败 (GCU count: 0)

**问题描述**: 安装器检测到GCU设备数量为0

**可能原因**:
- 硬件驱动未正确安装
- 设备未正确连接或识别
- 在Docker容器中运行，无法访问物理硬件
- PCI设备权限问题

**解决方案**:

#### 1.1 检查硬件连接
```bash
# 检查PCI设备
lspci | grep -i enflame

# 检查设备文件
ls -la /dev/gcu*

# 检查内核模块
lsmod | grep gcu
```

#### 1.2 安装/重新安装驱动
```bash
# 检查驱动安装状态
dmesg | grep -i gcu
dmesg | grep -i enflame

# 如果需要重新安装驱动
modprobe -r gcu_driver  # 卸载现有驱动
modprobe gcu_driver     # 重新加载驱动
```

#### 1.3 容器环境配置
如果在Docker容器中运行，需要：
```bash
# 启动容器时添加设备映射
docker run --device=/dev/gcu0:/dev/gcu0 \
           --device=/dev/gcu1:/dev/gcu1 \
           --privileged \
           your_container_image

# 或在docker-compose.yml中配置
devices:
  - "/dev/gcu0:/dev/gcu0"
  - "/dev/gcu1:/dev/gcu1"
privileged: true
```

### 2. 不完整的组件安装

**问题描述**: 只安装了torch-gcu，缺少ECCL、SDK、TopsFactor等关键组件

**影响**:
- ECCL分布式训练功能不可用
- 缺少必要的运行时环境
- 性能优化功能受限

**解决方案**:

#### 2.1 使用完整安装脚本
```bash
# 使用我们准备的完整安装脚本
./server_install_topsrider.sh

# 或手动安装缺失组件
dpkg -i tops-eccl_2.5.136-1_amd64.deb
dpkg -i tops-sdk_2.5.136-1_amd64.deb
dpkg -i topsfactor_2.5.136-1_amd64.deb
```

#### 2.2 验证安装
```bash
# 检查已安装的包
dpkg -l | grep -E "(eccl|tops|torch)"

# 运行验证脚本
python3 test_eccl_backend.py
```

### 3. 环境变量配置问题

**问题描述**: 安装后环境变量未正确配置

**解决方案**:

#### 3.1 配置必要的环境变量
```bash
# 添加到 ~/.bashrc 或 /etc/environment
export TOPS_INSTALL_PATH=/usr/local/tops
export LD_LIBRARY_PATH=$TOPS_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export PATH=$TOPS_INSTALL_PATH/bin:$PATH

# ECCL相关环境变量
export ECCL_ROOT=/usr/local/eccl
export ECCL_LIBRARY_PATH=$ECCL_ROOT/lib

# 重新加载环境变量
source ~/.bashrc
```

#### 3.2 验证环境变量
```bash
echo $TOPS_INSTALL_PATH
echo $LD_LIBRARY_PATH
echo $ECCL_ROOT
```

### 4. Python环境兼容性问题

**问题描述**: torch_gcu与Python版本不匹配

**解决方案**:

#### 4.1 检查Python版本
```bash
python3 --version
python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
```

#### 4.2 安装对应版本的torch_gcu
```bash
# Python 3.8
pip3 install torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl

# Python 3.9
pip3 install torch_gcu-1.10.0+2.5.136-py3.9-none-any.whl

# Python 3.10
pip3 install torch_gcu-1.10.0+2.5.136-py3.10-none-any.whl
```

### 5. 权限问题

**问题描述**: 非root用户无法访问GCU设备

**解决方案**:

#### 5.1 设置设备权限
```bash
# 检查设备权限
ls -la /dev/gcu*

# 设置用户组权限
sudo usermod -a -G gcu $USER
sudo chmod 666 /dev/gcu*
```

#### 5.2 创建udev规则
```bash
# 创建 /etc/udev/rules.d/99-gcu.rules
echo 'KERNEL=="gcu*", GROUP="gcu", MODE="0666"' | sudo tee /etc/udev/rules.d/99-gcu.rules

# 重新加载udev规则
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 6. 网络和依赖问题

**问题描述**: 安装过程中网络连接或依赖包问题

**解决方案**:

#### 6.1 更新包索引
```bash
apt-get update
apt-get install -f  # 修复依赖问题
```

#### 6.2 安装必要依赖
```bash
apt-get install -y build-essential python3-dev libtinfo5
```

### 7. 验证安装成功

#### 7.1 运行完整验证
```bash
# 使用我们的验证脚本
python3 scripts/test_eccl_backend.py

# 检查输出报告
cat /tmp/eccl_test_report_*.json
```

#### 7.2 手动验证关键功能
```python
# 测试torch_gcu导入
import torch
import torch_gcu

# 测试设备可用性
print(f"GCU available: {torch.gcu.is_available()}")
print(f"GCU device count: {torch.gcu.device_count()}")

# 测试ECCL后端
import torch.distributed as dist
print(f"ECCL backend available: {dist.is_eccl_available()}")
```

## 常见错误代码和解决方案

| 错误代码 | 描述 | 解决方案 |
|---------|------|----------|
| `ImportError: No module named 'torch_gcu'` | torch_gcu未安装 | 安装对应Python版本的torch_gcu |
| `RuntimeError: No GCU devices found` | 未检测到GCU设备 | 检查硬件连接和驱动 |
| `ECCL backend not available` | ECCL后端不可用 | 安装ECCL包并配置环境变量 |
| `Permission denied: /dev/gcu0` | 设备权限问题 | 设置正确的设备权限 |

## 联系支持

如果以上解决方案无法解决问题，请收集以下信息：
1. 完整的安装日志
2. 系统信息 (`uname -a`, `lsb_release -a`)
3. 硬件信息 (`lspci | grep -i enflame`)
4. 验证脚本输出 (`test_eccl_backend.py`)

## 相关文件
- `server_install_topsrider.sh` - 完整安装脚本
- `test_eccl_backend.py` - 验证脚本
- `topsrider_install_log_analysis.md` - 日志分析报告