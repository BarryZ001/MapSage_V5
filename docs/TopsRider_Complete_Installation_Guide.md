# TopsRider 完整安装指南

## 📋 概述

本指南详细说明如何在T20服务器上安装TopsRider软件栈，包括eccl、torch_gcu等关键组件。

## 🎯 安装目标

安装完成后，您将拥有：
- ✅ TopsPlatform 驱动和平台组件
- ✅ tops-eccl 分布式通信库
- ✅ torch_gcu PyTorch GCU支持
- ✅ Horovod 分布式训练框架
- ✅ AI开发工具包

## 📦 安装包信息

**安装包**: `TopsRider_t2x_2.5.136_deb_amd64.run`
**版本**: 2.5.136
**支持Python版本**: 3.6, 3.8

### 关键组件列表

| 组件 | 描述 | 必需性 |
|------|------|--------|
| topsplatform | 驱动和平台基础 | ✅ 必需 |
| topsfactor | 核心SDK | ✅ 必需 |
| tops-sdk | 开发SDK | ✅ 必需 |
| tops-eccl | 分布式通信库 | ✅ 必需 |
| torch-gcu | PyTorch GCU支持 | ✅ 必需 |
| horovod_115 | 分布式训练框架 | 🔶 推荐 |
| ai_development_toolkit | AI开发工具 | 🔶 推荐 |
| tops-models | 模型库 | 🔶 可选 |

## 🚀 快速安装

### 方法1: 使用自动安装脚本 (推荐)

```bash
# 1. 确保安装包在正确位置
sudo ls -la /installer/TopsRider_t2x_2.5.136_deb_amd64.run

# 2. 运行自动安装脚本
sudo bash scripts/install_topsrider_complete.sh
```

### 方法2: 手动安装

```bash
# 1. 查看安装包组件
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -l

# 2. 安装基础平台
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C topsplatform

# 3. 安装核心SDK组件
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C topsfactor
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-sdk
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl

# 4. 安装torch_gcu (根据Python版本选择)
# Python 3.8
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.8
# Python 3.6
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.6

# 5. 安装Horovod (可选)
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C horovod_115 --python=3.8

# 6. 安装AI工具包 (可选)
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C ai_development_toolkit
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-models
```

## ⚙️ 环境配置

### 设置环境变量

```bash
# 创建环境变量配置文件
sudo tee /etc/profile.d/topsrider.sh << 'EOF'
# TopsRider Environment Variables
export TOPS_INSTALL_PATH=/usr/local/tops
export TOPS_RUNTIME_PATH=/usr/local/tops/runtime
export TOPSRIDER_PATH=/usr/local/tops
export GCU_DEVICE_PATH=/dev/gcu

# 添加到库路径
export LD_LIBRARY_PATH=/usr/local/tops/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/tops/runtime/lib:$LD_LIBRARY_PATH

# 添加到Python路径
export PYTHONPATH=/usr/local/tops/python:$PYTHONPATH
EOF

# 使环境变量生效
source /etc/profile.d/topsrider.sh
```

### 验证环境变量

```bash
echo "TOPS_INSTALL_PATH: $TOPS_INSTALL_PATH"
echo "TOPSRIDER_PATH: $TOPSRIDER_PATH"
echo "GCU_DEVICE_PATH: $GCU_DEVICE_PATH"
```

## 🔍 安装验证

### 1. 运行环境检测脚本

```bash
python scripts/check_torch_gcu_environment.py
```

### 2. 手动验证关键组件

```bash
# 检查Python模块
python3 -c "import torch_gcu; print('torch_gcu version:', torch_gcu.__version__)"
python3 -c "import eccl; print('eccl available')"

# 检查GCU设备
ls -la /dev/gcu*

# 检查安装目录
ls -la /usr/local/tops/
```

### 3. 运行测试脚本

```bash
python scripts/test_fixed_training_setup.py
```

## 🐛 常见问题解决

### 问题1: Python版本不匹配

**症状**: torch_gcu安装失败，提示Python版本不支持

**解决方案**:
```bash
# 检查Python版本
python3 --version

# 如果是Python 3.10/3.11，尝试安装Python 3.8版本的包
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C torch-gcu --python=3.8
```

### 问题2: 权限不足

**症状**: 安装过程中出现权限错误

**解决方案**:
```bash
# 确保使用sudo权限
sudo bash scripts/install_topsrider_complete.sh
```

### 问题3: 环境变量未生效

**症状**: 安装后torch_gcu仍然无法导入

**解决方案**:
```bash
# 重新加载环境变量
source /etc/profile.d/topsrider.sh

# 或者重新登录系统
logout
# 重新登录
```

### 问题4: GCU设备不可用

**症状**: `/dev/gcu*` 设备不存在

**解决方案**:
```bash
# 检查驱动是否正确安装
lsmod | grep gcu

# 重启系统使驱动生效
sudo reboot
```

## 📊 安装后检查清单

- [ ] TopsPlatform 驱动已安装
- [ ] tops-eccl 模块可导入
- [ ] torch_gcu 模块可导入
- [ ] GCU设备 `/dev/gcu*` 存在
- [ ] 环境变量正确设置
- [ ] 环境检测脚本通过
- [ ] 测试脚本全部通过

## 🔄 卸载指南

如需卸载TopsRider：

```bash
# 停止相关服务
sudo systemctl stop tops*

# 卸载软件包
sudo apt remove --purge tops-*
sudo apt remove --purge topsfactor
sudo apt autoremove

# 清理环境变量
sudo rm -f /etc/profile.d/topsrider.sh

# 清理安装目录
sudo rm -rf /usr/local/tops
```

## 📞 技术支持

如遇到安装问题，请：

1. 查看安装日志
2. 运行环境检测脚本
3. 检查官方文档
4. 联系燧原技术支持

## 📝 更新日志

- **v1.0** (2024-01-20): 初始版本，支持TopsRider 2.5.136
- 包含完整的安装脚本和验证流程
- 支持Python 3.6/3.8自动检测安装