# TopsRider组件服务器部署指南

本指南详细说明如何在T20服务器上部署TopsRider组件，解决ECCL后端不可用的问题。

## 概述

通过分析TopsRider安装包，我们识别出了以下关键组件：
- **ECCL库**: `tops-eccl_2.5.136-1_amd64.deb` - 集合通信库
- **SDK**: `tops-sdk_2.5.136-1_amd64.deb` - 开发工具包
- **TopsFactor**: `topsfactor_2.5.136-1_amd64.deb` - 运行时组件
- **torch_gcu**: `torch_gcu-1.10.0+2.5.136-py3.x-none-any.whl` - PyTorch GCU支持

## 部署流程

### 第一步：传输文件到服务器

使用提供的传输脚本将必要的组件从本地传输到T20服务器：

```bash
# 在本地执行
./scripts/transfer_to_server.sh
```

该脚本会：
1. 检查本地安装包的完整性
2. 测试与T20服务器的连接
3. 在服务器上创建安装目录 `/tmp/topsrider_install`
4. 传输所有必要的DEB包和wheel包
5. 传输安装脚本和测试脚本

### 第二步：在服务器上执行安装

登录到T20服务器，执行安装脚本：

```bash
# 登录到T20服务器
ssh user@your-t20-server

# 切换到root用户（安装需要管理员权限）
sudo su -

# 执行安装脚本
/tmp/topsrider_install/server_install_topsrider.sh
```

安装脚本会自动：
1. 检查系统兼容性
2. 备份现有环境
3. 安装DEB包（ECCL、SDK、TopsFactor）
4. 安装torch_gcu（根据Python版本自动选择）
5. 配置环境变量
6. 验证安装结果
7. 生成安装报告

### 第三步：验证安装

安装完成后，运行测试脚本验证功能：

```bash
# 重新加载环境变量
source ~/.bashrc

# 运行测试脚本
python3 /tmp/topsrider_install/test_eccl_backend.py
```

测试脚本会检查：
- 环境变量配置
- torch_gcu导入和GCU设备可用性
- ECCL库文件存在性
- ECCL后端可用性
- 基本张量操作
- 分布式初始化

## 关键文件说明

### 本地文件结构
```
scripts/
├── transfer_to_server.sh          # 传输脚本
├── server_install_topsrider.sh    # 服务器安装脚本
└── test_eccl_backend.py           # 测试脚本

docs/
├── server_installation_guide.md   # 详细安装指南
├── integration_plan.md           # 集成计划
└── server_deployment_guide.md    # 本文件

analysis/
├── topsrider_analysis_report.md  # 包分析报告
└── topsrider_analysis_report.json # 包分析数据
```

### 服务器端文件结构（安装后）
```
/tmp/topsrider_install/
├── tops-eccl_2.5.136-1_amd64.deb
├── tops-sdk_2.5.136-1_amd64.deb
├── topsfactor_2.5.136-1_amd64.deb
├── torch_gcu-1.10.0+2.5.136-py3.8-none-any.whl
├── server_install_topsrider.sh
└── test_eccl_backend.py

/etc/profile.d/
└── topsrider.sh                   # 环境变量配置

/tmp/
├── topsrider_install.log          # 安装日志
├── topsrider_install_report.txt   # 安装报告
├── eccl_test_results.json         # 测试结果（JSON）
└── eccl_test_report.txt           # 测试报告（文本）
```

## 环境变量配置

安装完成后，以下环境变量会被自动配置：

```bash
# TopsRider Environment Variables
export TOPS_SDK_PATH="/usr/local/tops"
export ECCL_ROOT="/usr/local/eccl"
export LD_LIBRARY_PATH="/usr/local/eccl/lib:/usr/local/tops/lib:$LD_LIBRARY_PATH"
export PATH="/usr/local/eccl/bin:/usr/local/tops/bin:$PATH"

# ECCL Configuration
export ECCL_ASYNC_DISABLE=false
export ECCL_MAX_NCHANNELS=2
export ECCL_RUNTIME_3_0_ENABLE=true
export ECCL_DEBUG=0

# GCU Configuration
export ENFLAME_VISIBLE_DEVICES=all
export GCU_MEMORY_FRACTION=0.9
```

## 验证ECCL后端

安装完成后，可以通过以下Python代码验证ECCL后端：

```python
import torch
import torch_gcu
import torch.distributed as dist

# 检查GCU可用性
print(f"GCU available: {torch_gcu.is_available()}")
print(f"GCU device count: {torch_gcu.device_count()}")

# 检查分布式后端
print(f"Available backends: {[b for b in ['nccl', 'gloo', 'mpi'] if getattr(dist, f'is_{b}_available')()]}")

# 测试ECCL后端（如果可用）
try:
    # 设置分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组（使用ECCL后端）
    dist.init_process_group(backend='eccl', world_size=1, rank=0)
    print("ECCL backend initialized successfully!")
    
    # 清理
    dist.destroy_process_group()
    
except Exception as e:
    print(f"ECCL backend test failed: {e}")
```

## 故障排除

### 常见问题

1. **DEB包安装失败**
   ```bash
   # 检查依赖
   apt-get install -f
   
   # 强制安装
   dpkg -i --force-depends package.deb
   ```

2. **torch_gcu导入失败**
   ```bash
   # 检查Python版本兼容性
   python3 --version
   
   # 重新安装对应版本的wheel包
   pip3 install torch_gcu-1.10.0+2.5.136-py3.x-none-any.whl --force-reinstall
   ```

3. **环境变量未生效**
   ```bash
   # 手动加载环境变量
   source /etc/profile.d/topsrider.sh
   
   # 或重新登录
   exit
   ssh user@server
   ```

4. **ECCL库文件未找到**
   ```bash
   # 检查库文件位置
   find /usr -name "libeccl.so" 2>/dev/null
   
   # 更新库缓存
   ldconfig
   ```

### 回滚方案

如果安装出现问题，可以使用备份进行回滚：

```bash
# 恢复环境变量
cp /tmp/topsrider_backup_*/bashrc.backup ~/.bashrc
cp /tmp/topsrider_backup_*/environment.backup /etc/environment

# 卸载DEB包
dpkg -r tops-eccl tops-sdk topsfactor

# 卸载Python包
pip3 uninstall torch_gcu -y

# 删除环境变量配置
rm -f /etc/profile.d/topsrider.sh
```

## 性能优化建议

1. **ECCL配置优化**
   ```bash
   # 根据硬件调整通道数
   export ECCL_MAX_NCHANNELS=4  # 适用于多GCU环境
   
   # 启用异步通信
   export ECCL_ASYNC_DISABLE=false
   
   # 启用3.0运行时
   export ECCL_RUNTIME_3_0_ENABLE=true
   ```

2. **内存配置**
   ```bash
   # 调整GCU内存使用比例
   export GCU_MEMORY_FRACTION=0.8  # 使用80%的GCU内存
   ```

3. **调试配置**
   ```bash
   # 启用详细日志（仅调试时使用）
   export ECCL_DEBUG=1
   export TORCH_DISTRIBUTED_DEBUG=DETAIL
   ```

## 下一步操作

1. **集成到项目**：将ECCL后端集成到现有的分布式训练代码中
2. **性能测试**：运行实际的训练任务测试性能
3. **监控部署**：设置监控确保组件正常运行
4. **文档更新**：更新项目文档反映新的部署要求

## 联系支持

如果遇到问题，请：
1. 查看安装日志：`/tmp/topsrider_install.log`
2. 查看测试报告：`/tmp/eccl_test_report.txt`
3. 收集系统信息并联系技术支持

---

*本指南基于TopsRider 2.5.136版本编写，其他版本可能需要相应调整。*