# ECCL 安装和配置指南

## 概述

根据燧原科技官方文档，`eccl` 是 Tops 相关软件的核心依赖之一，用于分布式通信。本指南详细说明 eccl 的安装、配置和故障排除。

## 官方依赖说明

根据 torch_gcu 官方文档 v3.5.0：

**Tops相关软件依赖:**
- `topsruntime` - 运行时环境
- `eccl` - 分布式通信库
- `topsaten` - 算子库
- `sdk` - 开发工具包

## ECCL 在 TopsRider 中的处理

### 1. 安装包组件

TopsRider 安装包 `TopsRider_t2x_2.5.136_deb_amd64.run` 包含以下 eccl 相关组件：

```bash
# 查看安装包组件
/installer/TopsRider_t2x_2.5.136_deb_amd64.run -l
```

### 2. 安装流程

#### 步骤1: 安装 tops-eccl 组件
```bash
# 使用 TopsRider 安装包安装 eccl
sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl
```

#### 步骤2: 验证安装
```bash
# 检查 eccl Python 模块
python3 -c "import eccl; print('eccl 可用')"
```

#### 步骤3: 环境变量配置
```bash
# 确保以下环境变量已设置
export TOPS_INSTALL_PATH=/usr/local/tops
export TOPS_RUNTIME_PATH=/usr/local/tops/runtime
export LD_LIBRARY_PATH=/usr/local/tops/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/tops/python:$PYTHONPATH
```

## 常见问题和解决方案

### 问题1: eccl 模块导入失败

**症状:**
```
ModuleNotFoundError: No module named 'eccl'
```

**解决方案:**

1. **检查安装状态**
   ```bash
   # 运行诊断脚本
   python3 scripts/diagnose_eccl_installation.py
   ```

2. **重新安装 tops-eccl**
   ```bash
   sudo /installer/TopsRider_t2x_2.5.136_deb_amd64.run -y -C tops-eccl
   ```

3. **检查 Python 路径**
   ```bash
   # 确认 eccl 模块位置
   find /usr/local/tops -name "*eccl*" -type f
   
   # 添加到 Python 路径
   export PYTHONPATH=/usr/local/tops/python:$PYTHONPATH
   ```

### 问题2: 环境变量未设置

**解决方案:**
```bash
# 创建环境变量配置
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

### 问题3: 分布式训练中 eccl 后端错误

**症状:**
```
RuntimeError: ProcessGroupECCL is not available
```

**解决方案:**

1. **检查 eccl 后端支持**
   ```bash
   python3 -c "
   import torch
   import torch_gcu
   print('Available backends:', torch.distributed.get_available_backends())
   print('ECCL backend available:', 'eccl' in torch.distributed.get_available_backends())
   "
   ```

2. **使用正确的后端初始化**
   ```python
   import torch
   import torch.distributed as dist
   import torch_gcu
   
   # 初始化分布式环境
   dist.init_process_group(backend='eccl', init_method='env://')
   ```

## 完整安装验证

运行以下脚本验证 eccl 安装：

```bash
# 使用完整的环境检测脚本
python3 scripts/check_torch_gcu_environment.py

# 或使用专门的 eccl 诊断脚本
python3 scripts/diagnose_eccl_installation.py
```

## 与官方文档的对应关系

根据官方文档，eccl 作为 Tops 相关软件依赖：

1. **libtorch_gcu 依赖**: eccl 是 torch_gcu 正常工作的必要组件
2. **分布式训练**: eccl 提供多卡分布式通信支持
3. **版本兼容**: 需要与 torch_gcu v2.5.1 版本匹配

## 注意事项

1. **版本匹配**: 确保 eccl 版本与 torch_gcu 版本兼容
2. **权限要求**: 安装需要 root 权限
3. **环境变量**: 必须正确设置环境变量才能正常使用
4. **Python 版本**: 支持 Python 3.6 和 3.8

## 相关文件

- 安装脚本: `scripts/install_topsrider_complete.sh`
- 诊断脚本: `scripts/diagnose_eccl_installation.py`
- 环境检测: `scripts/check_torch_gcu_environment.py`