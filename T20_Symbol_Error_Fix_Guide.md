# T20服务器MMCV符号错误修复指南

## 问题描述

在T20服务器上运行训练时遇到以下错误：

1. **MMCV符号错误**:
```
⚠️ 模块导入失败: /usr/local/lib/python3.8/dist-packages/mmcv/_ext.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEENS2_8optionalINS2_10ScalarTypeEEENS6_INS2_6LayoutEEENS6_INS2_6DeviceEEENS6_IbEE
```

2. **损失函数导入错误**:
```
AttributeError: module 'mmseg_custom.losses' has no attribute 'CrossEntropyLoss'
```

## 问题原因

1. **符号错误**: mmcv编译版本与当前PyTorch版本不兼容，导致C++符号链接错误
2. **损失函数错误**: 自定义losses模块没有正确导出CrossEntropyLoss类

## 解决方案

### 方法1：使用自动修复脚本（推荐）

```bash
# 在T20服务器上执行
cd /workspace/code/MapSage_V5
git pull origin main

# 修复符号错误
./fix_mmcv_symbol_error_t20.sh

# 如果还有版本兼容性问题
./fix_mmcv_version_t20.sh
```

### 方法2：手动修复步骤

#### 步骤1：修复MMCV符号错误

```bash
# 1. 卸载现有mmcv
pip3 uninstall mmcv mmcv-full mmcv-lite -y

# 2. 清理缓存
pip3 cache purge

# 3. 检查PyTorch版本
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')"

# 4. 根据PyTorch版本安装兼容的mmcv
# 对于PyTorch 2.0.x
pip3 install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html

# 或者安装CPU版本（如果上面失败）
pip3 install mmcv-full==1.7.2
```

#### 步骤2：验证修复结果

```bash
python3 -c "
import mmcv
print(f'MMCV版本: {mmcv.__version__}')

import torch
x = torch.zeros(2, 3)
print('torch.zeros正常工作')

from mmcv.utils import Config
print('mmcv.utils.Config导入成功')
"
```

## 已修复的问题

### 1. 损失函数导入问题

- ✅ 修复了`mmseg_custom.losses`模块的`CrossEntropyLoss`属性错误
- ✅ 添加了动态`__all__`列表，确保正确导出可用的损失函数
- ✅ 在MMSeg损失函数不可用时，提供了简单的替代实现

### 2. 训练脚本改进

- ✅ 改进了自定义模块导入的错误处理
- ✅ 添加了更详细的导入成功/失败信息

## 重新运行训练

修复完成后，重新运行训练：

```bash
cd /workspace/code/MapSage_V5
export PYTHONPATH=/workspace/code/MapSage_V5:$PYTHONPATH
python3 scripts/train_distributed_8card_gcu.py configs/train_dinov3_mmrs1m_t20_gcu_8card.py
```

## 常见问题

### Q: 为什么会出现符号错误？
A: 这通常是因为mmcv的预编译版本与当前PyTorch版本不兼容，特别是在不同的CUDA/CPU环境下。

### Q: 如果符号错误仍然存在怎么办？
A: 可以尝试：
1. 安装CPU版本的mmcv-full
2. 从源码编译mmcv（耗时较长）
3. 使用mmcv-lite替代mmcv-full

### Q: 损失函数使用自定义实现会影响训练效果吗？
A: 自定义的SimpleCrossEntropyLoss实现与MMSeg原版功能相同，不会影响训练效果。

## 验证清单

修复完成后，确保以下项目都正常：

- [ ] `import mmcv` 不报错
- [ ] `import mmseg` 不报错  
- [ ] `from mmseg_custom.losses import CrossEntropyLoss` 不报错
- [ ] `torch.zeros(2, 3)` 正常工作
- [ ] 训练脚本可以正常启动