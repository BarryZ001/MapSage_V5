# T20服务器MMCV版本兼容性修复指南

## 问题描述

在T20服务器上运行训练时遇到以下错误：
```
AssertionError: MMCV==2.0.1 is used but incompatible. Please install mmcv>=(1, 3, 13, 0, 0, 0), <(1, 8, 0, 0, 0, 0).
```

这是因为mmseg要求mmcv版本必须满足：`1.3.13 <= version < 1.8.0`，而当前安装的是mmcv==2.0.1。

## 解决方案

### 方法1：使用自动修复脚本（推荐）

```bash
# 在T20服务器上执行
cd /workspace/code/MapSage_V5
git pull origin main
./fix_mmcv_version_t20.sh
```

### 方法2：手动修复

```bash
# 1. 卸载现有版本
pip3 uninstall mmcv mmcv-full mmcv-lite -y

# 2. 清理缓存
pip3 cache purge

# 3. 安装兼容版本
pip3 install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html

# 4. 验证安装
python3 -c "import mmcv; print(f'mmcv version: {mmcv.__version__}')"
python3 -c "import mmseg; print('mmseg导入成功')"
```

## 版本兼容性说明

- **mmseg要求**: `1.3.13 <= mmcv < 1.8.0`
- **推荐版本**: `mmcv-full==1.7.2`
- **不兼容版本**: `mmcv==2.0.1` (太新)

## 验证修复结果

修复完成后，运行以下命令验证：

```bash
python3 -c "
import mmcv
from packaging import version
mmcv_version = version.parse(mmcv.__version__)
mmcv_min = version.parse('1.3.13')
mmcv_max = version.parse('1.8.0')
if mmcv_min <= mmcv_version < mmcv_max:
    print('✅ mmcv版本兼容mmseg要求')
else:
    print('❌ mmcv版本不兼容')
"
```

## 重新运行训练

修复完成后，可以重新运行训练：

```bash
cd /workspace/code/MapSage_V5
export PYTHONPATH=/workspace/code/MapSage_V5:$PYTHONPATH
python3 scripts/train_distributed_8card_gcu.py configs/train_dinov3_mmrs1m_t20_gcu_8card.py
```

## 常见问题

### Q: 为什么不能使用mmcv==2.0.1？
A: mmseg是基于mmcv 1.x版本开发的，与2.x版本API不兼容。

### Q: 如果修复脚本失败怎么办？
A: 可以尝试手动安装，或者检查网络连接和pip源配置。

### Q: 是否需要重新安装mmseg？
A: 通常不需要，只需要安装兼容版本的mmcv即可。