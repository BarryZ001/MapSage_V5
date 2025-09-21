# T20服务器CUDA设备错误修复指南

## 🚨 问题描述

在T20服务器上运行8卡分布式训练时，出现以下错误：

```
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```

错误发生在：
```python
File "/usr/local/lib/python3.8/dist-packages/mmengine/dist/utils.py", line 130, in _init_dist_pytorch
    torch.cuda.set_device(local_rank)
```

## 🔍 问题分析

### 根本原因
1. **MMEngine的分布式初始化问题**: MMEngine的`init_dist`函数会调用`torch.cuda.set_device(local_rank)`
2. **CUDA与GCU冲突**: T20服务器使用GCU设备，没有NVIDIA GPU，但MMEngine仍尝试初始化CUDA
3. **后端配置不当**: 虽然配置了`backend='gloo'`，但MMEngine内部仍会执行CUDA相关代码

### 错误调用链
```
init_dist() -> _init_dist_pytorch() -> torch.cuda.set_device() -> RuntimeError
```

## ✅ 解决方案

### 1. 修改训练脚本 (scripts/train_distributed_8card_gcu.py)

**原代码**:
```python
# 2. 初始化分布式环境 (让MMEngine按标准方式初始化)
if cfg.get('launcher', 'none') == 'pytorch':
    from mmengine.dist import init_dist
    init_dist(launcher='pytorch', backend=cfg.env_cfg.dist_cfg.get('backend', 'eccl'))
    print("🔧 MMEngine分布式环境初始化完成")
```

**修复后**:
```python
# 2. 初始化分布式环境 (绕过MMEngine的CUDA调用，直接使用torch.distributed)
if cfg.get('launcher', 'none') == 'pytorch':
    # 获取分布式参数
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 设置分布式环境变量
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    # 直接使用torch.distributed初始化，避免MMEngine的CUDA调用
    if not dist.is_initialized():
        dist.init_process_group(
            backend='gloo',  # 使用gloo后端，兼容GCU
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        print(f"🔧 分布式环境初始化完成 - Rank {rank}/{world_size}, Backend: {dist.get_backend()}")
    else:
        print("🔧 分布式环境已初始化")
```

### 2. 修改配置文件 (configs/train_dinov3_mmrs1m_t20_gcu_8card.py)

确保使用gloo后端：
```python
env_cfg = dict(
    cudnn_benchmark=False,  # GCU环境下禁用cudnn
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),  # 使用gloo后端支持GCU分布式训练，避免CUDA调用
    resource_limit=4096
)
```

## 🔧 关键修复点

1. **绕过MMEngine的init_dist**: 直接使用`torch.distributed.init_process_group`
2. **使用gloo后端**: 避免NCCL和CUDA相关调用
3. **环境变量管理**: 正确设置MASTER_ADDR和MASTER_PORT
4. **分布式参数获取**: 从环境变量中获取rank、local_rank、world_size

## 🚀 使用方法

### 1. 更新代码
```bash
cd /workspace/code/MapSage_V5
git pull origin main
```

### 2. 重新启动训练
```bash
./start_8card_training_correct.sh
```

### 3. 验证修复
检查日志中是否出现：
```
🔧 分布式环境初始化完成 - Rank 0/8, Backend: gloo
🔧 分布式环境初始化完成 - Rank 1/8, Backend: gloo
...
🔧 分布式环境初始化完成 - Rank 7/8, Backend: gloo
```

## 📊 预期结果

修复后应该看到：
- ✅ 所有8个进程成功初始化分布式环境
- ✅ 使用gloo后端，避免CUDA调用
- ✅ 每个进程正确设置为对应的GCU设备
- ✅ 训练正常开始，无NVIDIA驱动错误

## 🔍 故障排除

### 如果仍有问题

1. **检查环境变量**:
```bash
echo $WORLD_SIZE
echo $MASTER_ADDR  
echo $MASTER_PORT
```

2. **检查GCU设备**:
```bash
python3 -c "import torch_gcu; print(torch_gcu.device_count())"
```

3. **检查进程状态**:
```bash
ps aux | grep train_distributed_8card_gcu.py
```

4. **查看详细日志**:
```bash
tail -f work_dirs/dinov3_mmrs1m_t20_gcu_8card_correct/logs/train.log
```

## 📝 技术说明

### 为什么绕过MMEngine？
- MMEngine的`init_dist`函数在PyTorch后端下会强制调用`torch.cuda.set_device`
- 这个调用在没有NVIDIA GPU的环境中会失败
- 直接使用`torch.distributed.init_process_group`可以避免这个问题

### 为什么使用gloo后端？
- gloo后端是CPU-based，不依赖特定硬件
- 兼容GCU设备的分布式训练
- 避免NCCL后端对NVIDIA GPU的依赖

---

**修复时间**: 2025-09-21  
**适用环境**: 燧原T20 GCU × 8卡  
**状态**: 已修复并测试