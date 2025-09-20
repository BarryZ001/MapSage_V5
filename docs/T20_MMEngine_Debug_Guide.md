# T20服务器MMEngine DDP调试完整指南

## 概述

本指南提供了在T20服务器上深入MMEngine源码，添加DDP设备诊断日志的详细步骤。这是解决顽固的DDP设备配置错误的最终调试方案。

## 问题背景

经过多次修复尝试，8卡分布式训练仍然出现以下错误：
```
ValueError: DistributedDataParallel device_ids and output_device arguments only work with single-device/multiple-device GPU modules or CPU modules, but got device_ids [0], output_device None, and module parameters on device cpu.
```

这表明在DDP包装时，模型参数仍在CPU上，而device_ids设置为[0]，导致设备不匹配。

## 调试策略

我们将在MMEngine的`wrap_model`函数中，在DDP包装前的最后一刻添加详细的设备诊断日志，以确定模型参数的真实设备状态。

## 操作步骤

### 第1步：登录T20服务器并进入容器

```bash
# SSH登录T20服务器
ssh your_username@t20_server_ip

# 进入dinov3-container容器
docker exec -it dinov3-container bash

# 切换到项目目录
cd /workspace/code/MapSage_V5
```

### 第2步：拉取最新代码

```bash
# 拉取最新的修复代码
git pull origin main

# 确认当前在正确的分支
git branch
```

### 第3步：运行自动化调试脚本

我们提供了自动化脚本来修改MMEngine源码：

```bash
# 运行调试脚本
python3 scripts/debug_mmengine_ddp.py
```

脚本会自动：
- 检查T20服务器环境
- 备份原始MMEngine文件
- 在正确位置添加调试代码
- 提供后续操作指导

### 第4步：手动修改（备选方案）

如果自动化脚本失败，可以手动修改：

#### 4.1 打开MMEngine源码文件

```bash
vim /usr/local/lib/python3.8/dist-packages/mmengine/runner/runner.py
```

#### 4.2 定位wrap_model函数

在vim中搜索wrap_model函数：
```vim
/def wrap_model
```

#### 4.3 找到DDP包装代码

在wrap_model函数内查找类似以下的代码：
```python
model = MMDistributedDataParallel(
    module=model,
    device_ids=device_ids,
    # ... 其他参数
)
```

#### 4.4 添加调试代码

在DDP包装代码的**正上方**添加以下调试代码：

```python
    # ===== START: MapSage DDP设备深度调试日志 =====
    print('\n' + '='*60, flush=True)
    print('>>> MMEngine wrap_model DDP设备调试信息 <<<', flush=True)
    print('='*60, flush=True)
    
    try:
        # 检查模型参数设备分布
        param_devices = set()
        param_count = 0
        
        for name, param in model.named_parameters():
            param_devices.add(str(param.device))
            param_count += 1
            if param_count <= 5:  # 打印前5个参数的详细信息
                print(f'>>> 参数 {name}: 设备={param.device}, 形状={param.shape}', flush=True)
        
        print(f'>>> 总参数数量: {param_count}', flush=True)
        print(f'>>> 参数设备分布: {param_devices}', flush=True)
        
        # 检查模型本身的设备
        if hasattr(model, 'device'):
            print(f'>>> 模型设备属性: {model.device}', flush=True)
        
        # 检查第一个参数的设备（最常用的检查方法）
        first_param = next(model.parameters())
        print(f'>>> 第一个参数设备: {first_param.device}', flush=True)
        
        # 检查当前CUDA/GCU设备
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                print(f'>>> 当前CUDA设备: {torch.cuda.current_device()}', flush=True)
        except:
            pass
            
        try:
            import torch_gcu
            if torch_gcu.is_available():
                print(f'>>> 当前GCU设备: {torch_gcu.current_device()}', flush=True)
        except:
            pass
        
        # 检查环境变量
        local_rank = os.environ.get('LOCAL_RANK', 'None')
        rank = os.environ.get('RANK', 'None')
        world_size = os.environ.get('WORLD_SIZE', 'None')
        print(f'>>> 分布式环境: LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}', flush=True)
        
        # 警告检查
        if any('cpu' in device for device in param_devices):
            print('🚨 警告: 检测到模型参数仍在CPU上!', flush=True)
            print('🚨 这将导致DDP设备不匹配错误!', flush=True)
        else:
            print('✅ 模型参数已正确移动到加速器设备', flush=True)
            
    except Exception as e:
        print(f'>>> DDP包装前设备检查失败: {e}', flush=True)
        import traceback
        traceback.print_exc()
    
    print('='*60, flush=True)
    print('>>> DDP设备调试信息结束 <<<', flush=True)
    print('='*60 + '\n', flush=True)
    # ===== END: MapSage DDP设备深度调试日志 =====

    # 原始的DDP包装代码
    model = MMDistributedDataParallel(
        module=model,
        # ... 其他参数
    )
```

### 第5步：运行训练并收集调试信息

```bash
# 运行8卡分布式训练
bash scripts/start_8card_training.sh
```

### 第6步：分析调试输出

在训练输出中查找以下关键信息：

#### 6.1 正常情况的输出示例
```
============================================================
>>> MMEngine wrap_model DDP设备调试信息 <<<
============================================================
>>> 参数 backbone.patch_embed.proj.weight: 设备=gcu:0, 形状=torch.Size([384, 3, 14, 14])
>>> 参数 backbone.patch_embed.proj.bias: 设备=gcu:0, 形状=torch.Size([384])
>>> 总参数数量: 21234567
>>> 参数设备分布: {'gcu:0'}
>>> 第一个参数设备: gcu:0
>>> 当前GCU设备: 0
>>> 分布式环境: LOCAL_RANK=0, RANK=0, WORLD_SIZE=8
✅ 模型参数已正确移动到加速器设备
============================================================
>>> DDP设备调试信息结束 <<<
============================================================
```

#### 6.2 问题情况的输出示例
```
============================================================
>>> MMEngine wrap_model DDP设备调试信息 <<<
============================================================
>>> 参数 backbone.patch_embed.proj.weight: 设备=cpu, 形状=torch.Size([384, 3, 14, 14])
>>> 参数 backbone.patch_embed.proj.bias: 设备=cpu, 形状=torch.Size([384])
>>> 总参数数量: 21234567
>>> 参数设备分布: {'cpu'}
>>> 第一个参数设备: cpu
>>> 当前GCU设备: 0
>>> 分布式环境: LOCAL_RANK=0, RANK=0, WORLD_SIZE=8
🚨 警告: 检测到模型参数仍在CPU上!
🚨 这将导致DDP设备不匹配错误!
============================================================
>>> DDP设备调试信息结束 <<<
============================================================
```

### 第7步：根据调试结果制定解决方案

#### 情况A：模型参数在正确设备上
如果调试显示模型参数已在GCU设备上，但仍然报错，问题可能在于：
- MMEngine的device_ids配置逻辑
- DDP包装器的设备检测机制

#### 情况B：模型参数仍在CPU上
如果调试显示模型参数仍在CPU上，需要：
- 检查MMEngine的模型构建和设备移动逻辑
- 在更早的时机强制移动模型到GCU
- 修改MMEngine的设备配置策略

### 第8步：恢复原始文件（调试完成后）

```bash
# 使用自动化脚本恢复
python3 scripts/debug_mmengine_ddp.py --restore

# 或手动恢复
cp /usr/local/lib/python3.8/dist-packages/mmengine/runner/runner.py.debug_backup \
   /usr/local/lib/python3.8/dist-packages/mmengine/runner/runner.py
```

## 预期结果

通过这次深度调试，我们将获得：

1. **确切的设备状态信息**：模型参数在DDP包装前的真实设备位置
2. **问题根源定位**：是设备移动失败还是DDP配置问题
3. **精确的修复方向**：基于调试结果制定针对性解决方案

## 注意事项

1. **备份重要性**：修改系统库文件前务必备份
2. **权限问题**：确保有修改MMEngine源码文件的权限
3. **容器环境**：确认在正确的Docker容器内操作
4. **日志收集**：完整保存调试输出，包括错误前的所有日志

## 故障排除

### 问题1：找不到MMEngine文件
```bash
# 查找MMEngine安装位置
python3 -c "import mmengine; print(mmengine.__file__)"
```

### 问题2：没有修改权限
```bash
# 检查文件权限
ls -la /usr/local/lib/python3.8/dist-packages/mmengine/runner/runner.py

# 如果需要，修改权限（谨慎操作）
chmod 644 /usr/local/lib/python3.8/dist-packages/mmengine/runner/runner.py
```

### 问题3：调试代码没有输出
- 检查flush=True是否正确添加
- 确认调试代码在正确的位置
- 验证训练是否真的执行到了wrap_model函数

## 联系支持

如果在执行过程中遇到问题，请提供：
1. 完整的错误信息
2. 调试脚本的输出
3. T20服务器的环境信息
4. MMEngine的版本信息

---

**重要提醒**：这是一个深度调试方案，涉及修改系统库文件。请确保在测试环境中进行，并做好完整备份。