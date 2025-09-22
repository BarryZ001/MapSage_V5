# torch_gcu分布式接口修复说明

## 问题背景

根据燧原T20官方文档，由于PyTorch源码的限制，需要将以下函数替换为torch_gcu.distributed模块中的对应函数：

1. `torch.distributed.destroy_process_group` → `torch_gcu.distributed.destroy_process_group`
2. `torch.distributed.batch_isend_irecv` → `torch_gcu.distributed.batch_isend_irecv`

## 修复方案

### 1. 创建统一补丁文件

创建了 `torch_gcu_distributed_patch.py` 文件，提供统一的分布式接口：

```python
from torch_gcu_distributed_patch import cleanup_distributed

# 在训练结束时调用
cleanup_distributed()
```

### 2. 修复的训练脚本

- `scripts/train_distributed_gcu_fixed.py` - 完全修复版本
- `start_8card_training_gcu_fixed.sh` - 使用修复版本的启动脚本

### 3. 需要修复的其他文件

以下文件中仍有需要修复的 `destroy_process_group` 调用：

- `scripts/train_distributed_8card_gcu.py` (已部分修复)
- `scripts/train_distributed_pytorch_ddp_8card_gcu.py`
- `scripts/train_distributed_8card_gcu_simplified.py`
- `scripts/fix_eccl_backend_issue.py`

## 使用方法

### 在T20服务器上运行

1. 使用修复版训练脚本：
```bash
bash start_8card_training_gcu_fixed.sh
```

2. 或直接运行：
```bash
python3 scripts/train_distributed_gcu_fixed.py configs/train_dinov3_mmrs1m_t20_gcu.py --launcher pytorch
```

### 在本地测试

本地环境没有torch_gcu，会自动回退到标准的torch.distributed接口。

## 官方文档参考

- 文档链接: https://support.enflame-tech.com/onlinedoc_dev_3.5/3-model/infer/torch_gcu/torch_gcu2.5/content/source/torch_gcu_distributed_support.html
- 支持的分布式API详见官方文档第9章

## 注意事项

1. 只有在T20服务器上才需要使用torch_gcu.distributed接口
2. 本地开发环境会自动回退到标准接口
3. 所有修改都向后兼容，不会影响现有功能