#!/usr/bin/env python3
"""
torch_gcu分布式接口补丁
根据官方文档要求，替换特定的torch.distributed函数为torch_gcu.distributed版本
"""

import torch.distributed as dist

# 全局标志
USE_GCU_DISTRIBUTED = False
gcu_dist = None

def init_gcu_distributed():
    """初始化torch_gcu分布式支持"""
    global USE_GCU_DISTRIBUTED, gcu_dist
    
    try:
        # 动态导入torch_gcu.distributed模块
        import importlib
        gcu_dist_module = importlib.import_module('torch_gcu.distributed')
        gcu_dist = gcu_dist_module
        USE_GCU_DISTRIBUTED = True
        print("✅ torch_gcu.distributed模块导入成功")
        return True
    except (ImportError, ModuleNotFoundError) as e:
        print(f"⚠️ torch_gcu.distributed导入失败: {e}")
        USE_GCU_DISTRIBUTED = False
        gcu_dist = None
        return False

def destroy_process_group():
    """
    替换torch.distributed.destroy_process_group
    根据官方文档要求使用torch_gcu.distributed.destroy_process_group
    """
    if USE_GCU_DISTRIBUTED and gcu_dist is not None:
        try:
            gcu_dist.destroy_process_group()
            print("✅ 使用torch_gcu.distributed.destroy_process_group清理完成")
        except Exception as e:
            print(f"⚠️ torch_gcu分布式清理失败: {e}")
            # 回退到标准方法
            dist.destroy_process_group()
            print("✅ 回退到torch.distributed.destroy_process_group")
    else:
        dist.destroy_process_group()
        print("✅ 使用torch.distributed.destroy_process_group清理完成")

def batch_isend_irecv(*args, **kwargs):
    """
    替换torch.distributed.batch_isend_irecv
    根据官方文档要求使用torch_gcu.distributed.batch_isend_irecv
    """
    if USE_GCU_DISTRIBUTED and gcu_dist is not None:
        try:
            return gcu_dist.batch_isend_irecv(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ torch_gcu batch_isend_irecv失败: {e}")
            # 回退到标准方法
            return dist.batch_isend_irecv(*args, **kwargs)
    else:
        return dist.batch_isend_irecv(*args, **kwargs)

def cleanup_distributed():
    """清理分布式训练环境的统一接口"""
    if dist.is_initialized():
        destroy_process_group()

# 自动初始化
init_gcu_distributed()

# 导出接口
__all__ = [
    'init_gcu_distributed',
    'destroy_process_group', 
    'batch_isend_irecv',
    'cleanup_distributed',
    'USE_GCU_DISTRIBUTED',
    'gcu_dist'
]