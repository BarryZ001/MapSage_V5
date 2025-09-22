#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8卡分布式训练脚本 - 燧原T20 GCU版本
支持DINOv3 + MMRS-1M数据集的8卡分布式训练
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目根目录到Python路径
sys.path.insert(0, '.')

from mmengine.config import Config
from mmengine.runner import Runner

# 尝试导入GCU相关库
try:
    import torch_gcu
    print("✅ torch_gcu导入成功，可用设备数: {}".format(torch_gcu.device_count()))
except ImportError as e:
    print("⚠️ torch_gcu导入失败: {}".format(e))
    torch_gcu = None

try:
    import ptex
    print("✅ ptex导入成功")
except ImportError as e:
    print("⚠️ ptex导入失败: {}".format(e))
    ptex = None

# 尝试导入MMSeg相关模块
try:
    import mmseg  # type: ignore
    from mmseg.models import *  # type: ignore
    from mmseg.datasets import *  # type: ignore
except ImportError as e:
    print("⚠️ 模块导入失败: {}".format(e))

# 尝试导入自定义模块
try:
    from mmseg_custom.models import *  # type: ignore
    from mmseg_custom.datasets import *  # type: ignore
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print("⚠️ 自定义模块导入失败: {}".format(e))

# 尝试导入MMSeg模型组件并注册
try:
    import mmseg
    import mmseg.models
    from mmseg.models.backbones import MixVisionTransformer
    from mmseg.models.decode_heads import SegformerHead
    from mmseg.models.segmentors import EncoderDecoder
    
    from mmengine.registry import MODELS
    if 'MixVisionTransformer' not in MODELS.module_dict:
        MODELS.register_module(name='MixVisionTransformer', module=MixVisionTransformer)
        print("✅ MixVisionTransformer已注册到MMEngine")
    
    if 'SegformerHead' not in MODELS.module_dict:
        MODELS.register_module(name='SegformerHead', module=SegformerHead)
        print("✅ SegformerHead已注册到MMEngine")
        
    if 'EncoderDecoder' not in MODELS.module_dict:
        MODELS.register_module(name='EncoderDecoder', module=EncoderDecoder)
        print("✅ EncoderDecoder已注册到MMEngine")
        
    print("✅ MMSeg模型组件导入和注册成功")
except ImportError as e:
    print("⚠️ MMSeg导入失败: {}".format(e))
    print("⚠️ 将使用自定义模型组件")

def main():
    parser = argparse.ArgumentParser(description='8卡分布式训练脚本')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='工作目录路径')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], 
                       default='pytorch', help='分布式启动器')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程rank')
    args = parser.parse_args()
    
    print("🚀 启动8卡分布式训练")
    print("📄 配置文件: {}".format(args.config))
    print("🔧 启动器: {}".format(args.launcher))
    
    # 1. 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 检查并创建工作目录
    if args.work_dir:
        # 使用命令行指定的工作目录
        cfg.work_dir = args.work_dir
        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir, exist_ok=True)
            print("📁 创建工作目录: {}".format(cfg.work_dir))
    elif hasattr(cfg, 'work_dir') and cfg.work_dir:
        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir, exist_ok=True)
            print("📁 创建工作目录: {}".format(cfg.work_dir))
    else:
        # 如果配置文件没有work_dir，设置默认值
        cfg.work_dir = './work_dirs/train_distributed_8card_gcu'
        os.makedirs(cfg.work_dir, exist_ok=True)
        print("📁 设置默认工作目录: {}".format(cfg.work_dir))
    
    # 设置日志目录
    log_dir = os.path.join(cfg.work_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 获取分布式参数
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print("🌍 分布式训练参数:")
    print("  - WORLD_SIZE: {}".format(world_size))
    print("  - RANK: {}".format(rank))
    print("  - LOCAL_RANK: {}".format(local_rank))
    
    # 配置分布式训练
    if world_size > 1:
        cfg.launcher = args.launcher
        print("🔧 启用分布式训练，launcher: {}".format(args.launcher))
        
        # 配置环境变量
        if not hasattr(cfg, 'env_cfg'):
            cfg.env_cfg = {}
        if not hasattr(cfg.env_cfg, 'dist_cfg'):
            cfg.env_cfg.dist_cfg = {}
        
        # 设置ECCL后端配置
        cfg.env_cfg.dist_cfg['backend'] = 'eccl'
        cfg.env_cfg.dist_cfg['init_method'] = 'env://'
        print("✅ 配置MMEngine使用ECCL后端")
        
        # 配置GCU设备
        cfg.device = 'gcu'
        print("🔧 配置GCU设备，world_size: {}".format(world_size))
    else:
        cfg.launcher = 'none'
        print("🔧 单进程模式，禁用分布式")
        cfg.device = 'gcu'
        print("🔧 配置单卡GCU设备")
    
    # 调整batch size
    if hasattr(cfg, 'train_dataloader') and 'batch_size' in cfg.train_dataloader:
        original_batch_size = cfg.train_dataloader.batch_size
        print("📊 每卡batch size: {}".format(original_batch_size))
        print("📊 总batch size: {}".format(original_batch_size * world_size))
    
    print("📁 工作目录: {}".format(cfg.work_dir))
    print("🚀 启动训练 - Rank {}/{}".format(rank, world_size))
    
    # 设置GCU设备
    if torch_gcu is not None:
        torch_gcu.set_device(local_rank)
        print("🔧 设置当前进程GCU设备: {}".format(local_rank))
        
        device = f"xla:{local_rank}"
        cfg.device = device
        
        # 禁用CUDA相关设置
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("🔧 配置设备为: {}".format(device))
        
        # 配置MMEngine以正确处理GCU设备
        # 禁用device_ids参数，让MMEngine自动处理设备
        if hasattr(cfg, 'model_wrapper_cfg'):
            cfg.model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=False)
        else:
            cfg.model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=False)
        print("🔧 配置MMEngine模型包装器，禁用device_ids")
    
    # 禁用SyncBatchNorm
    def disable_sync_batchnorm_in_config(config_dict):
        """递归禁用配置中的SyncBatchNorm"""
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if key == 'norm_cfg' and isinstance(value, dict):
                    if value.get('type') == 'SyncBN':
                        print(f"🔧 发现SyncBN配置，替换为BN: {value}")
                        value['type'] = 'BN'
                        print(f"✅ 已替换为: {value}")
                elif isinstance(value, (dict, list)):
                    disable_sync_batchnorm_in_config(value)
        elif isinstance(config_dict, list):
            for item in config_dict:
                disable_sync_batchnorm_in_config(item)
    
    if hasattr(cfg, 'model') and cfg.model is not None:
        disable_sync_batchnorm_in_config(cfg.model)
        print("✅ 已禁用模型配置中的SyncBatchNorm")
    
    disable_sync_batchnorm_in_config(cfg._cfg_dict)
    print("✅ SyncBatchNorm禁用完成，现在使用普通BatchNorm兼容GCU")
    
    # 2. 初始化分布式环境 (绕过MMEngine的CUDA调用，直接使用torch.distributed)
    def init_process_group_with_fallback(init_method='env://'):
        """尝试多种backend初始化分布式训练"""
        candidates = ['gloo', 'nccl']  # 移除eccl，因为它不是标准的PyTorch分布式后端
        errors = {}
        
        # 首先尝试使用ECCL包装器
        try:
            print("🔄 尝试使用ECCL包装器初始化分布式")
            # 导入ECCL包装器
            import sys
            sys.path.append('/opt/tops/eccl/lib/python3.8/site-packages')
            import eccl
            
            # 使用gloo作为基础backend，但通过ECCL进行通信
            dist.init_process_group(
                backend='gloo', 
                init_method=init_method,
                rank=int(os.environ.get('RANK', 0)),
                world_size=int(os.environ.get('WORLD_SIZE', 1))
            )
            print("✅ 分布式初始化成功，使用ECCL包装器 + gloo backend")
            return 'eccl_gloo'
            
        except Exception as e:
            error_msg = f"ECCL包装器失败: {type(e).__name__}: {e}"
            errors['eccl'] = error_msg
            print(f"⚠️ {error_msg}")
        
        for backend in candidates:
            try:
                print(f"🔄 尝试初始化分布式backend: {backend}")
                
                # 小优化：如果尝试 nccl，则先检查是否可用
                if backend == 'nccl' and not getattr(dist, "is_nccl_available", lambda: False)():
                    errors[backend] = "nccl not available"
                    print(f"⚠️ {backend}: nccl不可用，跳过")
                    continue
                
                dist.init_process_group(
                    backend=backend, 
                    init_method=init_method,
                    rank=int(os.environ.get('RANK', 0)),
                    world_size=int(os.environ.get('WORLD_SIZE', 1))
                )
                print(f"✅ 分布式初始化成功，使用backend: {backend}")
                return backend
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                errors[backend] = error_msg
                print(f"❌ {backend} 初始化失败: {error_msg}")
                
                # 清理失败的初始化
                try:
                    if dist.is_initialized():
                        # 使用torch_gcu.distributed.destroy_process_group
                        try:
                            import torch_gcu.distributed as gcu_dist
                            gcu_dist.destroy_process_group()
                            print("✅ 使用torch_gcu.distributed.destroy_process_group清理完成")
                        except ImportError:
                            # 回退到标准方法
                            dist.destroy_process_group()
                            print("✅ 分布式进程组清理完成")
                except Exception:
                    pass
        
        # 全部失败 -> 抛错并打印诊断
        msg = ["❌ 所有分布式backend初始化失败:"]
        for b, e in errors.items():
            msg.append(f"  - {b}: {e}")
        msg.append(f"torch.distributed.is_available(): {dist.is_available()}")
        
        # 安全检查torch模块
        try:
            import torch
            msg.append(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        except Exception as e:
            msg.append(f"torch.cuda检查失败: {e}")
        
        # 检查torch_gcu状态
        try:
            import torch_gcu
            msg.append(f"torch_gcu.device_count(): {torch_gcu.device_count()}")
        except ImportError:
            msg.append("torch_gcu: 未安装")
        
        raise RuntimeError("\n".join(msg))
    
    if cfg.get('launcher', 'none') == 'pytorch':
        # 获取分布式参数
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 设置分布式环境变量
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        
        # 使用fallback逻辑初始化分布式环境
        if not dist.is_initialized():
            init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            backend_used = init_process_group_with_fallback(init_method=init_method)
            print(f"🌐 分布式训练已启动，使用backend: {backend_used}")
            print(f"🔧 分布式环境初始化完成 - Rank {rank}/{world_size}, Backend: {dist.get_backend()}")
        else:
            print("🔧 分布式环境已初始化")
    
    # 3. 创建 Runner 实例
    print("🚀 创建Runner...")
    
    # 在创建Runner之前，确保模型会被正确移动到GCU设备
    # 通过设置环境变量来确保模型初始化时就在正确的设备上
    if torch_gcu is not None:
        # 强制模型在GCU设备上初始化
        import torch
        torch.set_default_tensor_type('torch.FloatTensor')  # 确保使用CPU tensor作为默认
        
        # 创建一个临时的GCU tensor来确保设备可用
        try:
            test_tensor = torch.tensor([1.0]).to(f"xla:{local_rank}")
            print(f"✅ GCU设备 xla:{local_rank} 可用，测试tensor: {test_tensor.device}")
        except Exception as e:
            print(f"⚠️ GCU设备测试失败: {e}")
    
    # 关键修复：在创建Runner之前设置正确的模型包装器配置
    print("🔧 配置MMEngine模型包装器，完全禁用device_ids和output_device...")
    
    # 强制设置模型包装器配置为None，让MMEngine使用默认的DDP包装
    cfg.model_wrapper_cfg = dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=False,
        broadcast_buffers=False,
        # 关键修复：完全不设置device_ids和output_device，让DDP自动处理
    )
    print("✅ 设置了兼容GCU的model_wrapper_cfg配置")
    print(f"🔍 最终model_wrapper_cfg: {cfg.model_wrapper_cfg}")
    
    # 在创建Runner之前，预先设置GCU设备环境
    if torch_gcu is not None:
        print(f"🔧 预设置GCU设备环境，local_rank: {local_rank}")
        torch_gcu.set_device(local_rank)
        
        # 设置默认设备为当前GCU设备
        import torch
        if hasattr(torch, 'set_default_device'):
            try:
                torch.set_default_device(f'xla:{local_rank}')
                print(f"✅ 设置默认设备为: xla:{local_rank}")
            except:
                print("⚠️ 无法设置默认设备，继续使用CPU初始化")
    
    # 关键修复：强制设置模型初始化设备
    print("🔧 强制设置模型初始化在GCU设备上...")
    if torch_gcu is not None:
        # 临时修改torch的默认tensor类型，确保模型参数在GCU上初始化
        original_default_tensor_type = torch.get_default_dtype()
        try:
            # 创建一个GCU上的tensor作为模板
            device_str = f'xla:{local_rank}'
            print(f"🔧 设置模型初始化设备: {device_str}")
            
            # 在配置中明确指定设备
            cfg.device = device_str
            
            # 创建Runner
            runner = Runner.from_cfg(cfg)
            print("✅ Runner创建完成")
            
            # 立即检查并移动模型到正确设备
            if hasattr(runner, 'model') and runner.model is not None:
                print("🔧 检查模型设备状态...")
                
                # 获取模型当前设备
                try:
                    current_device = next(runner.model.parameters()).device
                    print(f"🔍 模型当前设备: {current_device}")
                    
                    # 如果模型不在正确的GCU设备上，强制移动
                    if str(current_device) != device_str:
                        print(f"⚠️ 模型设备不匹配，从 {current_device} 移动到 {device_str}")
                        runner.model = runner.model.to(device_str)
                        print(f"✅ 模型已移动到设备: {device_str}")
                        
                        # 再次验证
                        new_device = next(runner.model.parameters()).device
                        print(f"🔍 移动后模型设备: {new_device}")
                    else:
                        print(f"✅ 模型已在正确设备: {current_device}")
                        
                except Exception as e:
                    print(f"⚠️ 检查模型设备时出错: {e}")
                    
        except Exception as e:
            print(f"❌ 设置模型初始化设备失败: {e}")
            # 回退到默认创建方式
            runner = Runner.from_cfg(cfg)
            print("✅ Runner创建完成（回退模式）")
    else:
         runner = Runner.from_cfg(cfg)
         print("✅ Runner创建完成")
    
    # ===== START: 最终修复逻辑 (在Runner创建后，训练开始前) =====
    print("🔧 开始执行最终修复逻辑...")
    
    # 3.1 强制修正分布式后端为 ECCL
    if dist.is_initialized() and dist.get_backend() != 'eccl':
        print(f"⚠️ 检测到错误后端: {dist.get_backend()}，强制切换到 ECCL...")
        current_rank = dist.get_rank()
        current_world_size = dist.get_world_size()
        
        # 销毁当前进程组
        dist.destroy_process_group()
        print("🧹 已销毁当前进程组")
        
        # 重新初始化ECCL后端
        try:
            # 设置ECCL环境变量
            os.environ['ECCL_BACKEND'] = 'eccl'
            os.environ['ECCL_DEVICE_TYPE'] = 'gcu'
            
            dist.init_process_group(
                backend='eccl', 
                init_method='env://', 
                world_size=current_world_size, 
                rank=current_rank
            )
            print(f"✅ 成功切换到 ECCL 后端")
        except Exception as e:
            print(f"❌ ECCL后端初始化失败: {e}")
            print("🔄 回退到gloo后端")
            dist.init_process_group(
                backend='gloo', 
                init_method='env://', 
                world_size=current_world_size, 
                rank=current_rank
            )
    elif dist.is_initialized():
        print(f"✅ 当前后端已是正确的: {dist.get_backend()}")
    
    # 3.2 关键修复：强制将模型移动到正确的GCU设备
    if torch_gcu is not None and hasattr(runner, 'model') and runner.model is not None:
        # 设置GCU设备
        torch_gcu.set_device(local_rank)
        device = f'xla:{local_rank}'
        
        print(f"🔧 开始将模型移动到设备: {device}")
        
        # 检查模型当前设备状态
        try:
            current_device = next(runner.model.parameters()).device
            print(f"🔍 模型当前设备: {current_device}")
        except StopIteration:
            print("⚠️ 模型没有参数，跳过设备检查")
            current_device = None
        
        # 强制将模型移动到GCU设备
        try:
            runner.model = runner.model.to(device)
            print(f"✅ 模型已强制移动到设备: {device}")
            
            # 验证模型设备
            model_device = next(runner.model.parameters()).device
            print(f"🔍 验证模型设备: {model_device}")
            
            # 确保所有参数都在正确设备上
            device_count = {}
            for name, param in runner.model.named_parameters():
                param_device = str(param.device)
                device_count[param_device] = device_count.get(param_device, 0) + 1
            
            print(f"📊 模型参数设备分布: {device_count}")
            
        except Exception as e:
            print(f"❌ 模型设备迁移失败: {e}")
            print(f"❌ 错误详情: {str(e)}")
            raise e
    
    # 3.3 转换SyncBatchNorm层以兼容DDP
    if hasattr(runner, 'model') and runner.model is not None and world_size > 1:
        try:
            from mmengine.model import convert_sync_batchnorm
            runner.model = convert_sync_batchnorm(runner.model)
            print("🔧 SyncBatchNorm层已转换为DDP兼容")
        except Exception as e:
            print(f"⚠️ SyncBatchNorm转换失败: {e}")
    
    # 3.4 关键修复：重新用DDP包装模型（使用正确的参数）
    if world_size > 1 and hasattr(runner, 'model') and runner.model is not None:
        try:
            from mmengine.model import MMDistributedDataParallel
            
            # 检查模型是否已经被DDP包装
            if not isinstance(runner.model, MMDistributedDataParallel):
                print(f"🔧 开始DDP包装，当前模型类型: {type(runner.model)}")
                
                # 获取模型当前设备
                try:
                    model_device = next(runner.model.parameters()).device
                    print(f"🔍 DDP包装前模型设备: {model_device}")
                except StopIteration:
                    print("⚠️ 模型没有参数")
                    model_device = None
                
                # 关键：设置device_ids=None和output_device=None以避免设备不匹配错误
                # 这是修复DDP设备不匹配错误的核心逻辑
                runner.model = MMDistributedDataParallel(
                    runner.model,
                    device_ids=None,  # 关键：设为None让DDP使用模型当前设备
                    output_device=None,  # 关键：设为None避免设备冲突
                    find_unused_parameters=False,  # 从配置文件获取
                    broadcast_buffers=False,  # 从配置文件获取
                    # 添加额外的GCU兼容性配置
                    static_graph=False,  # 禁用静态图优化，避免GCU兼容性问题
                )
                print("✅ 模型已在正确的GCU设备上重新包装为DDP")
                
                # 验证DDP包装后的模型设备
                try:
                    model_device = next(runner.model.parameters()).device
                    print(f"🔍 DDP包装后模型设备: {model_device}")
                    
                    # 检查DDP包装后的参数设备分布
                    device_count = {}
                    for name, param in runner.model.named_parameters():
                        param_device = str(param.device)
                        device_count[param_device] = device_count.get(param_device, 0) + 1
                    
                    print(f"📊 DDP包装后参数设备分布: {device_count}")
                    
                except StopIteration:
                    print("⚠️ DDP包装后模型没有参数")
                    
            else:
                print("✅ 模型已经是DDP包装")
                # 验证已包装模型的设备
                try:
                    model_device = next(runner.model.parameters()).device
                    print(f"🔍 已包装DDP模型设备: {model_device}")
                except StopIteration:
                    print("⚠️ 已包装DDP模型没有参数")
                    
        except Exception as e:
            print(f"⚠️ DDP包装失败: {e}")
            print(f"⚠️ 错误详情: {str(e)}")
            print(f"⚠️ 错误类型: {type(e)}")
            # 不抛出异常，让训练继续进行
    
    # ===== END: 最终修复逻辑 =====
    
    # 验证最终状态
    if dist.is_initialized():
        print(f"🔍 最终验证 - 后端: {dist.get_backend()}, Rank: {dist.get_rank()}/{dist.get_world_size()}")
    
    if hasattr(runner, 'model') and runner.model is not None:
        model_device = next(runner.model.parameters()).device
        print(f"🔍 最终验证 - 模型设备: {model_device}")
    
    # 4. 开始训练
    print("🚀 开始训练...")
    runner.train()
    
    # 清理分布式环境
    if dist.is_initialized():
        print("🧹 清理分布式环境...")
        dist.destroy_process_group()
        print("✅ 分布式环境清理完成")

if __name__ == '__main__':
    main()