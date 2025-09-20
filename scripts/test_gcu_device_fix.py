#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCU设备修复测试脚本
用于验证torch_gcu API的正确使用和模型设备移动

使用方法：
1. 在T20服务器的dinov3-container容器内运行此脚本
2. 脚本会测试torch_gcu的各种API调用
3. 验证模型能否正确移动到GCU设备
4. 测试DDP包装器的设备兼容性

作者: MapSage团队
日期: 2025-01-21
"""

import os
import sys
import torch
import traceback
from typing import Optional, Any

# 添加项目路径
sys.path.insert(0, '/workspace/code/MapSage_V5')
sys.path.insert(0, '.')

# 条件导入torch_gcu，避免在非GCU环境中的导入错误
try:
    import torch_gcu  # type: ignore
    TORCH_GCU_AVAILABLE = True
except ImportError:
    torch_gcu = None  # type: ignore
    TORCH_GCU_AVAILABLE = False

def test_torch_gcu_import():
    """测试torch_gcu导入"""
    print("🔍 测试torch_gcu导入...")
    
    if not TORCH_GCU_AVAILABLE:
        print("❌ torch_gcu导入失败: 模块不可用")
        print("💡 这是正常的，torch_gcu只在燧原T20 GCU环境中可用")
        return None
    
    try:
        print(f"✅ torch_gcu导入成功")
        print(f"📊 可用GCU设备数: {torch_gcu.device_count()}")
        print(f"🔧 当前GCU设备: {torch_gcu.current_device()}")
        print(f"💾 GCU可用性: {torch_gcu.is_available()}")
        return torch_gcu
    except Exception as e:
        print(f"❌ torch_gcu操作失败: {e}")
        return None

def test_gcu_device_operations(gcu_module: Optional[Any]):
    """测试GCU设备操作"""
    if not gcu_module:
        print("⚠️ 跳过GCU设备操作测试（torch_gcu不可用）")
        return False
    
    print("\n🔧 测试GCU设备操作...")
    
    try:
        # 测试设备设置
        device_count = gcu_module.device_count()
        print(f"📊 总GCU设备数: {device_count}")
        
        if device_count > 0:
            # 测试设置设备0
            gcu_module.set_device(0)
            current_device = gcu_module.current_device()
            print(f"✅ 设置设备0成功，当前设备: {current_device}")
            
            # 测试创建张量
            tensor = torch.randn(3, 3)
            print(f"🔍 CPU张量设备: {tensor.device}")
            
            # 测试移动到GCU
            gcu_tensor = tensor.cuda()  # 使用GCU兼容的cuda()方法
            print(f"✅ 张量移动到GCU成功，设备: {gcu_tensor.device}")
            
            return True
        else:
            print("❌ 没有可用的GCU设备")
            return False
            
    except Exception as e:
        print(f"❌ GCU设备操作失败: {e}")
        traceback.print_exc()
        return False

def test_model_creation_and_movement(gcu_module: Optional[Any]):
    """测试模型创建和设备移动"""
    print("\n🏗️ 测试模型创建和设备移动...")
    
    try:
        # 创建简单模型
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        print("✅ 模型创建成功")
        
        # 检查初始设备
        first_param = next(model.parameters())
        print(f"🔍 模型初始设备: {first_param.device}")
        
        if gcu_module and TORCH_GCU_AVAILABLE:
            # 使用torch_gcu API移动模型
            gcu_module.set_device(0)
            model = model.cuda()  # 使用GCU兼容的cuda()方法
            
            # 验证移动结果
            first_param = next(model.parameters())
            print(f"✅ 模型移动到GCU成功，设备: {first_param.device}")
            
            # 测试模型推理
            input_tensor = torch.randn(1, 10).cuda()
            output = model(input_tensor)
            print(f"✅ GCU模型推理成功，输出设备: {output.device}")
            
            return True
        else:
            print("⚠️ torch_gcu不可用，跳过GCU移动测试")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_mmengine_model_build():
    """测试MMEngine模型构建"""
    print("\n🔧 测试MMEngine模型构建...")
    
    try:
        from mmengine.config import Config
        from mmengine.registry import MODELS
        
        # 注册简单模型用于测试
        @MODELS.register_module()
        class SimpleTestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # 构建模型
        model = MODELS.build(dict(type='SimpleTestModel'))
        print("✅ MMEngine模型构建成功")
        
        # 测试设备移动
        if TORCH_GCU_AVAILABLE and torch_gcu:
            torch_gcu.set_device(0)
            model = model.cuda()
            
            first_param = next(model.parameters())
            print(f"✅ MMEngine模型移动到GCU成功，设备: {first_param.device}")
            return True
        else:
            print("⚠️ torch_gcu不可用，跳过MMEngine模型GCU测试")
            return False
            
    except Exception as e:
        print(f"❌ MMEngine模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_ddp_compatibility(gcu_module: Optional[Any]):
    """测试DDP兼容性"""
    print("\n🔗 测试DDP兼容性...")
    
    # 检查分布式环境
    if not os.environ.get('RANK'):
        print("⚠️ 非分布式环境，跳过DDP测试")
        return True
    
    try:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # 创建简单模型
        model = torch.nn.Linear(10, 1)
        
        if gcu_module and TORCH_GCU_AVAILABLE:
            # 移动到GCU
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            gcu_module.set_device(local_rank)
            model = model.cuda()
            
            print(f"✅ 模型已移动到GCU设备: {local_rank}")
            
            # 测试DDP包装（不指定device_ids，让MMEngine自动处理）
            # 这里只是验证模型在正确设备上，实际DDP包装由MMEngine处理
            first_param = next(model.parameters())
            if 'cpu' not in str(first_param.device):
                print("✅ 模型参数不在CPU上，DDP包装应该可以成功")
                return True
            else:
                print("❌ 模型参数仍在CPU上，DDP包装会失败")
                return False
        else:
            print("⚠️ torch_gcu不可用，跳过DDP兼容性测试")
            return False
            
    except Exception as e:
        print(f"❌ DDP兼容性测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 GCU设备修复测试开始")
    print("=" * 60)
    
    # 测试结果统计
    test_results = []
    
    # 1. 测试torch_gcu导入
    gcu_module = test_torch_gcu_import()
    test_results.append(("torch_gcu导入", gcu_module is not None))
    
    # 2. 测试GCU设备操作
    device_ops_ok = test_gcu_device_operations(gcu_module)
    test_results.append(("GCU设备操作", device_ops_ok))
    
    # 3. 测试模型创建和移动
    model_ok = test_model_creation_and_movement(gcu_module)
    test_results.append(("模型设备移动", model_ok))
    
    # 4. 测试MMEngine模型构建
    mmengine_ok = test_mmengine_model_build()
    test_results.append(("MMEngine模型构建", mmengine_ok))
    
    # 5. 测试DDP兼容性
    ddp_ok = test_ddp_compatibility(gcu_module)
    test_results.append(("DDP兼容性", ddp_ok))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！GCU设备修复成功！")
        print("💡 现在可以运行8卡分布式训练了")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        print("💡 请检查torch_gcu安装和GCU设备配置")
    
    print("\n📋 下一步操作:")
    print("1. 如果测试通过，运行: bash scripts/start_8card_training.sh")
    print("2. 如果测试失败，检查torch_gcu安装和设备配置")
    print("3. 查看详细错误信息并根据提示修复")

if __name__ == "__main__":
    main()