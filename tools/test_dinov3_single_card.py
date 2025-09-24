#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3 + MMRS-1M 单卡测试脚本
用于验证配置文件、数据集和模型是否正确设置，然后再进行8卡分布式训练

使用方法:
python tools/test_dinov3_single_card.py [config_file]
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault('PYTORCH_GCU_ALLOC_CONF', 'backend:topsMallocAsync')
os.environ.setdefault('TORCH_ECCL_AVOID_RECORD_STREAMS', 'false')
os.environ.setdefault('TORCH_ECCL_ASYNC_ERROR_HANDLING', '3')

def test_torch_gcu():
    """测试torch_gcu环境"""
    print("🔍 测试torch_gcu环境...")
    
    try:
        import torch
        import torch_gcu
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ torch_gcu可用: {torch_gcu.is_available()}")
        
        if torch_gcu.is_available():
            device_count = torch_gcu.device_count()
            print(f"✅ GCU设备数: {device_count}")
            
            # 测试设备访问
            device = torch.device('xla:0')
            test_tensor = torch.randn(2, 2).to(device)
            print(f"✅ 设备测试成功: {device}")
            del test_tensor
            
            return True
        else:
            print("❌ torch_gcu不可用")
            return False
            
    except ImportError as e:
        print(f"❌ torch_gcu导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ torch_gcu测试失败: {e}")
        return False

def test_mmengine_mmseg():
    """测试MMEngine和MMSegmentation"""
    print("\n🔍 测试MMEngine和MMSegmentation...")
    
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        print("✅ MMEngine导入成功")
        
        import mmseg
        print("✅ MMSegmentation导入成功")
        
        # 测试自定义模块
        import mmseg_custom.models
        import mmseg_custom.datasets
        print("✅ 自定义模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_config_loading(config_path):
    """测试配置文件加载"""
    print(f"\n🔍 测试配置文件加载: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        print("✅ 配置文件加载成功")
        
        # 检查关键配置
        if hasattr(cfg, 'model'):
            print("✅ 模型配置存在")
        else:
            print("❌ 模型配置缺失")
            return False
            
        if hasattr(cfg, 'train_dataloader'):
            print("✅ 训练数据加载器配置存在")
        else:
            print("❌ 训练数据加载器配置缺失")
            return False
            
        # 显示关键配置信息
        print(f"   - 工作目录: {cfg.get('work_dir', 'N/A')}")
        print(f"   - 数据根目录: {cfg.get('data_root', 'N/A')}")
        print(f"   - 批次大小: {cfg.train_dataloader.get('batch_size', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_dataset_loading(config_path):
    """测试数据集加载"""
    print(f"\n🔍 测试数据集加载...")
    
    try:
        from mmengine.config import Config
        from mmseg_custom.datasets import MMRS1MDataset
        
        cfg = Config.fromfile(config_path)
        
        # 创建数据集实例
        dataset_cfg = cfg.train_dataloader.dataset
        print(f"   - 数据集类型: {dataset_cfg.type}")
        print(f"   - 数据根目录: {dataset_cfg.data_root}")
        
        # 实例化数据集
        dataset = MMRS1MDataset(**dataset_cfg)
        print(f"✅ 数据集实例化成功")
        print(f"   - 数据集长度: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试加载第一个样本
            sample = dataset[0]
            print(f"✅ 样本加载成功")
            print(f"   - 样本键: {list(sample.keys())}")
            
            if 'img_path' in sample:
                print(f"   - 图像路径: {sample['img_path']}")
            
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation(config_path):
    """测试模型创建"""
    print(f"\n🔍 测试模型创建...")
    
    try:
        from mmengine.config import Config
        from mmengine.registry import MODELS
        
        cfg = Config.fromfile(config_path)
        
        # 创建模型
        model = MODELS.build(cfg.model)
        print("✅ 模型创建成功")
        print(f"   - 模型类型: {type(model).__name__}")
        
        # 测试模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   - 总参数数: {total_params:,}")
        print(f"   - 可训练参数数: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_forward_pass(config_path):
    """测试单次前向传播"""
    print(f"\n🔍 测试单次前向传播...")
    
    try:
        import torch
        import torch_gcu
        from mmengine.config import Config
        from mmengine.registry import MODELS
        from mmseg_custom.datasets import MMRS1MDataset
        
        cfg = Config.fromfile(config_path)
        
        # 设置设备
        device = torch.device('xla:0' if torch_gcu.is_available() else 'cpu')
        print(f"   - 使用设备: {device}")
        
        # 创建模型并移到设备
        model = MODELS.build(cfg.model)
        model = model.to(device)
        model.eval()
        
        # 创建数据集并获取样本
        dataset_cfg = cfg.train_dataloader.dataset
        dataset = MMRS1MDataset(**dataset_cfg)
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            # 创建模拟输入
            batch_size = 1
            img_size = cfg.get('img_size', (512, 512))
            
            # 模拟输入数据
            inputs = {
                'inputs': torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device),
                'data_samples': []  # 简化的数据样本
            }
            
            print(f"   - 输入形状: {inputs['inputs'].shape}")
            
            # 前向传播
            with torch.no_grad():
                outputs = model(**inputs)
            
            print("✅ 前向传播成功")
            if hasattr(outputs, 'shape'):
                print(f"   - 输出形状: {outputs.shape}")
            elif isinstance(outputs, (list, tuple)):
                print(f"   - 输出数量: {len(outputs)}")
            
            return True
        else:
            print("⚠️ 数据集为空，跳过前向传播测试")
            return True
            
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='DINOv3单卡测试脚本')
    parser.add_argument('config', 
                        nargs='?',
                        default='configs/train_dinov3_mmrs1m_t20_gcu_8card_single_test.py',
                        help='配置文件路径')
    parser.add_argument('--skip-forward', action='store_true',
                        help='跳过前向传播测试')
    args = parser.parse_args()

    print("🚀 DINOv3 + MMRS-1M 单卡测试开始")
    print("=" * 60)
    
    # 测试项目列表
    tests = [
        ("torch_gcu环境", test_torch_gcu),
        ("MMEngine和MMSegmentation", test_mmengine_mmseg),
        ("配置文件加载", lambda: test_config_loading(args.config)),
        ("数据集加载", lambda: test_dataset_loading(args.config)),
        ("模型创建", lambda: test_model_creation(args.config)),
    ]
    
    if not args.skip_forward:
        tests.append(("前向传播", lambda: test_single_forward_pass(args.config)))
    
    # 执行测试
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        
        try:
            success = test_func()
            end_time = time.time()
            duration = end_time - start_time
            
            results.append((test_name, success, duration))
            
            if success:
                print(f"✅ {test_name} 测试通过 ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} 测试失败 ({duration:.2f}s)")
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            results.append((test_name, False, duration))
            print(f"❌ {test_name} 测试异常: {e} ({duration:.2f}s)")
    
    # 显示测试结果汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name:<20} {status} ({duration:.2f}s)")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以进行8卡分布式训练")
        print("\n🚀 启动8卡分布式训练:")
        print(f"   bash tools/start_dinov3_8card_training.sh {args.config}")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请修复后再进行8卡训练")
        return 1

if __name__ == '__main__':
    sys.exit(main())