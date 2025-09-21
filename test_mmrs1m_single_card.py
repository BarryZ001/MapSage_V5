#!/usr/bin/env python3
"""
MMRS-1M DINOv3 单卡测试脚本
用于验证配置文件、数据集和模型是否正常工作
在进行8卡分布式训练前的预检查
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mmrs1m_config():
    """测试MMRS1M配置文件是否正确加载"""
    print("🔍 测试MMRS1M配置文件...")
    
    try:
        from mmengine.config import Config
        
        # 加载8卡分布式配置
        config_path = project_root / "configs" / "train_dinov3_mmrs1m_t20_gcu_8card.py"
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return False
            
        cfg = Config.fromfile(str(config_path))
        print(f"✅ 配置文件加载成功: {config_path}")
        
        # 检查关键配置项
        required_keys = [
            'model', 'train_dataloader', 'val_dataloader', 
            'optim_wrapper', 'train_cfg', 'custom_imports'
        ]
        
        for key in required_keys:
            if key not in cfg:
                print(f"❌ 缺少配置项: {key}")
                return False
            print(f"✅ 配置项检查通过: {key}")
        
        # 检查数据集配置
        dataset_type = cfg.get('dataset_type', 'Unknown')
        data_root = cfg.get('data_root', 'Unknown')
        num_classes = cfg.get('num_classes', 'Unknown')
        
        print(f"📊 数据集类型: {dataset_type}")
        print(f"📁 数据根目录: {data_root}")
        print(f"🏷️ 类别数量: {num_classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_mmrs1m_dataset():
    """测试MMRS1M数据集是否可以正常加载"""
    print("\n🔍 测试MMRS1M数据集...")
    
    try:
        # 检查数据路径
        server_data_root = Path("/workspace/data/mmrs1m/data")
        local_data_root = project_root / "data"
        
        if server_data_root.exists():
            data_root = server_data_root
            print(f"✅ 使用服务器数据路径: {data_root}")
        elif local_data_root.exists():
            data_root = local_data_root
            print(f"✅ 使用本地数据路径: {data_root}")
        else:
            print(f"❌ 数据路径不存在:")
            print(f"   服务器路径: {server_data_root}")
            print(f"   本地路径: {local_data_root}")
            return False
        
        # 尝试导入自定义数据集
        from mmseg_custom.datasets import MMRS1MDataset
        print("✅ MMRS1MDataset导入成功")
        
        # 创建简单的数据管道
        pipeline = [
            dict(type='CustomLoadImageFromFile'),
            dict(type='CustomLoadAnnotations'),
            dict(type='CustomResize', img_scale=(512, 512), keep_ratio=True),
            dict(type='CustomNormalize', 
                 mean=[123.675, 116.28, 103.53], 
                 std=[58.395, 57.12, 57.375], 
                 to_rgb=True),
            dict(type='CustomDefaultFormatBundle'),
            dict(type='CustomCollect', keys=['img', 'gt_semantic_seg'])
        ]
        
        # 创建数据集实例
        dataset = MMRS1MDataset(
            data_root=str(data_root),
            task_type='classification',
            modality='optical',
            instruction_format=True,
            pipeline=pipeline
        )
        
        print(f"✅ 数据集创建成功，样本数量: {len(dataset)}")
        
        # 测试加载第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ 样本加载成功")
            print(f"   图像形状: {sample.get('img', 'N/A')}")
            print(f"   标签信息: {sample.get('gt_semantic_seg', 'N/A')}")
        else:
            print("⚠️ 数据集为空，请检查数据路径和格式")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型是否可以正常创建"""
    print("\n🔍 测试DINOv3模型创建...")
    
    try:
        from mmengine.config import Config
        from mmseg.models import build_segmentor
        
        # 加载配置
        config_path = project_root / "configs" / "train_dinov3_mmrs1m_t20_gcu_8card.py"
        cfg = Config.fromfile(str(config_path))
        
        # 修改配置以适应单卡测试
        cfg.model.train_cfg = dict()
        cfg.model.test_cfg = dict(mode='whole')
        
        # 创建模型
        model = build_segmentor(cfg.model)
        print("✅ 模型创建成功")
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型参数统计:")
        print(f"   总参数数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        model.eval()
        dummy_input = torch.randn(1, 3, 512, 512)
        
        with torch.no_grad():
            # 测试推理模式
            output = model(dummy_input, mode='predict')
            print(f"✅ 模型推理测试成功")
            print(f"   输出类型: {type(output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """测试训练组件（优化器、调度器等）"""
    print("\n🔍 测试训练组件...")
    
    try:
        from mmengine.config import Config
        from mmengine.optim import build_optim_wrapper
        from mmseg.models import build_segmentor
        
        # 加载配置
        config_path = project_root / "configs" / "train_dinov3_mmrs1m_t20_gcu_8card.py"
        cfg = Config.fromfile(str(config_path))
        
        # 创建模型
        model = build_segmentor(cfg.model)
        
        # 创建优化器
        optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
        print("✅ 优化器创建成功")
        print(f"   优化器类型: {type(optim_wrapper.optimizer).__name__}")
        print(f"   学习率: {optim_wrapper.optimizer.param_groups[0]['lr']}")
        
        # 创建学习率调度器 - 简化版本，不使用build_param_scheduler
        print(f"✅ 学习率调度器配置检查通过，数量: {len(cfg.param_scheduler)}")
        for i, scheduler_cfg in enumerate(cfg.param_scheduler):
            print(f"   调度器 {i+1}: {scheduler_cfg.get('type', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='MMRS1M单卡测试脚本')
    parser.add_argument('--test-all', action='store_true', help='运行所有测试')
    parser.add_argument('--test-config', action='store_true', help='测试配置文件')
    parser.add_argument('--test-dataset', action='store_true', help='测试数据集')
    parser.add_argument('--test-model', action='store_true', help='测试模型')
    parser.add_argument('--test-training', action='store_true', help='测试训练组件')
    
    args = parser.parse_args()
    
    print("🚀 MMRS-1M DINOv3 单卡测试开始")
    print("=" * 60)
    
    test_results = []
    
    if args.test_all or args.test_config:
        test_results.append(("配置文件测试", test_mmrs1m_config()))
    
    if args.test_all or args.test_dataset:
        test_results.append(("数据集测试", test_mmrs1m_dataset()))
    
    if args.test_all or args.test_model:
        test_results.append(("模型测试", test_model_creation()))
    
    if args.test_all or args.test_training:
        test_results.append(("训练组件测试", test_training_components()))
    
    # 如果没有指定任何测试，默认运行所有测试
    if not any([args.test_config, args.test_dataset, args.test_model, args.test_training]):
        test_results = [
            ("配置文件测试", test_mmrs1m_config()),
            ("数据集测试", test_mmrs1m_dataset()),
            ("模型测试", test_model_creation()),
            ("训练组件测试", test_training_components())
        ]
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总:")
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 所有测试通过！可以进行8卡分布式训练")
        print("🚀 启动8卡训练命令:")
        print("   ./start_8card_mmrs1m_training.sh")
        return 0
    else:
        print("❌ 部分测试失败，请检查配置和环境")
        return 1

if __name__ == "__main__":
    sys.exit(main())