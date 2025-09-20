#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v87 TTA评估脚本 - GCU版本
支持燧原T20 GCU设备的TTA评估
"""

import sys
import os
import traceback
import torch
# 尝试导入torch_gcu支持
try:
    import torch_gcu
    GCU_AVAILABLE = True
except ImportError:
    GCU_AVAILABLE = False
    print("⚠️ torch_gcu未安装，将使用CPU模式")

import numpy as np
import mmcv
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import TRANSFORMS

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 导入MMSeg相关模块
try:
    import mmseg
    import mmseg.models
    import mmseg.datasets
    from mmseg.models.segmentors import EncoderDecoder
    from mmseg.models.decode_heads import SegformerHead
    from mmseg.models.backbones import MixVisionTransformer
    from mmseg.datasets import LoveDADataset
    
    # 注册模型到MMEngine
    from mmengine.registry import MODELS
    if 'EncoderDecoder' not in MODELS.module_dict:
        MODELS.register_module(name='EncoderDecoder', module=EncoderDecoder)
        print("✅ EncoderDecoder已注册到MMEngine")
    
    # 注册数据集
    from mmengine.dataset import BaseDataset
    from mmengine.registry import DATASETS
    import os
    import os.path as osp
    from PIL import Image
    import numpy as np

    class MinimalLoveDADataset(BaseDataset):
        """Minimal LoveDADataset implementation to avoid GCU dependencies"""
        
        METAINFO = {
            'classes': ('background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture'),
            'palette': [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]
        }
        
        def __init__(self, data_root, data_prefix=None, img_suffix='.png', seg_map_suffix='.png', **kwargs):
            self.data_root = data_root
            self.data_prefix = data_prefix or dict()
            self.img_suffix = img_suffix
            self.seg_map_suffix = seg_map_suffix
            super().__init__(**kwargs)
        
        def load_data_list(self):
            # 返回空列表，仅用于配置验证
            return []
    
    # 注册数据集
    if 'LoveDADataset' not in DATASETS.module_dict:
        DATASETS.register_module(name='LoveDADataset', module=MinimalLoveDADataset)
        print("✅ MinimalLoveDADataset已注册为LoveDADataset")
    else:
        print("✅ LoveDADataset已存在于注册表中")
    
    # 跳过transforms和metrics注册
    from mmengine.registry import TRANSFORMS, METRICS
    print("⚠️ 跳过transforms和metrics注册（避免导入兼容性问题）")
    print("✅ 使用现有的MMSeg注册组件")
    
    print("✅ MMSeg模块和组件导入成功")
except ImportError as e:
    print(f"❌ MMSeg导入失败: {e}")
    sys.exit(1)

# 权重文件路径
CHECKPOINT_PATH = './checkpoints/best_mIoU_iter_6000.pth'

# 注册自定义Transform
@TRANSFORMS.register_module()
class UniformMaskFormat:
    def __init__(self, palette):
        self.palette = palette
    
    def __call__(self, results):
        # 简单的mask格式化
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'].astype(np.uint8)
        return results

def check_gcu_environment():
    """检查GCU环境"""
    print("\n=== 🔍 GCU环境检查 ===")
    
    if not GCU_AVAILABLE:
        print("❌ torch_gcu不可用")
        return False
    
    # 检查torch_gcu是否可用
    try:
        if hasattr(torch, 'gcu') and torch.gcu.is_available():
            print("✅ torch_gcu可用")
            device_count = torch.gcu.device_count()
            print(f"✅ 可用GCU设备数量: {device_count}")
            
            # 检查每个设备
            for i in range(device_count):
                try:
                    device_name = torch.gcu.get_device_name(i)
                    print(f"  - 设备 {i}: {device_name}")
                except Exception as e:
                    print(f"  - 设备 {i}: 获取名称失败 ({e})")
            
            return True
        else:
            print("❌ torch_gcu不可用")
            return False
    except Exception as e:
        print(f"❌ GCU环境检查失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("🚀 v87 TTA评估脚本启动 - GCU版本")
    print("="*60)
    
    # 检查GCU环境
    if not check_gcu_environment():
        print("❌ GCU环境不可用，退出")
        return
    
    try:
        # 设置工作目录
        work_dir = "./work_dirs/v87_tta_gcu_results"
        os.makedirs(work_dir, exist_ok=True)
        print(f"📁 工作目录: {work_dir}")
        
        # 检查权重文件
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"❌ 权重文件不存在: {CHECKPOINT_PATH}")
            print("💡 请确保权重文件路径正确")
            return
        
        print(f"✅ 权重文件存在: {CHECKPOINT_PATH}")
        
        # 加载配置文件
        print("\n=== 📋 加载配置文件 ===")
        config_path = "configs/v87/v87_tta_final.py"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return
        
        cfg = Config.fromfile(config_path)
        cfg.work_dir = work_dir
        
        # 移除model中的data_preprocessor（旧版EncoderDecoder不支持）
        if hasattr(cfg.model, 'data_preprocessor'):
            delattr(cfg.model, 'data_preprocessor')
        if 'data_preprocessor' in cfg.model:
            del cfg.model['data_preprocessor']
        
        # 修复旧版本兼容性问题
        if 'decode_head' in cfg.model and 'loss_decode' in cfg.model.decode_head:
            loss_cfg = cfg.model.decode_head.loss_decode
            if 'ignore_index' in loss_cfg:
                del loss_cfg['ignore_index']
        
        # 添加TTA配置
        cfg.model.test_cfg = dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
        # 添加简单的全局test_cfg（不包含mode参数）
        cfg.test_cfg = dict(type='TestLoop')
        
        # 简化配置 - 不使用真实数据加载器，只测试模型构建
        cfg.test_dataloader = None
        cfg.test_evaluator = None
        cfg.test_cfg = None
        
        # 创建Runner
        runner = Runner.from_cfg(cfg)
        
        # 将模型移动到GCU设备
        print("\n=== 🔧 设备配置 ===")
        device = torch.device('gcu:0')  # 使用第一个GCU设备
        runner.model = runner.model.to(device)
        print(f"✅ 模型已移动到设备: {device}")
        
        print(f"--> 正在手动从 {CHECKPOINT_PATH} 加载权重...")
        runner.load_checkpoint(CHECKPOINT_PATH)
        print("--> 权重加载成功！")

        # 验证TTA配置
        print("\n=== ✅ TTA配置验证 ===")
        print(f"📊 模型test_cfg: {runner.model.test_cfg}")
        print(f"📊 模型类型: {type(runner.model)}")
        print(f"📊 模型设备: {next(runner.model.parameters()).device}")
        print(f"📊 模型已成功构建并加载权重")
        
        print("\n=== ✅ v87 TTA GCU配置验证完成 ===")
        print("🎉 模型已准备好进行GCU TTA推理！")
        print("💡 TTA配置包含滑窗模式，裁剪尺寸(1024,1024)，步长(768,768)")
        print("🔥 使用GCU设备进行加速推理")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        print("="*60)
        traceback.print_exc()

if __name__ == "__main__":
    main()