# scripts/train.py (最终简化版)

import argparse
import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules  # type: ignore
from mmseg.registry import MODELS, DATASETS, TRANSFORMS, VISUALIZERS  # type: ignore
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS  # type: ignore
from mmengine.registry import MODELS as MMENGINE_MODELS  # type: ignore

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation training script')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    # 注册所有模块 (非常重要!)
    register_all_modules(init_default_scope=False)
    
    # 检查关键模型是否已注册
    print(f"📋 MODELS registry has {len(MODELS.module_dict)} modules")
    print(f"📋 EncoderDecoder in MODELS: {'EncoderDecoder' in MODELS.module_dict}")
    print(f"📋 MMENGINE_MODELS registry has {len(MMENGINE_MODELS.module_dict)} modules")
    print(f"📋 EncoderDecoder in MMENGINE_MODELS: {'EncoderDecoder' in MMENGINE_MODELS.module_dict}")
    
    # 确保关键模型已注册到mmengine注册表
    try:
        from mmseg.models import EncoderDecoder  # type: ignore
        # 注册到mmseg注册表（如果还没有）
        if 'EncoderDecoder' not in MODELS.module_dict:
            MODELS.register_module(module=EncoderDecoder, force=True)
            print("✅ EncoderDecoder registered to mmseg registry")
        # 注册到mmengine注册表
        if 'EncoderDecoder' not in MMENGINE_MODELS.module_dict:
            MMENGINE_MODELS.register_module(module=EncoderDecoder, force=True)
            print("✅ EncoderDecoder registered to mmengine registry")
    except ImportError as e:
        print(f"⚠️ Failed to import EncoderDecoder: {e}")
        # 尝试从其他位置导入
        try:
            from mmseg.models.segmentors import EncoderDecoder  # type: ignore
            MODELS.register_module(module=EncoderDecoder, force=True)
            MMENGINE_MODELS.register_module(module=EncoderDecoder, force=True)
            print("✅ EncoderDecoder imported from segmentors and registered")
        except ImportError:
            print("❌ Could not import EncoderDecoder from any location")
    
    # 确保数据预处理器已注册
    try:
        from mmseg.models.data_preprocessor import SegDataPreProcessor  # type: ignore
        if 'SegDataPreProcessor' not in MODELS.module_dict:
            MODELS.register_module(module=SegDataPreProcessor, force=True)
            print("✅ SegDataPreProcessor registered to mmseg registry")
        if 'SegDataPreProcessor' not in MMENGINE_MODELS.module_dict:
            MMENGINE_MODELS.register_module(module=SegDataPreProcessor, force=True)
            print("✅ SegDataPreProcessor registered to mmengine registry")
    except ImportError as e:
        print(f"⚠️ Failed to import SegDataPreProcessor: {e}")
        # 尝试从其他位置导入
        try:
            from mmseg.models import SegDataPreProcessor  # type: ignore
            MODELS.register_module(module=SegDataPreProcessor, force=True)
            MMENGINE_MODELS.register_module(module=SegDataPreProcessor, force=True)
            print("✅ SegDataPreProcessor imported and registered")
        except ImportError:
            print("❌ Could not import SegDataPreProcessor from any location")
    
    # 确保MixVisionTransformer已注册
    try:
        from mmseg.models.backbones import MixVisionTransformer  # type: ignore
        if 'MixVisionTransformer' not in MODELS.module_dict:
            MODELS.register_module(module=MixVisionTransformer, force=True)
            print("✅ MixVisionTransformer registered to mmseg registry")
        if 'MixVisionTransformer' not in MMENGINE_MODELS.module_dict:
            MMENGINE_MODELS.register_module(module=MixVisionTransformer, force=True)
            print("✅ MixVisionTransformer registered to mmengine registry")
    except ImportError as e:
        print(f"⚠️ Failed to import MixVisionTransformer: {e}")
        # 尝试从其他位置导入
        try:
            from mmseg.models import MixVisionTransformer  # type: ignore
            MODELS.register_module(module=MixVisionTransformer, force=True)
            MMENGINE_MODELS.register_module(module=MixVisionTransformer, force=True)
            print("✅ MixVisionTransformer imported and registered")
        except ImportError:
            print("❌ Could not import MixVisionTransformer from any location")
    
    # 确保SegformerHead已注册
    try:
        from mmseg.models.decode_heads import SegformerHead  # type: ignore
        if 'SegformerHead' not in MODELS.module_dict:
            MODELS.register_module(module=SegformerHead, force=True)
            print("✅ SegformerHead registered to mmseg registry")
        if 'SegformerHead' not in MMENGINE_MODELS.module_dict:
            MMENGINE_MODELS.register_module(module=SegformerHead, force=True)
            print("✅ SegformerHead registered to mmengine registry")
    except ImportError as e:
        print(f"⚠️ Failed to import SegformerHead: {e}")
        # 尝试从其他位置导入
        try:
            from mmseg.models import SegformerHead  # type: ignore
            MODELS.register_module(module=SegformerHead, force=True)
            MMENGINE_MODELS.register_module(module=SegformerHead, force=True)
            print("✅ SegformerHead imported and registered")
        except ImportError:
            print("❌ Could not import SegformerHead from any location")
    
    # 确保CrossEntropyLoss已注册
    try:
        from mmseg.models.losses import CrossEntropyLoss  # type: ignore
        if 'CrossEntropyLoss' not in MODELS.module_dict:
            MODELS.register_module(module=CrossEntropyLoss, force=True)
            print("✅ CrossEntropyLoss registered to mmseg registry")
        if 'CrossEntropyLoss' not in MMENGINE_MODELS.module_dict:
            MMENGINE_MODELS.register_module(module=CrossEntropyLoss, force=True)
            print("✅ CrossEntropyLoss registered to mmengine registry")
    except ImportError as e:
        print(f"⚠️ Failed to import CrossEntropyLoss: {e}")
        # 尝试从其他位置导入
        try:
            from mmseg.models import CrossEntropyLoss  # type: ignore
            MODELS.register_module(module=CrossEntropyLoss, force=True)
            MMENGINE_MODELS.register_module(module=CrossEntropyLoss, force=True)
            print("✅ CrossEntropyLoss imported and registered")
        except ImportError:
            print("❌ Could not import CrossEntropyLoss from any location")
    
    # 确保SegVisualizationHook已注册
    try:
        from mmengine.registry import HOOKS  # type: ignore
        from mmseg.engine.hooks import SegVisualizationHook  # type: ignore
        if 'SegVisualizationHook' not in HOOKS.module_dict:
            HOOKS.register_module(module=SegVisualizationHook, force=True)
            print("✅ SegVisualizationHook registered to hooks registry")
    except ImportError as e:
        print(f"⚠️ Failed to import SegVisualizationHook: {e}")
        # 尝试从其他位置导入
        try:
            from mmengine.registry import HOOKS  # type: ignore
            from mmseg.engine import SegVisualizationHook  # type: ignore
            HOOKS.register_module(module=SegVisualizationHook, force=True)
            print("✅ SegVisualizationHook imported and registered")
        except ImportError:
            print("❌ Could not import SegVisualizationHook from any location")
    
    # 验证注册状态
    print(f"🔍 Final check - EncoderDecoder in MODELS: {'EncoderDecoder' in MODELS.module_dict}")
    print(f"🔍 Final check - EncoderDecoder in MMENGINE_MODELS: {'EncoderDecoder' in MMENGINE_MODELS.module_dict}")
    print(f"🔍 Final check - SegDataPreProcessor in MODELS: {'SegDataPreProcessor' in MODELS.module_dict}")
    print(f"🔍 Final check - SegDataPreProcessor in MMENGINE_MODELS: {'SegDataPreProcessor' in MMENGINE_MODELS.module_dict}")
    print(f"🔍 Final check - MixVisionTransformer in MODELS: {'MixVisionTransformer' in MODELS.module_dict}")
    print(f"🔍 Final check - MixVisionTransformer in MMENGINE_MODELS: {'MixVisionTransformer' in MMENGINE_MODELS.module_dict}")
    print(f"🔍 Final check - SegformerHead in MODELS: {'SegformerHead' in MODELS.module_dict}")
    print(f"🔍 Final check - SegformerHead in MMENGINE_MODELS: {'SegformerHead' in MMENGINE_MODELS.module_dict}")
    print(f"🔍 Final check - CrossEntropyLoss in MODELS: {'CrossEntropyLoss' in MODELS.module_dict}")
    print(f"🔍 Final check - CrossEntropyLoss in MMENGINE_MODELS: {'CrossEntropyLoss' in MMENGINE_MODELS.module_dict}")
    
    # 检查SegVisualizationHook注册状态
    try:
        from mmengine.registry import HOOKS  # type: ignore
        print(f"🔍 Final check - SegVisualizationHook in HOOKS: {'SegVisualizationHook' in HOOKS.module_dict}")
    except ImportError:
        print("⚠️ Could not import HOOKS registry")
    
    # 确保CityscapesDataset已注册
    try:
        from mmengine.registry import DATASETS  # type: ignore
        from mmseg.datasets import CityscapesDataset  # type: ignore
        if 'CityscapesDataset' not in DATASETS.module_dict:
            DATASETS.register_module(module=CityscapesDataset, force=True)
            print("✅ CityscapesDataset registered to datasets registry")
    except ImportError as e:
        print(f"⚠️ Failed to import CityscapesDataset: {e}")
        # 尝试从其他位置导入
        try:
            from mmengine.registry import DATASETS  # type: ignore
            from mmseg.datasets.cityscapes import CityscapesDataset  # type: ignore
            DATASETS.register_module(module=CityscapesDataset, force=True)
            print("✅ CityscapesDataset imported and registered")
        except ImportError:
            print("❌ Could not import CityscapesDataset from any location")
    
    # 检查CityscapesDataset注册状态
    try:
        from mmengine.registry import DATASETS  # type: ignore
        print(f"🔍 Final check - CityscapesDataset in DATASETS: {'CityscapesDataset' in DATASETS.module_dict}")
    except ImportError:
        print("⚠️ Could not import DATASETS registry")
    
    # 确保数据变换组件已注册
    try:
        from mmengine.registry import TRANSFORMS  # type: ignore
        from mmseg.datasets.transforms import *  # type: ignore
        
        # 注册常用的数据变换
        transform_classes = [
            'RandomCrop', 'RandomFlip', 'PhotoMetricDistortion', 'Normalize',
            'Pad', 'DefaultFormatBundle', 'Collect', 'LoadImageFromFile',
            'LoadAnnotations', 'Resize', 'RandomResize', 'ResizeToMultiple',
            'RandomRotate', 'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray',
            'SegRescale', 'BioMedical3DRandomCrop', 'BioMedical3DPad',
            'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur', 'BioMedicalRandomGamma'
        ]
        
        for transform_name in transform_classes:
            try:
                if transform_name not in TRANSFORMS.module_dict:
                    # 尝试从全局命名空间获取类
                    if transform_name in globals():
                        transform_cls = globals()[transform_name]
                        TRANSFORMS.register_module(name=transform_name, module=transform_cls, force=True)
                        print(f"✅ {transform_name} registered to transforms registry")
            except Exception as e:
                print(f"⚠️ Failed to register {transform_name}: {e}")
                
    except ImportError as e:
        print(f"⚠️ Failed to import transforms: {e}")
        # 尝试单独导入关键变换
        try:
            from mmengine.registry import TRANSFORMS  # type: ignore
            from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations  # type: ignore
            from mmseg.datasets.transforms.transforms import RandomCrop, RandomFlip, Resize  # type: ignore
            
            key_transforms = {
                'LoadImageFromFile': LoadImageFromFile,
                'LoadAnnotations': LoadAnnotations,
                'RandomCrop': RandomCrop,
                'RandomFlip': RandomFlip,
                'Resize': Resize
            }
            
            for name, cls in key_transforms.items():
                if name not in TRANSFORMS.module_dict:
                    TRANSFORMS.register_module(name=name, module=cls, force=True)
                    print(f"✅ {name} registered to transforms registry")
                    
        except ImportError as e2:
            print(f"❌ Could not import key transforms: {e2}")
    
    # 检查关键变换注册状态
    try:
        from mmengine.registry import TRANSFORMS  # type: ignore
        key_transforms = ['RandomCrop', 'RandomFlip', 'LoadImageFromFile', 'LoadAnnotations', 'Resize']
        for transform_name in key_transforms:
            print(f"🔍 Final check - {transform_name} in TRANSFORMS: {transform_name in TRANSFORMS.module_dict}")
    except ImportError:
        print("⚠️ Could not import TRANSFORMS registry")
    
    # 确保可视化器已注册 - 同时注册到mmseg和mmengine注册表
    try:
        from mmseg.visualization import SegLocalVisualizer  # type: ignore
        # 注册到mmseg注册表
        if 'SegLocalVisualizer' not in VISUALIZERS.module_dict:
            VISUALIZERS.register_module(module=SegLocalVisualizer, force=True)
        # 注册到mmengine注册表
        if 'SegLocalVisualizer' not in MMENGINE_VISUALIZERS.module_dict:
            MMENGINE_VISUALIZERS.register_module(module=SegLocalVisualizer, force=True)
        print("✅ SegLocalVisualizer registered to both registries")
    except ImportError as e:
        print(f"⚠️ Failed to import SegLocalVisualizer: {e}")
        # 如果导入失败，尝试从mmengine导入基础可视化器
        from mmengine.visualization import Visualizer
        VISUALIZERS.register_module(name='SegLocalVisualizer', module=Visualizer, force=True)
        MMENGINE_VISUALIZERS.register_module(name='SegLocalVisualizer', module=Visualizer, force=True)
        print("✅ Fallback visualizer registered to both registries")

    # 从文件加载配置
    cfg = Config.fromfile(args.config)

    # 设置工作目录
    if 'work_dir' not in cfg:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # 从配置构建并启动Runner
    runner = Runner.from_cfg(cfg)

    # 开始训练
    runner.train()

if __name__ == '__main__':
    main()