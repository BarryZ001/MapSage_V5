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
    
    # 验证注册状态
    print(f"🔍 Final check - EncoderDecoder in MODELS: {'EncoderDecoder' in MODELS.module_dict}")
    print(f"🔍 Final check - EncoderDecoder in MMENGINE_MODELS: {'EncoderDecoder' in MMENGINE_MODELS.module_dict}")
    print(f"🔍 Final check - SegDataPreProcessor in MODELS: {'SegDataPreProcessor' in MODELS.module_dict}")
    print(f"🔍 Final check - SegDataPreProcessor in MMENGINE_MODELS: {'SegDataPreProcessor' in MMENGINE_MODELS.module_dict}")
    print(f"🔍 Final check - MixVisionTransformer in MODELS: {'MixVisionTransformer' in MODELS.module_dict}")
    print(f"🔍 Final check - MixVisionTransformer in MMENGINE_MODELS: {'MixVisionTransformer' in MMENGINE_MODELS.module_dict}")
    
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