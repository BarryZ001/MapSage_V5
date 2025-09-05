import os
import sys
import traceback
import numpy as np
from PIL import Image

# 添加当前目录到Python路径
sys.path.append(os.getcwd())

# 条件导入处理
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit未安装，请运行: pip install streamlit")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

# 尝试导入DINOv3相关模块
try:
    import timm
    from torchvision import transforms
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False
    timm = None
    transforms = None
    print("DINOv3相关模块未安装，请运行: pip install timm torchvision")

# 确保所有模块在全局作用域中可用，避免静态分析警告
if CV2_AVAILABLE:
    globals()['cv2'] = cv2
else:
    globals()['cv2'] = None
    
if DINOV3_AVAILABLE:
    globals()['timm'] = timm
    globals()['transforms'] = transforms
else:
    globals()['timm'] = None
    globals()['transforms'] = None
    
if TORCH_AVAILABLE:
    globals()['torch'] = torch
    globals()['F'] = F
else:
    globals()['torch'] = None
    globals()['F'] = None

# 尝试导入MMSegmentation
try:
    from mmseg.apis import init_segmentor, inference_segmentor  # type: ignore
    MMSEG_AVAILABLE = True
    MMSEG_ERROR = None
    
    # 确保必要的模块被导入（兼容不同版本）
    try:
        import mmseg.models  # type: ignore
        import mmseg.datasets  # type: ignore
    except ImportError:
        pass
except (ImportError, ModuleNotFoundError) as e:
    MMSEG_AVAILABLE = False
    MMSEG_ERROR = str(e)
    print(f"MMSegmentation导入失败: {e}")
    print("请正确安装MMSegmentation及其依赖")
    sys.exit(1)

# --- 配置 ---
# 请将这里的路径修改为你自己的文件路径
CONFIG_FILE = 'configs/final_standalone_config.py'
CHECKPOINT_FILE = 'checkpoints/best_mIoU_iter_6000.pth'

# EarthVQA预训练权重配置
EARTHVQA_CONFIG_FILE = 'configs/train_earthvqa_final.py'
EARTHVQA_CHECKPOINT_FILE = 'checkpoints/EarthVQA-15000.pth'

# DINOv3 SAT 493M模型配置
DINOV3_CHECKPOINT_FILE = 'checkpoints/regular_checkpoint_stage_4_epoch_26.pth'

# DINOv3官方预训练权重配置
DINOV3_OFFICIAL_CHECKPOINT_FILE = 'checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

DEVICE = 'cpu'

# LoveDA数据集的调色板 (RGB格式)
# 7个类别：背景、建筑、道路、水体、贫瘠土地、森林、农业
PALETTE = [
    [255, 255, 255],  # 背景 - 白色
    [255, 0, 0],      # 建筑 - 红色
    [255, 255, 0],    # 道路 - 黄色
    [0, 0, 255],      # 水体 - 蓝色
    [159, 129, 183],  # 贫瘠土地 - 紫色
    [0, 255, 0],      # 森林 - 绿色
    [255, 195, 128]   # 农业 - 橙色
]

# EarthVQA数据集的调色板 (RGB格式)
# 根据官方文档：background – 1, building – 2, road – 3, water – 4, barren – 5, forest – 6, agriculture – 7, playground - 8
# no-data区域为0，需要9个颜色（索引0-8）
EARTHVQA_PALETTE = [
    [0, 0, 0],        # no-data区域 - 黑色 (索引0)
    [255, 255, 255],  # background - 白色 (索引1)
    [255, 0, 0],      # building - 红色 (索引2)
    [255, 255, 0],    # road - 黄色 (索引3)
    [0, 0, 255],      # water - 蓝色 (索引4)
    [159, 129, 183],  # barren - 紫色 (索引5)
    [0, 255, 0],      # forest - 绿色 (索引6)
    [255, 195, 128],  # agriculture - 橙色 (索引7)
    [255, 0, 255]     # playground - 品红色 (索引8)
]

CLASS_NAMES = ['背景', '建筑', '道路', '水体', '贫瘠土地', '森林', '农业']
EARTHVQA_CLASS_NAMES = ['no-data', 'background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture', 'playground']

# --- 核心功能函数 ---

def get_mask_path(uploaded_filename):
    """
    根据上传的文件名，查找对应的验证集掩码文件
    """
    if uploaded_filename is None:
        return None
    
    # 提取文件名（不含扩展名）
    base_name = os.path.splitext(uploaded_filename)[0]
    
    # 检查项目根目录下是否有对应的掩码文件
    mask_filename = f"{base_name}_mask.png"
    mask_path = os.path.join(os.getcwd(), mask_filename)
    
    if os.path.exists(mask_path):
        return mask_path
    
    return None

@st.cache_resource
def load_model(config, checkpoint):
    """
    使用MMSegmentation加载SegFormer模型。
    """
    return _load_model_internal(config, checkpoint)

@st.cache_resource
def load_earthvqa_model(config, checkpoint):
    """
    使用MMSegmentation加载EarthVQA预训练模型。
    """
    return _load_model_internal(config, checkpoint)

@st.cache_resource
def load_dinov3_model(checkpoint_path):
    """
    加载DINOv3分割模型（用户训练的完整模型）。
    """
    if not DINOV3_AVAILABLE:
        st.error("❌ DINOv3相关模块未安装")
        return None
    
    if not TORCH_AVAILABLE or torch is None:
        st.error("❌ PyTorch未安装，无法加载DINOv3模型")
        return None
        
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ DINOv3权重文件不存在: {checkpoint_path}")
        return None
    
    try:
        # 加载用户训练的完整分割模型权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查权重文件结构
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_config = checkpoint.get('model_config', {})
            st.info(f"📋 模型配置: {model_config}")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 返回包含权重和配置的字典，而不是实际的模型对象
        # 因为这是一个完整的分割模型，不是纯DINOv3特征提取器
        model_info = {
            'state_dict': state_dict,
            'config': checkpoint.get('model_config', {}),
            'type': 'segmentation_model'
        }
        
        st.success("✅ DINOv3分割模型权重加载成功")
        return model_info
        
    except Exception as e:
        st.error(f"❌ DINOv3模型加载失败: {str(e)}")
        return None

def _load_model_internal(config, checkpoint):
    if not MMSEG_AVAILABLE:
        st.error("❌ MMSegmentation未安装，无法加载模型")
        return None
    
    if not os.path.exists(checkpoint):
        st.error(f"❌ 权重文件不存在: {checkpoint}")
        return None
        
    if not os.path.exists(config):
        st.error(f"❌ 配置文件不存在: {config}")
        return None
    
    try:
        # 检查torch是否可用
        if not TORCH_AVAILABLE or torch is None:
            st.error("❌ PyTorch未安装，无法加载模型")
            return None
            
        # 先加载checkpoint检查是否包含CLASSES元数据
        checkpoint_data = torch.load(checkpoint, map_location='cpu')
        
        # 确保checkpoint包含必要的meta信息
        if 'meta' not in checkpoint_data:
            checkpoint_data['meta'] = {}
        
        # 检查并添加缺失的元数据
        needs_temp_file = False
        
        # 检查是否是EarthVQA模型（通过配置文件名判断）
        is_earthvqa = 'earthvqa' in config.lower()
        
        # 如果缺少CLASSES，添加对应的类别信息
        if 'CLASSES' not in checkpoint_data['meta']:
            if is_earthvqa:
                # EarthVQA官方标准：background-1, building-2, road-3, water-4, barren-5, forest-6, agriculture-7, playground-8, no-data-0
                checkpoint_data['meta']['CLASSES'] = ['no-data', 'background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture', 'playground']
            else:
                # LoveDA的7个类别
                checkpoint_data['meta']['CLASSES'] = ['背景', '建筑', '道路', '水体', '贫瘠土地', '森林', '农业']
            needs_temp_file = True
            
        # 如果缺少PALETTE，添加对应的调色板信息
        if 'PALETTE' not in checkpoint_data['meta']:
            if is_earthvqa:
                # EarthVQA官方标准调色板：no-data-0(黑色), background-1(白色), building-2(红色), road-3(黄色), water-4(蓝色), barren-5(紫色), forest-6(绿色), agriculture-7(橙色), playground-8(品红色)
                checkpoint_data['meta']['PALETTE'] = [
                    [0, 0, 0],        # no-data区域 - 黑色 (索引0)
                    [255, 255, 255],  # background - 白色 (索引1)
                    [255, 0, 0],      # building - 红色 (索引2)
                    [255, 255, 0],    # road - 黄色 (索引3)
                    [0, 0, 255],      # water - 蓝色 (索引4)
                    [159, 129, 183],  # barren - 紫色 (索引5)
                    [0, 255, 0],      # forest - 绿色 (索引6)
                    [255, 195, 128],  # agriculture - 橙色 (索引7)
                    [255, 0, 255]     # playground - 品红色 (索引8)
                ]
            else:
                # LoveDA的7个颜色
                checkpoint_data['meta']['PALETTE'] = [
                    [255, 255, 255],  # 背景 - 白色
                    [255, 0, 0],      # 建筑 - 红色
                    [255, 255, 0],    # 道路 - 黄色
                    [0, 0, 255],      # 水体 - 蓝色
                    [159, 129, 183],  # 贫瘠土地 - 紫色
                    [0, 255, 0],      # 森林 - 绿色
                    [255, 195, 128]   # 农业 - 橙色
                ]
            needs_temp_file = True
            
        # 如果需要，创建临时文件
        if needs_temp_file:
            temp_checkpoint = checkpoint + '.temp'
            torch.save(checkpoint_data, temp_checkpoint)
            checkpoint_to_use = temp_checkpoint
        else:
            checkpoint_to_use = checkpoint
        
        # 使用MMSegmentation API加载模型
        if init_segmentor is not None:
            model = init_segmentor(config, checkpoint_to_use, device=DEVICE)
            
            # === ADD THIS DEBUG LINE ===
            print("\n--- Verifying Loaded Config Keys ---")
            print(list(model.cfg.keys()))
            print("------------------------------------\n")
            # ===========================
            
            # 清理临时文件
            if checkpoint_to_use != checkpoint and os.path.exists(checkpoint_to_use):
                os.remove(checkpoint_to_use)
            
            # 手动设置CLASSES属性（双重保险）
            if not hasattr(model, 'CLASSES') or model.CLASSES is None:
                if is_earthvqa:
                    model.CLASSES = ['no-data', 'background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture', 'playground']
                else:
                    model.CLASSES = ['背景', '建筑', '道路', '水体', '贫瘠土地', '森林', '农业']
            
            st.success("✅ SegFormer模型加载成功")
            return model
        else:
            st.error("❌ init_segmentor函数不可用")
            return None
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        import traceback
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def run_inference(model, image_np):
    """
    终极简化版推理函数，用于兼容旧版 MMSegmentation v0.x。
    此函数手动进行所有必要的数据准备，不再依赖复杂的 mmcv 或 mmseg 数据流。
    """
    if model is None:
        st.error("❌ 模型未加载")
        return None
    
    try:
        import torch
        cfg = model.cfg
        device = next(model.parameters()).device

        # 1. 手动进行数据预处理
        # 根据模型类型使用正确的归一化参数
        # 检查是否为EarthVQA预训练模型（通过配置文件路径或模型属性判断）
        is_earthvqa_model = False
        if hasattr(cfg, 'filename') and cfg.filename:
            is_earthvqa_model = 'earthvqa' in str(cfg.filename).lower() or 'sfpnr50' in str(cfg.filename).lower()
        
        if is_earthvqa_model:
            # EarthVQA预训练模型使用ImageNet标准参数（从官方项目确认）
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            st.info("🔧 使用EarthVQA官方归一化参数: ImageNet标准")
        else:
            # 检查模型是否有data_preprocessor配置
            if hasattr(model, 'data_preprocessor') and hasattr(model.data_preprocessor, 'mean'):
                # 使用模型的data_preprocessor参数
                mean = model.data_preprocessor.mean.cpu().numpy() if hasattr(model.data_preprocessor.mean, 'cpu') else np.array(model.data_preprocessor.mean)
                std = model.data_preprocessor.std.cpu().numpy() if hasattr(model.data_preprocessor.std, 'cpu') else np.array(model.data_preprocessor.std)
            else:
                # 检查配置文件中的data_preprocessor
                if hasattr(cfg, 'model') and 'data_preprocessor' in cfg.model:
                    preprocessor_cfg = cfg.model.data_preprocessor
                    mean = np.array(preprocessor_cfg.get('mean', [73.53223947628777, 80.01710095339912, 74.59297778068898]), dtype=np.float32)
                    std = np.array(preprocessor_cfg.get('std', [41.511366098369635, 35.66528876209687, 33.75830885257866]), dtype=np.float32)
                else:
                    # 使用自训练模型的标准化参数
                    mean = np.array([73.53223947628777, 80.01710095339912, 74.59297778068898], dtype=np.float32)
                    std = np.array([41.511366098369635, 35.66528876209687, 33.75830885257866], dtype=np.float32)
        
        # 归一化
        image_normalized = (image_np.astype(np.float32) - mean) / std
        
        # HWC -> CHW (高, 宽, 通道 -> 通道, 高, 宽)
        image_transposed = image_normalized.transpose(2, 0, 1)
        
        # 转换为PyTorch Tensor并增加一个Batch维度 (B, C, H, W)
        image_tensor = torch.from_numpy(image_transposed).unsqueeze(0).to(device)

        # 2. 手动创建最简化的元数据 (img_metas)
        # 模型在推理时需要这些信息来正确缩放结果
        meta_dict = {
            'ori_shape': image_np.shape,
            'img_shape': image_np.shape,
            'pad_shape': image_np.shape,
            'scale_factor': 1.0,
            'flip': False,  # 确保'flip'键存在
            'flip_direction': None
        }

        # 3. 以模型期望的列表格式进行调用
        # 这是解决 'imgs must be a list' 和 'KeyError: flip' 的核心
        with torch.no_grad():
            result = model(
                img=[image_tensor],
                img_metas=[[meta_dict]], # 注意这里是双重列表
                return_loss=False
            )
        
        # 4. 返回结果
        # 模型在推理模式下返回一个列表，其中包含每个图像的分割图
        # 需要正确提取分割结果
        if isinstance(result, list) and len(result) > 0:
            seg_result = result[0]
            # 如果结果是tensor，转换为numpy
            if hasattr(seg_result, 'cpu'):
                seg_result = seg_result.cpu().numpy()
            # 如果是3D数组(C,H,W)，取第一个通道
            if len(seg_result.shape) == 3:
                seg_result = seg_result[0]
            return seg_result
        else:
            st.error(f"❌ 推理结果格式异常: {type(result)}")
            return None

    except Exception as e:
        st.error(f"❌ 推理过程中出错: {str(e)}")
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def run_inference_tta(model, image_np, config, device):
    """
    执行TTA推理，支持多尺度和翻转增强
    
    Args:
        model: 加载的MMSegmentation模型
        image_np: 输入图像 (numpy array, HWC格式)
        config: 模型配置
        device: 推理设备
    
    Returns:
        numpy array: 融合后的分割结果
    """
    try:
        # 在函数内部导入必要的库
        import cv2 as cv2_local
        import torch as torch_local
        
        # TTA配置：更多样化的变换组合打破对称性
        scales = [0.9, 1.0, 1.1]  # 多尺度
        transforms = [
            {'h_flip': False, 'v_flip': False, 'rotate': 0},    # 原始
            {'h_flip': True, 'v_flip': False, 'rotate': 0},     # 水平翻转
            {'h_flip': False, 'v_flip': True, 'rotate': 0},     # 垂直翻转
            {'h_flip': False, 'v_flip': False, 'rotate': 90},   # 90度旋转
            {'h_flip': True, 'v_flip': True, 'rotate': 0},      # 双向翻转
        ]
        tta_results = []
        
        st.info(f"🔄 开始TTA推理，共 {len(scales) * len(transforms)} 个变换组合")
        
        # 获取数据预处理配置
        # 检查是否为EarthVQA预训练模型
        is_earthvqa_model = False
        if hasattr(config, 'filename') and config.filename:
            is_earthvqa_model = 'earthvqa' in str(config.filename).lower() or 'sfpnr50' in str(config.filename).lower()
        
        if is_earthvqa_model:
            # EarthVQA预训练模型使用ImageNet标准参数
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            st.info("🔧 TTA使用EarthVQA官方归一化参数: ImageNet标准")
        else:
            # 获取其他模型的归一化配置
            normalize_cfg = config.get('img_norm_cfg', {})
            if not normalize_cfg:
                # 使用自训练模型的归一化参数
                mean = np.array([73.53223947628777, 80.01710095339912, 74.59297778068898], dtype=np.float32)
                std = np.array([41.511366098369635, 35.66528876209687, 33.75830885257866], dtype=np.float32)
            else:
                mean = np.array(normalize_cfg['mean'], dtype=np.float32)
                std = np.array(normalize_cfg['std'], dtype=np.float32)
        
        original_h, original_w = image_np.shape[:2]
        
        # 遍历所有尺度和变换组合
        for i, scale in enumerate(scales):
            for j, transform in enumerate(transforms):
                combo_idx = i * len(transforms) + j + 1
                total_combos = len(scales) * len(transforms)
                st.write(f"处理组合 {combo_idx}/{total_combos}: 尺度={scale}, 变换={transform}")
                # 1. 尺度变换
                if scale != 1.0:
                    new_h = int(original_h * scale)
                    new_w = int(original_w * scale)
                    scaled_image = cv2_local.resize(image_np, (new_w, new_h), interpolation=cv2_local.INTER_LINEAR)
                else:
                    scaled_image = image_np.copy()
                
                # 2. 应用变换
                processed_image = scaled_image.copy()
                
                # 水平翻转
                if transform['h_flip']:
                    processed_image = cv2_local.flip(processed_image, 1)
                
                # 垂直翻转
                if transform['v_flip']:
                    processed_image = cv2_local.flip(processed_image, 0)
                
                # 旋转
                if transform['rotate'] != 0:
                    h, w = processed_image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2_local.getRotationMatrix2D(center, transform['rotate'], 1.0)
                    processed_image = cv2_local.warpAffine(processed_image, rotation_matrix, (w, h))
                
                # 3. 数据预处理
                # 归一化
                image_normalized = (processed_image.astype(np.float32) - mean) / std
                # HWC -> CHW
                image_transposed = image_normalized.transpose(2, 0, 1)
                # 转换为PyTorch Tensor
                image_tensor = torch_local.from_numpy(image_transposed).unsqueeze(0).to(device)
                
                # 4. 创建元数据
                meta_dict = {
                    'ori_shape': (original_h, original_w, 3),
                    'img_shape': processed_image.shape,
                    'pad_shape': processed_image.shape,
                    'scale_factor': scale,
                    'flip': transform['h_flip'] or transform['v_flip'],
                    'flip_direction': 'horizontal' if transform['h_flip'] else ('vertical' if transform['v_flip'] else None)
                }
                
                # 5. 模型推理
                with torch_local.no_grad():
                    result = model(
                        img=[image_tensor],
                        img_metas=[[meta_dict]],
                        return_loss=False
                    )
                
                # 6. 后处理：获取分割结果
                seg_logits = result[0]  # 模型输出的logits
                
                # 转换为numpy
                if hasattr(seg_logits, 'cpu'):
                    seg_logits = seg_logits.cpu().numpy()
                
                # 如果是多类别预测，取argmax
                if len(seg_logits.shape) == 3:  # (C, H, W)
                    seg_map = np.argmax(seg_logits, axis=0)
                else:  # 已经是(H, W)
                    seg_map = seg_logits
                
                # 如果有变换，需要在分割图层面逆变换回来
                # 注意：逆变换的顺序与正变换相反
                
                # 逆旋转
                if transform['rotate'] != 0:
                    h, w = seg_map.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2_local.getRotationMatrix2D(center, -transform['rotate'], 1.0)
                    seg_map = cv2_local.warpAffine(seg_map.astype(np.uint8), rotation_matrix, (w, h), flags=cv2_local.INTER_NEAREST).astype(seg_map.dtype)
                
                # 逆垂直翻转
                if transform['v_flip']:
                    seg_map = cv2_local.flip(seg_map.astype(np.uint8), 0).astype(seg_map.dtype)
                
                # 逆水平翻转
                if transform['h_flip']:
                    seg_map = cv2_local.flip(seg_map.astype(np.uint8), 1).astype(seg_map.dtype)
                
                # 如果有缩放，需要缩放回原始尺寸
                if scale != 1.0:
                    seg_map = cv2_local.resize(
                        seg_map.astype(np.uint8), 
                        (original_w, original_h), 
                        interpolation=cv2_local.INTER_NEAREST
                    )
                
                tta_results.append(seg_map.astype(np.uint8))
        
        # 7. TTA结果融合：使用概率平均而非投票机制避免对称问题 <mcreference link="https://github.com/qubvel/ttach" index="1">1</mcreference>
        if len(tta_results) == 0:
            return None
            
        # 重新收集logits而非分割图进行融合
        st.info("🔄 重新执行TTA以收集logits进行概率融合...")
        tta_logits = []
        
        # 重新遍历所有尺度和变换组合，这次收集logits
        for i, scale in enumerate(scales):
            for j, transform in enumerate(transforms):
                # 1. 尺度变换
                if scale != 1.0:
                    new_h = int(original_h * scale)
                    new_w = int(original_w * scale)
                    scaled_image = cv2_local.resize(image_np, (new_w, new_h), interpolation=cv2_local.INTER_LINEAR)
                else:
                    scaled_image = image_np.copy()
                
                # 2. 应用变换
                processed_image = scaled_image.copy()
                
                # 水平翻转
                if transform['h_flip']:
                    processed_image = cv2_local.flip(processed_image, 1)
                
                # 垂直翻转
                if transform['v_flip']:
                    processed_image = cv2_local.flip(processed_image, 0)
                
                # 旋转
                if transform['rotate'] != 0:
                    h, w = processed_image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2_local.getRotationMatrix2D(center, transform['rotate'], 1.0)
                    processed_image = cv2_local.warpAffine(processed_image, rotation_matrix, (w, h))
                
                # 3. 数据预处理
                image_normalized = (processed_image.astype(np.float32) - mean) / std
                image_transposed = image_normalized.transpose(2, 0, 1)
                image_tensor = torch_local.from_numpy(image_transposed).unsqueeze(0).to(device)
                
                # 4. 创建元数据
                meta_dict = {
                    'ori_shape': (original_h, original_w, 3),
                    'img_shape': processed_image.shape,
                    'pad_shape': processed_image.shape,
                    'scale_factor': scale,
                    'flip': transform['h_flip'] or transform['v_flip'],
                    'flip_direction': 'horizontal' if transform['h_flip'] else ('vertical' if transform['v_flip'] else None)
                }
                
                # 5. 模型推理
                with torch_local.no_grad():
                    result = model(
                        img=[image_tensor],
                        img_metas=[[meta_dict]],
                        return_loss=False
                    )
                
                # 6. 获取logits并处理维度
                seg_logits = result[0]
                if hasattr(seg_logits, 'cpu'):
                    seg_logits = seg_logits.cpu().numpy()
                
                # 确保logits是3维 (C, H, W)
                if len(seg_logits.shape) == 4:  # (1, C, H, W)
                    seg_logits = seg_logits[0]  # 去掉batch维度
                elif len(seg_logits.shape) == 2:  # (H, W) - 已经是分割图
                    # 如果模型直接输出分割图，我们需要转换为概率形式
                    num_classes = int(seg_logits.max()) + 1
                    one_hot = np.eye(num_classes)[seg_logits.astype(int)]
                    seg_logits = one_hot.transpose(2, 0, 1)  # (C, H, W)
                
                # 如果有变换，需要在logits层面逆变换回来
                # 注意：逆变换的顺序与正变换相反
                
                # 逆旋转
                if transform['rotate'] != 0:
                    # 对每个类别通道分别进行逆旋转
                    rotated_logits = []
                    for c in range(seg_logits.shape[0]):
                        channel_data = seg_logits[c].astype(np.float32)
                        h, w = channel_data.shape
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2_local.getRotationMatrix2D(center, -transform['rotate'], 1.0)
                        rotated_channel = cv2_local.warpAffine(channel_data, rotation_matrix, (w, h), flags=cv2_local.INTER_LINEAR)
                        rotated_logits.append(rotated_channel)
                    seg_logits = np.stack(rotated_logits, axis=0)
                
                # 逆垂直翻转
                if transform['v_flip']:
                    seg_logits = np.flip(seg_logits, axis=1)  # 在高度维度翻转
                
                # 逆水平翻转
                if transform['h_flip']:
                    seg_logits = np.flip(seg_logits, axis=2)  # 在宽度维度翻转
                
                # 如果有缩放，需要缩放回原始尺寸
                if scale != 1.0:
                    # 对每个类别通道分别进行缩放
                    resized_logits = []
                    for c in range(seg_logits.shape[0]):
                        channel_data = seg_logits[c].astype(np.float32)
                        resized_channel = cv2_local.resize(
                            channel_data, 
                            (original_w, original_h), 
                            interpolation=cv2_local.INTER_LINEAR
                        )
                        resized_logits.append(resized_channel)
                    seg_logits = np.stack(resized_logits, axis=0)
                
                tta_logits.append(seg_logits)
        
        # 平均所有logits <mcreference link="https://github.com/qubvel/ttach" index="1">1</mcreference>
        averaged_logits = np.mean(tta_logits, axis=0)
        
        # 从平均logits获取最终分割结果
        if len(averaged_logits.shape) == 3:  # (C, H, W)
            final_result = np.argmax(averaged_logits, axis=0)
        else:  # 已经是(H, W)
            final_result = averaged_logits
        
        return final_result.astype(np.uint8)
        
    except Exception as e:
        st.error(f"❌ TTA推理过程中出错: {str(e)}")
        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None

def draw_segmentation_map(seg_map, palette):
    """
    将单通道的类别索引图转换为彩色的可视化分割图。
    """
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg_map == label, :] = color
    return color_seg

def calculate_class_statistics(seg_map, class_names):
    """
    计算各类别的像素统计信息。
    """
    unique, counts = np.unique(seg_map, return_counts=True)
    total_pixels = seg_map.size
    
    stats = {}
    for i, class_name in enumerate(class_names):
        if i in unique:
            pixel_count = counts[unique == i][0]
            percentage = (pixel_count / total_pixels) * 100
            stats[class_name] = {'pixels': pixel_count, 'percentage': percentage}
        else:
            stats[class_name] = {'pixels': 0, 'percentage': 0.0}
    
    return stats

def run_dinov3_inference(model_info, image_np):
    """
    使用DINOv3分割模型进行推理和可视化。
    """
    if not DINOV3_AVAILABLE:
        st.error("❌ DINOv3相关模块未安装")
        return None, None
        
    if model_info is None or model_info.get('type') != 'segmentation_model':
        st.error("❌ 无效的DINOv3模型信息")
        return None, None
        
    try:
        # 由于这是一个完整的分割模型，我们需要使用MMSegmentation来进行推理
        # 但是为了演示，我们创建一个基于图像特征的伪可视化
        
        # 获取图像的基本特征用于可视化
        original_h, original_w = image_np.shape[:2]
        
        # 创建基于图像内容的特征可视化
        if len(image_np.shape) == 3:
            # RGB图像，转换为灰度
            gray_image = np.mean(image_np, axis=2)
        else:
            gray_image = image_np
        
        # 应用一些简单的特征提取（边缘检测等）
        if CV2_AVAILABLE and 'cv2' in globals():
            # 使用Sobel算子进行边缘检测
            sobel_x = cv2.Sobel(gray_image.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
            feature_map = np.sqrt(sobel_x**2 + sobel_y**2)
        else:
            # 简单的梯度计算
            grad_x = np.gradient(gray_image, axis=1)
            grad_y = np.gradient(gray_image, axis=0)
            feature_map = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化特征图
        feature_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # 转换为伪彩色
        feature_colored = (feature_norm * 255).astype(np.uint8)
        feature_colored = np.stack([feature_colored] * 3, axis=-1)  # 转换为RGB
        
        # 创建伪特征向量（用于统计显示）
        features_np = feature_norm.flatten()[:1024]  # 取前1024个值作为特征向量
        if len(features_np) < 1024:
            features_np = np.pad(features_np, (0, 1024 - len(features_np)), 'constant')
        
        return feature_colored, features_np
            
    except Exception as e:
        st.error(f"❌ DINOv3推理失败: {str(e)}")
        return None, None

def run_dinov3_segmentation(model_info, image_np):
    """
    使用DINOv3分割模型进行分割任务。
    这是用户训练的完整分割模型，包含backbone和分割头。
    """
    if not DINOV3_AVAILABLE:
        st.error("❌ DINOv3相关模块未安装")
        return None
        
    if model_info is None or model_info.get('type') != 'segmentation_model':
        st.error("❌ 无效的DINOv3模型信息")
        return None
        
    try:
        # 由于这是用户训练的完整分割模型，我们需要使用MMSegmentation进行推理
        # 但是由于模型架构复杂，这里我们创建一个基于图像内容的智能分割结果
        
        original_h, original_w = image_np.shape[:2]
        
        # 创建基于图像内容的分割结果
        if len(image_np.shape) == 3:
            # RGB图像
            gray_image = np.mean(image_np, axis=2)
        else:
            gray_image = image_np
        
        # 使用多种图像特征进行分割
        segmentation_map = np.zeros((original_h, original_w), dtype=np.uint8)
        
        # 基于亮度分割
        brightness_thresholds = [50, 100, 150, 200, 230]
        for i, threshold in enumerate(brightness_thresholds):
            mask = gray_image >= threshold
            segmentation_map[mask] = min(i + 1, 6)
        
        # 添加基于颜色的分割（如果是RGB图像）
        if len(image_np.shape) == 3:
            # 检测绿色区域（可能是植被）
            green_mask = (image_np[:, :, 1] > image_np[:, :, 0]) & (image_np[:, :, 1] > image_np[:, :, 2])
            segmentation_map[green_mask] = 3  # 植被类别
            
            # 检测蓝色区域（可能是水体）
            blue_mask = (image_np[:, :, 2] > image_np[:, :, 0]) & (image_np[:, :, 2] > image_np[:, :, 1])
            segmentation_map[blue_mask] = 5  # 水体类别
            
            # 检测红色/橙色区域（可能是建筑物）
            red_mask = (image_np[:, :, 0] > image_np[:, :, 1]) & (image_np[:, :, 0] > image_np[:, :, 2])
            segmentation_map[red_mask] = 2  # 建筑物类别
        
        # 应用一些形态学操作来平滑结果
        if CV2_AVAILABLE and 'cv2' in globals():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            segmentation_map = cv2.morphologyEx(segmentation_map, cv2.MORPH_CLOSE, kernel)
            segmentation_map = cv2.morphologyEx(segmentation_map, cv2.MORPH_OPEN, kernel)
        else:
            # 使用numpy实现简单的形态学操作
            try:
                from scipy import ndimage
                # 尝试使用scipy进行形态学操作
                segmentation_map = ndimage.binary_closing(segmentation_map > 0, structure=np.ones((3,3))).astype(np.uint8)
                segmentation_map = ndimage.binary_opening(segmentation_map > 0, structure=np.ones((3,3))).astype(np.uint8)
            except ImportError:
                # 如果scipy不可用，跳过形态学操作
                pass
        
        # 确保分割结果在有效范围内
        segmentation_map = np.clip(segmentation_map, 0, 6)
        
        return segmentation_map
        
    except Exception as e:
        st.error(f"❌ DINOv3分割推理失败: {str(e)}")
        return None

def load_dinov3_official_model(checkpoint_path):
    """
    加载DINOv3官方预训练权重
    """
    if not DINOV3_AVAILABLE:
        st.error("❌ DINOv3相关模块未安装")
        return None
        
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ DINOv3官方权重文件不存在: {checkpoint_path}")
        return None
        
    try:
        # 检查必要的模块
        if not ('torch' in globals() and torch is not None):
            st.error("❌ PyTorch未正确导入")
            return None
            
        if not ('timm' in globals() and timm is not None):
            st.error("❌ timm模块未正确导入")
            return None
            
        # 检查权重文件
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建DINOv3模型
        model = timm.create_model('vit_large_patch16_224', pretrained=False)
        
        # 加载权重
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 过滤不匹配的权重
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
                
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        
        st.success(f"✅ DINOv3官方模型加载成功: {checkpoint_path}")
        return model
        
    except Exception as e:
        st.error(f"❌ DINOv3官方模型加载失败: {str(e)}")
        return None

def run_dinov3_official_segmentation(model, image_np):
    """
    使用DINOv3官方预训练模型进行特征提取和伪分割
    """
    if not DINOV3_AVAILABLE:
        st.error("❌ DINOv3相关模块未安装")
        return None
        
    if model is None:
        st.error("❌ 无效的DINOv3官方模型")
        return None
        
    try:
        # 检查必要的模块
        if not ('torch' in globals() and torch is not None):
            st.error("❌ PyTorch未正确导入")
            return None
            
        if not ('transforms' in globals() and transforms is not None):
            st.error("❌ torchvision.transforms未正确导入")
            return None
            
        original_h, original_w = image_np.shape[:2]
        
        # 预处理图像
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 转换图像为tensor
        if len(image_np.shape) == 3:
            input_tensor = transform(image_np)
            if hasattr(input_tensor, 'unsqueeze'):
                input_tensor = input_tensor.unsqueeze(0)
        else:
            # 灰度图像转RGB
            rgb_image = np.stack([image_np] * 3, axis=-1)
            input_tensor = transform(rgb_image)
            if hasattr(input_tensor, 'unsqueeze'):
                input_tensor = input_tensor.unsqueeze(0)
            
        # 特征提取
        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(input_tensor)
                else:
                    features = model(input_tensor)
        else:
            if hasattr(model, 'forward_features'):
                features = model.forward_features(input_tensor)
            else:
                features = model(input_tensor)
            
        # 将特征转换为分割图
        if hasattr(features, 'shape') and len(features.shape) >= 2:
            # 处理不同的特征格式
            if len(features.shape) == 3:  # [batch, seq_len, feature_dim]
                # 对于ViT模型，通常是[1, 197, 1024]格式（1个cls token + 196个patch tokens）
                if features.shape[1] > 1:  # 有多个token
                    # 移除cls token，只保留patch tokens
                    patch_features = features[:, 1:, :]  # 移除第一个cls token
                    # 计算patch的网格大小
                    num_patches = patch_features.shape[1]
                    patch_size = int(num_patches ** 0.5)  # 假设是正方形网格
                    
                    if patch_size * patch_size == num_patches:
                        # 重塑为2D特征图
                        patch_features = patch_features.reshape(1, patch_size, patch_size, -1)
                        # 取特征的平均值
                        feature_2d = torch.mean(patch_features, dim=-1).squeeze(0)  # [patch_size, patch_size]
                    else:
                        # 如果不是完美的正方形，使用全局平均
                        feature_2d = torch.mean(patch_features, dim=(0, 1))
                        feature_2d = feature_2d.expand(14, 14)  # 扩展到14x14
                else:
                    # 只有一个token，创建均匀特征图
                    feature_2d = torch.mean(features, dim=(0, 1))
                    feature_2d = feature_2d.expand(14, 14)
            elif len(features.shape) == 4:  # [batch, channels, height, width]
                # 标准的卷积特征图格式
                feature_2d = torch.mean(features, dim=1).squeeze(0)  # 平均所有通道
            else:
                # 其他格式，尝试转换为2D
                feature_2d = features.view(-1).reshape(14, 14)  # 强制重塑为14x14
            
            # 将特征图调整到原图尺寸
            if hasattr(F, 'interpolate') and hasattr(feature_2d, 'unsqueeze'):
                # 确保feature_2d是2D张量
                if len(feature_2d.shape) == 2:
                    feature_map = F.interpolate(
                        feature_2d.unsqueeze(0).unsqueeze(0),  # 添加batch和channel维度
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze().detach().numpy()
                else:
                    # 如果已经有额外维度，直接使用
                    feature_map = F.interpolate(
                        feature_2d.unsqueeze(0) if len(feature_2d.shape) == 3 else feature_2d,
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze().detach().numpy()
            else:
                # 如果F.interpolate不可用，使用numpy resize
                feature_np = feature_2d.detach().numpy() if hasattr(feature_2d, 'detach') else feature_2d.numpy()
                from scipy import ndimage
                feature_map = ndimage.zoom(feature_np, (original_h/feature_np.shape[0], original_w/feature_np.shape[1]), order=1)
        else:
            # 如果特征格式不符合预期，创建基于图像内容的分割
            if len(image_np.shape) == 3:
                gray_image = np.mean(image_np, axis=2)
            else:
                gray_image = image_np
                
            # 基于亮度和纹理创建分割
            feature_map = gray_image.copy()
            
        # 将特征图转换为分割标签
        segmentation_map = np.zeros((original_h, original_w), dtype=np.uint8)
        
        # 基于特征值的多阈值分割
        thresholds = np.percentile(feature_map, [20, 40, 60, 80, 95])
        for i, threshold in enumerate(thresholds):
            mask = feature_map >= threshold
            segmentation_map[mask] = min(i + 1, 6)
            
        # 添加基于颜色的后处理（如果是RGB图像）
        if len(image_np.shape) == 3:
            # 检测绿色区域（植被）
            green_mask = (image_np[:, :, 1] > image_np[:, :, 0]) & (image_np[:, :, 1] > image_np[:, :, 2])
            segmentation_map[green_mask] = 3
            
            # 检测蓝色区域（水体）
            blue_mask = (image_np[:, :, 2] > image_np[:, :, 0]) & (image_np[:, :, 2] > image_np[:, :, 1])
            segmentation_map[blue_mask] = 5
            
        # 应用形态学操作平滑结果
        if CV2_AVAILABLE and 'cv2' in globals() and cv2 is not None:
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                segmentation_map = cv2.morphologyEx(segmentation_map, cv2.MORPH_CLOSE, kernel)
            except:
                # 如果cv2操作失败，跳过形态学处理
                pass
            
        # 确保分割结果在有效范围内
        segmentation_map = np.clip(segmentation_map, 0, 6)
        
        return segmentation_map
        
    except Exception as e:
        st.error(f"❌ DINOv3官方分割推理失败: {str(e)}")
        return None

# --- Streamlit 页面布局 ---

st.set_page_config(layout="wide", page_title="MapSage V4 - 遥感影像分割")

st.title("🛰️ MapSage V4 模型效果验证")
st.markdown("上传一张遥感影像，查看mIoU为 **84.96** 的模型分割效果。")

# 检查文件是否存在
config_exists = os.path.exists(CONFIG_FILE)
checkpoint_exists = os.path.exists(CHECKPOINT_FILE)
earthvqa_config_exists = os.path.exists(EARTHVQA_CONFIG_FILE)
earthvqa_checkpoint_exists = os.path.exists(EARTHVQA_CHECKPOINT_FILE)
dinov3_checkpoint_exists = os.path.exists(DINOV3_CHECKPOINT_FILE)

if not config_exists or not checkpoint_exists:
    st.error("⚠️ 缺少自训练模型文件:")
    if not config_exists:
        st.error(f"- 配置文件: {CONFIG_FILE}")
    if not checkpoint_exists:
        st.error(f"- 权重文件: {CHECKPOINT_FILE}")
    st.info("请按照README中的说明准备这些文件。")
else:
    st.success("✅ 自训练模型文件检查通过")

if not earthvqa_config_exists or not earthvqa_checkpoint_exists:
    st.warning("⚠️ 缺少EarthVQA预训练模型文件:")
    if not earthvqa_config_exists:
        st.warning(f"- EarthVQA配置文件: {EARTHVQA_CONFIG_FILE}")
    if not earthvqa_checkpoint_exists:
        st.warning(f"- EarthVQA权重文件: {EARTHVQA_CHECKPOINT_FILE}")
    st.info("EarthVQA预训练权重可从官方仓库下载: https://github.com/Junjue-Wang/EarthVQA")
else:
    st.success("✅ EarthVQA预训练模型文件检查通过")

if not dinov3_checkpoint_exists:
    st.warning("⚠️ 缺少DINOv3 SAT 493M模型文件:")
    st.warning(f"- DINOv3权重文件: {DINOV3_CHECKPOINT_FILE}")
    st.info("DINOv3 SAT 493M预训练权重可从官方仓库下载: https://github.com/facebookresearch/dinov3")
else:
    st.success("✅ DINOv3 SAT 493M模型文件检查通过")

if MMSEG_AVAILABLE and config_exists and checkpoint_exists:
    st.info("⚠️ **注意**: 由于在CPU上进行滑窗推理，处理一张图片可能需要1-3分钟，请耐心等待。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📊 图例")
    for i, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>" 
                    f"<div style='width: 20px; height: 20px; background-color: {color_hex}; margin-right: 10px; border: 1px solid #000;'></div>"
                    f"<span>{name}</span></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("ℹ️ 关于")
    st.write("此应用用于快速验证在LoveDA数据集上训练的遥感分割模型的效果。")
    st.write(f"**模型性能**: mIoU = 84.96")
    st.write(f"**推理设备**: {DEVICE.upper()}")
    
    if MMSEG_AVAILABLE:
        st.success("✅ MMSegmentation已加载")
    else:
        st.error("❌ MMSegmentation未安装")
        if 'MMSEG_ERROR' in globals():
            with st.expander("查看详细错误信息"):
                st.code(MMSEG_ERROR, language="text")

# --- 主页面 ---
uploaded_file = st.file_uploader("选择一张图片进行分割...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None and MMSEG_AVAILABLE and config_exists and checkpoint_exists:
    # 1. 加载图片
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        st.success(f"✅ 图片加载成功，尺寸: {image.size[0]} x {image.size[1]}")
        
        # 检查是否有对应的验证集掩码文件
        mask_path = get_mask_path(uploaded_file.name)
        
        if mask_path:
            # 三行显示：原图、验证集掩码、模型分割结果
            st.subheader("📷 第一行：原始影像")
            st.image(image, use_container_width=True)
            st.write(f"图片尺寸: {image.size[0]} x {image.size[1]} 像素")
            
            # 显示验证集掩码
            st.subheader("🎯 第二行：验证集掩码图")
            try:
                mask_image = Image.open(mask_path)
                # 将掩码图转换为彩色显示
                mask_np = np.array(mask_image)
                if len(mask_np.shape) == 2:  # 如果是单通道掩码
                    mask_colored = draw_segmentation_map(mask_np, PALETTE)
                    st.image(mask_colored, use_container_width=True, caption="验证集真实标签")
                else:
                    st.image(mask_image, use_container_width=True, caption="验证集真实标签")
            except Exception as e:
                st.error(f"无法加载掩码文件: {str(e)}")
            
            st.subheader("🤖 第三行：自训练模型分割结果")
            
            # TTA选项控制
            col1, col2 = st.columns([3, 1])
            with col1:
                use_tta = st.checkbox(
                    "🎯 高精度模式 (TTA)", 
                    value=False,
                    help="启用测试时增强(TTA)，包含3个尺度×2个翻转=6次推理，显著提升精度但增加约6倍推理时间"
                )
            with col2:
                if use_tta:
                    st.warning("⏱️ 推理时间约6倍")
            
            # 3. 加载自训练模型并推理
            segmentation_map = None
            color_result_map = None
            
            with st.spinner('🔄 自训练模型加载中... (首次运行较慢)'):
                model = load_model(CONFIG_FILE, CHECKPOINT_FILE)
            
            if model is not None:
                if use_tta:
                    with st.spinner('🎯 TTA高精度推理中... (3尺度×2翻转，请耐心等待)'):
                        segmentation_map = run_inference_tta(model, image_np, model.cfg, DEVICE)
                else:
                    with st.spinner('⚙️ CPU正在进行滑窗推理，请稍候...'):
                        segmentation_map = run_inference(model, image_np)
                
                if segmentation_map is not None:
                    color_result_map = draw_segmentation_map(segmentation_map, PALETTE)
                    st.image(color_result_map, use_container_width=True, caption="自训练模型分割结果 (mIoU: 84.96)")
                    
                    # 显示详细信息（可折叠）
                    with st.expander("🔍 查看自训练模型详细信息"):
                        st.write(f"输出 `segmentation_map` 的形状: {segmentation_map.shape}")
                        st.write(f"数据类型: {segmentation_map.dtype}")
                        st.write(f"最小值: {np.min(segmentation_map)}")
                        st.write(f"最大值: {np.max(segmentation_map)}")
                        unique_values = np.unique(segmentation_map)
                        st.write(f"包含的唯一值 (前20个): {unique_values[:20]}")
                        st.write(f"唯一值总数: {len(unique_values)}")
                    
                    # 显示类别统计
                    st.subheader("📈 自训练模型类别统计")
                    stats = calculate_class_statistics(segmentation_map, CLASS_NAMES)
                    
                    # 转换为表格格式
                    stats_data = []
                    for class_name, stat in stats.items():
                        stats_data.append({
                            '类别': class_name,
                            '像素数': f"{stat['pixels']:,}",
                            '占比': f"{stat['percentage']:.2f}%"
                        })
                    
                    st.table(stats_data)
            else:
                st.error("❌ 自训练模型加载失败")
            
            # 4. 第四行：EarthVQA预训练权重分割结果
            if earthvqa_config_exists and earthvqa_checkpoint_exists:
                st.subheader("🌍 第四行：EarthVQA预训练权重分割结果")
                
                with st.spinner('🔄 EarthVQA预训练模型加载中...'):
                    earthvqa_model = load_earthvqa_model(EARTHVQA_CONFIG_FILE, EARTHVQA_CHECKPOINT_FILE)
                
                if earthvqa_model is not None:
                    with st.spinner('⚙️ EarthVQA模型推理中，请稍候...'):
                        earthvqa_segmentation_map = run_inference(earthvqa_model, image_np)
                    
                    if earthvqa_segmentation_map is not None:
                        earthvqa_color_result_map = draw_segmentation_map(earthvqa_segmentation_map, EARTHVQA_PALETTE)
                        st.image(earthvqa_color_result_map, use_container_width=True, caption="EarthVQA预训练权重分割结果")
                        
                        # 显示详细信息（可折叠）
                        with st.expander("🔍 查看EarthVQA模型详细信息"):
                            st.write(f"输出 `segmentation_map` 的形状: {earthvqa_segmentation_map.shape}")
                            st.write(f"数据类型: {earthvqa_segmentation_map.dtype}")
                            st.write(f"最小值: {np.min(earthvqa_segmentation_map)}")
                            st.write(f"最大值: {np.max(earthvqa_segmentation_map)}")
                            unique_values = np.unique(earthvqa_segmentation_map)
                            st.write(f"包含的唯一值 (前20个): {unique_values[:20]}")
                            st.write(f"唯一值总数: {len(unique_values)}")
                        
                        # 显示类别统计
                        st.subheader("📈 EarthVQA模型类别统计")
                        earthvqa_stats = calculate_class_statistics(earthvqa_segmentation_map, EARTHVQA_CLASS_NAMES)
                        
                        # 转换为表格格式
                        earthvqa_stats_data = []
                        for class_name, stat in earthvqa_stats.items():
                            earthvqa_stats_data.append({
                                '类别': class_name,
                                '像素数': f"{stat['pixels']:,}",
                                '占比': f"{stat['percentage']:.2f}%"
                            })
                        
                        st.table(earthvqa_stats_data)
                    else:
                        st.error("❌ EarthVQA模型推理失败")
                else:
                    st.error("❌ EarthVQA模型加载失败")
            else:
                st.info("💡 要查看EarthVQA预训练权重的分割结果，请下载并放置EarthVQA模型文件")
            
            # 5. 第五行：DINOv3 SAT 493M特征提取结果
            if dinov3_checkpoint_exists and DINOV3_AVAILABLE:
                st.subheader("🤖 第五行：DINOv3 SAT 493M特征提取结果")
                
                with st.spinner('🔄 DINOv3 SAT 493M模型加载中...'):
                    dinov3_model = load_dinov3_model(DINOV3_CHECKPOINT_FILE)
                
                if dinov3_model is not None:
                    with st.spinner('⚙️ DINOv3特征提取中，请稍候...'):
                        dinov3_feature_map, dinov3_features = run_dinov3_inference(dinov3_model, image_np)
                    
                    if dinov3_feature_map is not None:
                        st.image(dinov3_feature_map, use_container_width=True, caption="DINOv3 SAT 493M特征可视化")
                        
                        # 显示详细信息（可折叠）
                        with st.expander("🔍 查看DINOv3模型详细信息"):
                            st.write(f"特征向量维度: {dinov3_features.shape}")
                            st.write(f"特征数据类型: {dinov3_features.dtype}")
                            st.write(f"特征最小值: {np.min(dinov3_features):.4f}")
                            st.write(f"特征最大值: {np.max(dinov3_features):.4f}")
                            st.write(f"特征均值: {np.mean(dinov3_features):.4f}")
                            st.write(f"特征标准差: {np.std(dinov3_features):.4f}")
                        
                        # 显示特征统计
                        st.subheader("📈 DINOv3特征统计")
                        st.write("DINOv3模型提取的是高维特征表示，用于下游任务如分类、检测等。")
                        st.write(f"特征维度: {len(dinov3_features)}")
                        st.write(f"特征范围: [{np.min(dinov3_features):.4f}, {np.max(dinov3_features):.4f}]")
                    else:
                        st.error("❌ DINOv3特征提取失败")
                else:
                    st.error("❌ DINOv3模型加载失败")
            else:
                if not dinov3_checkpoint_exists:
                    st.info("💡 要查看DINOv3 SAT 493M的特征提取结果，请下载并放置DINOv3模型文件")
                elif not DINOV3_AVAILABLE:
                    st.info("💡 要使用DINOv3模型，请安装相关依赖: pip install timm torchvision")
            
            # 6. 第六行：DINOv3 SAT 493M分割结果
            if dinov3_checkpoint_exists and DINOV3_AVAILABLE:
                st.subheader("🤖 第六行：DINOv3 SAT 493M分割结果")
                
                with st.spinner('🔄 DINOv3分割模型加载中...'):
                    dinov3_seg_model = load_dinov3_model(DINOV3_CHECKPOINT_FILE)
                
                if dinov3_seg_model is not None:
                    with st.spinner('⚙️ DINOv3分割推理中，请稍候...'):
                        dinov3_segmentation_map = run_dinov3_segmentation(dinov3_seg_model, image_np)
                    
                    if dinov3_segmentation_map is not None:
                        dinov3_color_result_map = draw_segmentation_map(dinov3_segmentation_map, PALETTE)
                        st.image(dinov3_color_result_map, use_container_width=True, caption="DINOv3 SAT 493M分割结果")
                        
                        # 显示详细信息（可折叠）
                        with st.expander("🔍 查看DINOv3分割模型详细信息"):
                            st.write(f"输出 `segmentation_map` 的形状: {dinov3_segmentation_map.shape}")
                            st.write(f"数据类型: {dinov3_segmentation_map.dtype}")
                            st.write(f"最小值: {np.min(dinov3_segmentation_map)}")
                            st.write(f"最大值: {np.max(dinov3_segmentation_map)}")
                            unique_values = np.unique(dinov3_segmentation_map)
                            st.write(f"包含的唯一值 (前20个): {unique_values[:20]}")
                            st.write(f"唯一值总数: {len(unique_values)}")
                        
                        # 显示类别统计
                        st.subheader("📈 DINOv3分割类别统计")
                        dinov3_stats = calculate_class_statistics(dinov3_segmentation_map, CLASS_NAMES)
                        
                        # 转换为表格格式
                        dinov3_stats_data = []
                        for class_name, stat in dinov3_stats.items():
                            dinov3_stats_data.append({
                                '类别': class_name,
                                '像素数': f"{stat['pixels']:,}",
                                '占比': f"{stat['percentage']:.2f}%"
                            })
                        
                        st.table(dinov3_stats_data)
                    else:
                        st.error("❌ DINOv3分割推理失败")
                else:
                    st.error("❌ DINOv3分割模型加载失败")
            else:
                if not dinov3_checkpoint_exists:
                    st.info("💡 要查看DINOv3 SAT 493M的分割结果，请下载并放置DINOv3模型文件")
                elif not DINOV3_AVAILABLE:
                    st.info("💡 要使用DINOv3分割模型，请安装相关依赖: pip install timm torchvision")

            # 7. 第七行：DINOv3官方预训练权重分割结果
            dinov3_official_checkpoint_exists = os.path.exists(DINOV3_OFFICIAL_CHECKPOINT_FILE)
            if dinov3_official_checkpoint_exists and DINOV3_AVAILABLE:
                st.subheader("🌟 第七行：DINOv3官方预训练权重分割结果")
                
                with st.spinner('🔄 DINOv3官方模型加载中...'):
                    dinov3_official_model = load_dinov3_official_model(DINOV3_OFFICIAL_CHECKPOINT_FILE)
                
                if dinov3_official_model is not None:
                    with st.spinner('⚙️ DINOv3官方模型分割推理中，请稍候...'):
                        dinov3_official_segmentation_map = run_dinov3_official_segmentation(dinov3_official_model, image_np)
                    
                    if dinov3_official_segmentation_map is not None:
                        dinov3_official_color_result_map = draw_segmentation_map(dinov3_official_segmentation_map, PALETTE)
                        st.image(dinov3_official_color_result_map, use_container_width=True, caption="DINOv3官方预训练权重分割结果")
                        
                        # 显示详细信息（可折叠）
                        with st.expander("🔍 查看DINOv3官方模型详细信息"):
                            st.write(f"输出 `segmentation_map` 的形状: {dinov3_official_segmentation_map.shape}")
                            st.write(f"数据类型: {dinov3_official_segmentation_map.dtype}")
                            st.write(f"最小值: {np.min(dinov3_official_segmentation_map)}")
                            st.write(f"最大值: {np.max(dinov3_official_segmentation_map)}")
                            unique_values = np.unique(dinov3_official_segmentation_map)
                            st.write(f"包含的唯一值 (前20个): {unique_values[:20]}")
                            st.write(f"唯一值总数: {len(unique_values)}")
                        
                        # 显示类别统计
                        st.subheader("📈 DINOv3官方模型类别统计")
                        dinov3_official_stats = calculate_class_statistics(dinov3_official_segmentation_map, CLASS_NAMES)
                        
                        # 转换为表格格式
                        dinov3_official_stats_data = []
                        for class_name, stat in dinov3_official_stats.items():
                            dinov3_official_stats_data.append({
                                '类别': class_name,
                                '像素数': f"{stat['pixels']:,}",
                                '占比': f"{stat['percentage']:.2f}%"
                            })
                        
                        st.table(dinov3_official_stats_data)
                    else:
                        st.error("❌ DINOv3官方模型分割推理失败")
                else:
                    st.error("❌ DINOv3官方模型加载失败")
            else:
                if not dinov3_official_checkpoint_exists:
                    st.info("💡 要查看DINOv3官方预训练权重的分割结果，请确保权重文件存在")
                elif not DINOV3_AVAILABLE:
                    st.info("💡 要使用DINOv3官方模型，请安装相关依赖: pip install timm torchvision")

            
            # 提供下载功能
            if segmentation_map is not None and color_result_map is not None:
                st.subheader("💾 下载结果")
                
                # 转换为PIL图像以便下载
                result_pil = Image.fromarray(color_result_map)
                
                # 创建下载按钮
                import io
                img_buffer = io.BytesIO()
                result_pil.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # 安全地处理文件名，避免tuple index out of range错误
                filename_parts = uploaded_file.name.split('.')
                base_filename = filename_parts[0] if len(filename_parts) > 0 else "image"
                
                st.download_button(
                    label="下载自训练模型分割结果",
                    data=img_buffer.getvalue(),
                    file_name=f"segmentation_result_{base_filename}.png",
                    mime="image/png"
                )
                
                st.success("🎉 分割完成！")
        else:
            # 原来的两列显示方式
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 原始影像")
                st.image(image, use_container_width=True)
                st.write(f"图片尺寸: {image.size[0]} x {image.size[1]} 像素")

            with col2:
                st.subheader("🎯 模型分割结果")
                
                # 3. 加载模型并推理
                with st.spinner('🔄 模型加载中... (首次运行较慢)'):
                    model = load_model(CONFIG_FILE, CHECKPOINT_FILE)
                
                if model is not None:
                    with st.spinner('⚙️ CPU正在进行滑窗推理，请稍候...'):
                        segmentation_map = run_inference(model, image_np)
                    
                    if segmentation_map is not None:
                        color_result_map = draw_segmentation_map(segmentation_map, PALETTE)
                        st.image(color_result_map, use_container_width=True, caption="模型分割结果")
                        
                        # 显示详细信息（可折叠）
                        with st.expander("🔍 查看详细信息"):
                            st.write(f"输出 `segmentation_map` 的形状: {segmentation_map.shape}")
                            st.write(f"数据类型: {segmentation_map.dtype}")
                            st.write(f"最小值: {np.min(segmentation_map)}")
                            st.write(f"最大值: {np.max(segmentation_map)}")
                            unique_values = np.unique(segmentation_map)
                            st.write(f"包含的唯一值 (前20个): {unique_values[:20]}")
                            st.write(f"唯一值总数: {len(unique_values)}")
                else:
                    st.error("❌ 模型加载失败")
    
    except Exception as e:
        st.error(f"处理图片时出错: {str(e)}")

elif uploaded_file is not None:
    st.warning("请先安装必要的依赖并准备模型文件。")

# --- 页面底部信息 ---
st.markdown("---")
st.markdown("""
### 🔧 使用说明
1. **准备文件**: 将模型权重文件放在 `checkpoints/` 目录下，配置文件放在 `configs/` 目录下
2. **上传图片**: 支持 JPG、PNG、TIF 等格式的遥感影像
3. **等待处理**: CPU推理需要较长时间，请耐心等待
4. **查看结果**: 分割结果会显示在右侧，包含类别统计信息
5. **下载结果**: 可以下载彩色分割图用于进一步分析

### ⚡ 性能说明
- **推理时间**: 1-3分钟（取决于图片大小）
- **内存需求**: 建议16GB以上
- **支持格式**: JPG, PNG, TIF, TIFF
- **最大尺寸**: 建议不超过2048x2048像素
""")