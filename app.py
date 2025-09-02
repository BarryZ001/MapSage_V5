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

# 尝试导入MMSegmentation
try:
    from mmseg.apis import init_segmentor, inference_segmentor  # type: ignore
    # register_all_modules函数在较新版本中可能不存在
    try:
        from mmseg.utils import register_all_modules
    except ImportError:
        register_all_modules = None
    MMSEG_AVAILABLE = True
    MMSEG_ERROR = None
    # 注册所有模块（如果可用）
    if register_all_modules is not None:
        register_all_modules()
except (ImportError, ModuleNotFoundError) as e:
    MMSEG_AVAILABLE = False
    MMSEG_ERROR = str(e)
    print(f"MMSegmentation导入失败: {e}")
    print("请正确安装MMSegmentation及其依赖")
    sys.exit(1)

# --- 配置 ---
# 请将这里的路径修改为你自己的文件路径
CONFIG_FILE = 'final_standalone_config.py'
CHECKPOINT_FILE = 'checkpoints/best_mIoU_iter_6000.pth'
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

CLASS_NAMES = ['背景', '建筑', '道路', '水体', '贫瘠土地', '森林', '农业']

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
        
        # 如果缺少CLASSES，添加默认的类别信息
        if 'CLASSES' not in checkpoint_data['meta']:
            checkpoint_data['meta']['CLASSES'] = ['背景', '建筑', '道路', '水体', '贫瘠土地', '森林', '农业']
            needs_temp_file = True
            
        # 如果缺少PALETTE，添加默认的调色板信息
        if 'PALETTE' not in checkpoint_data['meta']:
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
        # 从配置文件的test pipeline中获取均值和标准差
        normalize_cfg = None
        for transform in cfg.data.test.pipeline:
            if transform['type'] == 'Normalize':
                normalize_cfg = transform
                break
        
        if normalize_cfg is None:
            # 使用默认的ImageNet归一化参数
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        else:
            mean = np.array(normalize_cfg['mean'], dtype=np.float32)
            std = np.array(normalize_cfg['std'], dtype=np.float32)
        
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
        return result[0]

    except Exception as e:
        st.error(f"❌ 推理过程中出错: {str(e)}")
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

# --- Streamlit 页面布局 ---

st.set_page_config(layout="wide", page_title="MapSage V4 - 遥感影像分割")

st.title("🛰️ MapSage V4 模型效果验证")
st.markdown("上传一张遥感影像，查看mIoU为 **84.96** 的模型分割效果。")

# 检查文件是否存在
config_exists = os.path.exists(CONFIG_FILE)
checkpoint_exists = os.path.exists(CHECKPOINT_FILE)

if not config_exists or not checkpoint_exists:
    st.error("⚠️ 缺少必要文件:")
    if not config_exists:
        st.error(f"- 配置文件: {CONFIG_FILE}")
    if not checkpoint_exists:
        st.error(f"- 权重文件: {CHECKPOINT_FILE}")
    st.info("请按照README中的说明准备这些文件。")
else:
    st.success("✅ 模型文件检查通过")

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
            
            st.subheader("🤖 第三行：模型分割结果")
            
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
                    
                    # 显示类别统计
                    st.subheader("📈 类别统计")
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
                    
                    # 提供下载功能
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
                        label="下载分割结果",
                        data=img_buffer.getvalue(),
                        file_name=f"segmentation_result_{base_filename}.png",
                        mime="image/png"
                    )
                    
                    st.success("🎉 分割完成！")
            else:
                st.error("❌ 模型加载失败")
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