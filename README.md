# MapSage V4 遥感分割模型验证应用

🛰️ 基于Streamlit的遥感影像分割模型效果验证工具

## 📊 项目概述

本应用用于在Mac Intel CPU环境下验证MapSage V4遥感分割模型的效果。模型在LoveDA数据集上训练，通过滑窗推理和TTA技术达到了**mIoU = 84.96**的优秀性能。

### 🎯 主要功能
- 🖼️ 支持多种格式的遥感影像上传
- 🔍 实时分割结果可视化
- 📈 详细的类别统计分析
- 💾 分割结果下载功能
- 🎨 直观的类别图例显示

### 🏷️ 支持的地物类别
1. **背景** - 白色
2. **建筑** - 红色
3. **道路** - 黄色
4. **水体** - 蓝色
5. **贫瘠土地** - 紫色
6. **森林** - 绿色
7. **农业** - 橙色

## 📁 项目结构

```
MapSage_V5/
├── app.py                              # Streamlit主应用
├── requirements.txt                     # Python依赖包
├── README.md                           # 使用说明
├── checkpoints/                        # 模型权重目录
│   └── best_mIoU_iter_6000.pth        # 您的模型权重文件
├── configs/                            # 配置文件目录
│   └── mapsage_v4_eval_config.py      # 模型配置文件
└── images/                             # 测试图片目录
    └── sample_image.png                # 示例测试图片
```

## 🚀 快速开始

### 步骤1: 环境准备

#### 1.1 创建虚拟环境（推荐）
```bash
# 进入项目目录
cd /Users/barryzhang/myDev3/MapSage_V5

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate
```

#### 1.2 安装依赖包
```bash
# 升级pip
pip install --upgrade pip

# 安装基础依赖
pip install -r requirements.txt

# 安装MMSegmentation相关包
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
```

### 步骤2: 准备模型文件

#### 2.1 模型权重文件
将您的模型权重文件重命名为 `best_mIoU_iter_6000.pth` 并放入 `checkpoints/` 目录：
```bash
# 示例：如果您的权重文件名为 your_model.pth
cp your_model.pth checkpoints/best_mIoU_iter_6000.pth
```

#### 2.2 配置文件
将您的模型配置文件重命名为 `mapsage_v4_eval_config.py` 并放入 `configs/` 目录：
```bash
# 示例：如果您的配置文件名为 your_config.py
cp your_config.py configs/mapsage_v4_eval_config.py
```

**重要**: 确保配置文件中包含滑窗推理设置：
```python
# 在配置文件中确保有类似设置
test_cfg = dict(
    mode='slide',
    crop_size=(1024, 1024),
    stride=(768, 768)
)
```

#### 2.3 测试图片（可选）
将一些遥感影像放入 `images/` 目录用于快速测试：
```bash
# 建议使用来自LoveDA验证集的图片
cp your_test_image.png images/
```

### 步骤3: 运行应用

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 启动Streamlit应用
streamlit run app.py
```

应用将在浏览器中自动打开，通常地址为: `http://localhost:8501`

## 🖥️ 使用指南

### 基本操作
1. **上传图片**: 点击"选择一张图片进行分割..."按钮上传遥感影像
2. **等待处理**: CPU推理需要1-3分钟，请耐心等待
3. **查看结果**: 分割结果将显示在右侧面板
4. **分析统计**: 查看各类别的像素统计信息
5. **下载结果**: 点击"下载分割结果"保存彩色分割图

### 支持的图片格式
- JPG / JPEG
- PNG
- TIF / TIFF

### 推荐图片尺寸
- **最佳**: 512x512 到 1024x1024 像素
- **最大**: 不超过 2048x2048 像素
- **注意**: 图片越大，处理时间越长

## ⚡ 性能说明

### 系统要求
- **操作系统**: macOS (Intel芯片)
- **内存**: 建议16GB以上
- **存储**: 至少2GB可用空间
- **Python**: 3.8 - 3.11

### 性能指标
- **推理时间**: 1-3分钟（取决于图片大小）
- **内存占用**: 2-4GB（加载模型后）
- **模型精度**: mIoU = 84.96
- **推理设备**: CPU only

### 性能优化建议
1. **图片预处理**: 将大图裁剪为1024x1024以下
2. **关闭其他应用**: 释放更多内存给推理过程
3. **使用SSD**: 确保项目在SSD上运行
4. **虚拟环境**: 使用独立的Python环境避免冲突

## 🔧 故障排除

### 常见问题

#### 1. MMSegmentation导入失败
```bash
# 重新安装MMSegmentation
pip uninstall mmsegmentation mmcv mmengine
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
```

#### 2. 模型加载失败
- 检查权重文件路径是否正确
- 确认配置文件与权重文件匹配
- 查看终端错误信息

#### 3. 推理速度过慢
- 减小输入图片尺寸
- 检查系统内存使用情况
- 确保没有其他占用CPU的程序

#### 4. 内存不足
```bash
# 检查内存使用
top -o MEM

# 如果内存不足，尝试减小batch size或图片尺寸
```

### 调试模式
如果遇到问题，可以在终端中查看详细日志：
```bash
# 启用详细日志
streamlit run app.py --logger.level=debug
```

## 📝 配置文件示例

如果您需要创建新的配置文件，可以参考以下模板：

```python
# configs/mapsage_v4_eval_config.py 示例
_base_ = [
    '_base_/models/segformer_mit-b2.py',
    '_base_/datasets/loveda.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_160k.py'
]

# 模型配置
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

# 数据集配置
dataset_type = 'LoveDADataset'
data_root = 'data/LoveDA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 其他配置...
```

## 🤝 技术支持

如果您在使用过程中遇到问题，请检查：

1. **环境配置**: 确保所有依赖包正确安装
2. **文件路径**: 检查模型权重和配置文件路径
3. **系统资源**: 确保有足够的内存和存储空间
4. **Python版本**: 建议使用Python 3.8-3.11

## 📄 许可证

本项目基于MapSage V4训练计划开发，用于学术研究和技术验证。

## 🔄 更新日志

- **v1.0.0** (2025-01): 初始版本，支持基本的分割功能
- 支持LoveDA数据集的7个地物类别
- 集成滑窗推理和结果可视化
- 添加类别统计和下载功能

---

**祝您使用愉快！** 🎉

如果这个应用帮助您验证了模型效果，欢迎分享您的使用体验和改进建议。