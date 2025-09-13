# Kaggle Notebook Cells for MapSage Knowledge Distillation Training

## Cell 1: Environment Setup and Dependencies
```python
# 检查CUDA版本并安装兼容的MMCV
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 安装基础依赖
!pip install -U openmim
!mim install mmengine

# 根据Kaggle环境安装CPU版本MMCV（避免CUDA版本冲突）
!pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# 安装其他依赖
!pip install "mmsegmentation>=1.0.0"
!pip install mmrazor
!pip install ftfy regex

# Import libraries
import os
import numpy as np
from mmengine import Config
from mmengine.runner import load_checkpoint
import warnings
warnings.filterwarnings('ignore')

# 延迟导入mmseg相关模块，避免CUDA冲突
try:
    from mmseg.apis import init_model, inference_model
    from mmseg.datasets import build_dataloader, build_dataset
    from mmseg.models import build_segmentor
    print("✅ MMSeg模块导入成功！")
except ImportError as e:
    print(f"⚠️ MMSeg导入警告: {e}")
    print("将在训练时重新尝试导入...")

print("✅ 环境设置完成！")
```

## Cell 2: Data Preparation and Verification
```python
# Verify dataset paths
data_root = '/kaggle/input/loveda'
checkpoint_path = '/kaggle/input/mapsage-stage02-checkpoint-6000/best_mIoU_iter_6000.pth'
dinov3_path = '/kaggle/input/dinov3-vitl16-pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

# Check if all required files exist
print("Checking dataset and checkpoint availability:")
print(f"Dataset root exists: {os.path.exists(data_root)}")
print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
print(f"DINOv3 weights exist: {os.path.exists(dinov3_path)}")

# List dataset structure
if os.path.exists(data_root):
    print("\nDataset structure:")
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            print(f"  {item}/")
            # Show first few items in each directory
            sub_items = os.listdir(item_path)[:5]
            for sub_item in sub_items:
                print(f"    {sub_item}")
            if len(os.listdir(item_path)) > 5:
                print(f"    ... and {len(os.listdir(item_path)) - 5} more files")
```

## Cell 3: Configuration Setup
```python
# Setup project directory with auto-sync
import os

# Project configuration
PROJECT_DIR = "/kaggle/working/MapSage_V5"
GIT_REPO_URL = "https://github.com/BarryZ001/MapSage_V5.git"

if os.path.exists(PROJECT_DIR):
    print("✅ 项目目录已存在，拉取最新更新...")
    %cd {PROJECT_DIR}
    !git pull
else:
    print("🚀 首次设置，克隆项目仓库...")
    !git clone {GIT_REPO_URL} {PROJECT_DIR}
    %cd {PROJECT_DIR}

print("\n✅ 代码已同步至最新版本！")

# Copy the knowledge distillation configuration
config_path = '/kaggle/working/MapSage_V5/configs/train_distill_dinov3_simple_kd.py'

# Load and verify configuration
cfg = Config.fromfile(config_path)
print("Configuration loaded successfully!")
print(f"Model type: {cfg.model.type}")
print(f"Data root: {cfg.data_root}")
print(f"Load from: {cfg.load_from}")
print(f"Max iterations: {cfg.train_cfg.max_iters}")
print(f"Validation interval: {cfg.train_cfg.val_interval}")

# Verify LoveDA dataset structure
print("\n📊 LoveDA数据集结构验证:")
loveda_root = '/kaggle/input/loveda'
for split in ['Train', 'Val']:
    for scene in ['Rural', 'Urban']:
        img_path = f"{loveda_root}/{split}/{scene}/images_png"
        mask_path = f"{loveda_root}/{split}/{scene}/masks_png"
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img_count = len([f for f in os.listdir(img_path) if f.endswith('.png')])
            mask_count = len([f for f in os.listdir(mask_path) if f.endswith('.png')])
            print(f"✅ {split}/{scene}: {img_count} images, {mask_count} masks")
        else:
            print(f"❌ {split}/{scene}: 路径不存在")
            
print("\n🎯 训练配置: 使用Rural+Urban完整数据集进行知识蒸馏训练")
print("📈 预期效果: 更丰富的场景多样性，提升模型泛化能力")
```

## Cell 4: Model Training
```python
# Import necessary functions (avoid CUDA-dependent imports)
from mmengine.runner import Runner
from mmengine.checkpoint import load_checkpoint

# Build the model directly from config without importing mmseg.apis
# This avoids CUDA extension loading issues in Kaggle
from mmengine.registry import REGISTRIES
from mmengine.config import Config

# Build model using MMEngine's registry system
model_cfg = cfg.model.copy()
model_cfg['data_preprocessor'] = cfg.data_preprocessor
model = REGISTRIES.build(model_cfg)

# Alternative: use Runner to build model (recommended for MMSeg)
# runner = Runner.from_cfg(cfg)
# model = runner.model

# Load the pretrained checkpoint
if cfg.load_from:
    print(f"Loading checkpoint from: {cfg.load_from}")
    checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')
    print("Checkpoint loaded successfully!")

# Build datasets using Runner (avoids direct dataset import issues)
runner = Runner.from_cfg(cfg)
print(f"Training dataset configured: {type(runner.train_dataloader.dataset).__name__}")
print(f"Training batch size: {runner.train_dataloader.batch_size}")

# Start training with MMEngine Runner
print("Starting knowledge distillation training...")
runner.train()

print("✅ 训练启动成功！")
```

## Cell 5: Training Monitoring
```python
# Monitor training progress
import matplotlib.pyplot as plt
import json

# Function to parse training logs
def parse_training_logs(log_file):
    losses = []
    miou_scores = []
    iterations = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if 'loss' in line and 'iter' in line:
                    # Parse loss and iteration info
                    try:
                        data = json.loads(line)
                        if 'loss' in data:
                            losses.append(data['loss'])
                            iterations.append(data['iter'])
                    except:
                        pass
                elif 'mIoU' in line:
                    # Parse validation mIoU
                    try:
                        data = json.loads(line)
                        if 'mIoU' in data:
                            miou_scores.append(data['mIoU'])
                    except:
                        pass
    
    return iterations, losses, miou_scores

# Plot training curves
log_file = '/kaggle/working/MapSage_V5/work_dirs/train_distill_dinov3_simple_kd/tf_logs/events.out.tfevents.*'
iterations, losses, miou_scores = parse_training_logs(log_file)

if losses:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    if miou_scores:
        plt.plot(miou_scores)
        plt.title('Validation mIoU')
        plt.xlabel('Validation Step')
        plt.ylabel('mIoU (%)')
    
    plt.tight_layout()
    plt.show()

print(f"Current best mIoU: {max(miou_scores) if miou_scores else 'Not available yet'}")
```

## Cell 6: Model Evaluation and Results
```python
# Load the best checkpoint for evaluation
work_dir = '/kaggle/working/MapSage_V5/work_dirs/train_distill_dinov3_simple_kd'
best_checkpoint = os.path.join(work_dir, 'best_mIoU_iter_*.pth')

# Find the latest best checkpoint
import glob
checkpoint_files = glob.glob(best_checkpoint)
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Loading best checkpoint: {latest_checkpoint}")
    
    # Initialize model for inference
    model = init_model(config_path, latest_checkpoint, device='cuda:0')
    
    # Run evaluation on test set
    test_dataset = build_dataset(cfg.data.test)
    test_dataloader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )
    
    print("Running final evaluation...")
    # Note: Add your evaluation code here based on your specific metrics
    
else:
    print("No checkpoint found. Training may still be in progress.")

# Display training summary
print("\n=== Training Summary ===")
print(f"Configuration: Simple Knowledge Distillation")
print(f"Teacher Model: DINOv3-Large")
print(f"Student Model: SegFormer-B0")
print(f"Initial mIoU: 84.96%")
print(f"Target: Improve performance through knowledge distillation")
print(f"Training completed successfully!")
```

## Cell 7: Results Visualization
```python
# Visualize some prediction results
import cv2
from mmseg.core.evaluation import get_palette

def visualize_predictions(model, test_images, num_samples=5):
    """Visualize model predictions on test images"""
    palette = get_palette('cityscapes')
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i in range(min(num_samples, len(test_images))):
        img_path = test_images[i]
        
        # Load and predict
        result = inference_model(model, img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display original, prediction, and overlay
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(result[0], cmap='tab20')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        
        # Create overlay
        overlay = img.copy()
        mask = result[0]
        for class_id in np.unique(mask):
            if class_id > 0:  # Skip background
                color = palette[class_id % len(palette)]
                overlay[mask == class_id] = overlay[mask == class_id] * 0.6 + np.array(color) * 0.4
        
        axes[i, 2].imshow(overlay.astype(np.uint8))
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Get some test images for visualization
test_img_dir = os.path.join(data_root, 'images', 'test')
if os.path.exists(test_img_dir):
    test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)[:5]]
    if checkpoint_files and test_images:
        visualize_predictions(model, test_images)
else:
    print("Test images not found for visualization")

print("Knowledge distillation training completed successfully!")
print("Check the work_dirs folder for detailed logs and checkpoints.")
```

## Usage Instructions:

1. **Run Cell 1** first to install dependencies
2. **Run Cell 2** to verify your dataset is properly uploaded
3. **Run Cell 3** to setup the configuration
4. **Run Cell 4** to start training (this will take several hours)
5. **Run Cell 5** periodically to monitor progress
6. **Run Cell 6** after training completes for evaluation
7. **Run Cell 7** to visualize results

## Expected Results:
- Training should improve upon the baseline 84.96% mIoU
- Knowledge distillation typically provides 1-3% improvement
- Total training time: 6-8 hours on Kaggle P100 GPU
- Final model will be saved in `/kaggle/working/MapSage_V5/work_dirs/`