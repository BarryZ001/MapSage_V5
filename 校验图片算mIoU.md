# Cell 1: 智能代码同步 (Smart Code Sync)
import os

# 定义项目目录和你的GitHub仓库地址
PROJECT_DIR = "/kaggle/working/MapSage_V5"
# 请确保这里的URL是正确的
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



# Cell 2: 安装环境依赖 (仅在会话首次启动时运行一次)

# 1. 确保MMLab的安装器是最新版本
!pip install -U openmim

# 2. 安装mmcv (使用4核并行编译)
!MAX_JOBS=4 mim install "mmcv<2.2.0,>=2.0.0"

# 3. 安装mmsegmentation (版本需与mmcv兼容)
!mim install "mmsegmentation==1.2.2"

# 4. 安装其他必要的库
!pip install ftfy timm scikit-image



# Cell 3: 运行评估
# (此单元格内容无需修改)

!python scripts/validate.py \
    configs/final_standalone_config.py \
    /kaggle/input/mapsage-stage02-checkpoint-6000/best_mIoU_iter_6000.pth \
    --data-root /kaggle/input/loveda