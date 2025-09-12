import mmcv
import numpy as np
from mmengine.registry import init_default_scope
from mmseg.datasets import build_dataset
from mmseg.datasets.transforms import PackSegInputs
from mmengine.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

# Your baseline training config file
config_file = './configs/train_segformer_b2_imagenet.py'

# Load the config
cfg = mmcv.Config.fromfile(config_file)

# Build the dataset from the config
train_dataset = build_dataset(cfg.train_dataloader.dataset)

# Get a single data sample from the dataset
# This will include the original image, mask, and all other info
data_sample = train_dataset[0]

# Create a visualizer
visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    alpha=0.6 # Transparency of the mask overlay
)

# Use the visualizer to draw the image and its mask
# This automatically handles the color palette
visualizer.add_datasample(
    'augmented_sample',
    data_sample['inputs'].permute(1, 2, 0).numpy(), # Image tensor
    data_sample['data_sample'], # Mask and other data
    out_file="augmentation_check.png" # Output filename
)

print("Verification image saved as 'augmentation_check.png'. Please inspect it.")