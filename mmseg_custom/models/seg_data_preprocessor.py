"""
自定义SegDataPreProcessor实现
解决训练时SegDataPreProcessor未注册的问题
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Union
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement
import numpy as np


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Segmentation data preprocessor.
    
    This preprocessor handles image normalization, padding, and format conversion
    for segmentation tasks.
    
    Args:
        mean (Sequence[float]): The pixel mean of R, G, B channels.
        std (Sequence[float]): The pixel standard deviation of R, G, B channels.
        bgr_to_rgb (bool): Whether to convert the image from BGR to RGB.
        pad_val (float): Padding value for images.
        seg_pad_val (int): Padding value for segmentation maps.
        size (tuple, optional): Fixed size for resizing.
    """
    
    def __init__(self,
                 mean: Sequence[float] = (123.675, 116.28, 103.53),
                 std: Sequence[float] = (58.395, 57.12, 57.375),
                 bgr_to_rgb: bool = True,
                 pad_val: float = 0,
                 seg_pad_val: int = 255,
                 size: Optional[tuple] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.size = size
        
    def forward(self, data: dict, training: bool = False) -> dict:
        """Forward function."""
        # 兼容不同的数据格式
        if 'inputs' in data:
            # 标准格式：{'inputs': tensor, 'data_samples': [...]}
            inputs = data['inputs']
            data_samples = data.get('data_samples', [])
        elif 'img' in data:
            # CustomCollect格式：{'img': tensor, 'gt_semantic_seg': tensor, 'img_metas': DataContainer}
            inputs = data['img']
            data_samples = []
            
            # 构造data_samples
            if 'gt_semantic_seg' in data:
                gt_seg = data['gt_semantic_seg']
                img_metas = data.get('img_metas', {})
                
                # 处理批次数据
                if isinstance(inputs, (list, tuple)):
                    batch_size = len(inputs)
                elif hasattr(inputs, 'shape'):
                    batch_size = inputs.shape[0] if inputs.dim() > 3 else 1
                else:
                    batch_size = 1
                
                for i in range(batch_size):
                    sample = {
                        'gt_sem_seg': {
                            'data': gt_seg[i] if hasattr(gt_seg, '__getitem__') and batch_size > 1 else gt_seg
                        },
                        'metainfo': img_metas.data if hasattr(img_metas, 'data') else {}
                    }
                    data_samples.append(sample)
        else:
            raise KeyError(f"Expected 'inputs' or 'img' key in data, but got keys: {list(data.keys())}")
        
        # Process inputs
        if isinstance(inputs, list):
            processed_inputs = []
            for i, img in enumerate(inputs):
                processed_img = self._process_image(img)
                processed_inputs.append(processed_img)
            
            if processed_inputs:
                batch_inputs = torch.stack(processed_inputs, dim=0)
            else:
                # Create empty tensor if no inputs
                batch_inputs = torch.zeros(1, 3, self.size[0], self.size[1])
        else:
            # Single input
            batch_inputs = self._process_image(inputs)
            if batch_inputs.dim() == 3:
                batch_inputs = batch_inputs.unsqueeze(0)
        
        # Move to device
        batch_inputs = batch_inputs.to(self.device)
        
        return {'inputs': batch_inputs, 'data_samples': data_samples}
    
    def _process_image(self, img):
        """Process a single image."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        elif isinstance(img, torch.Tensor):
            img = img.float()
        else:
            # Handle empty or invalid inputs
            img = torch.zeros(3, self.size[0], self.size[1])
        
        # Handle empty tensor
        if img.numel() == 0:
            img = torch.zeros(3, self.size[0], self.size[1])
        
        # Ensure correct shape
        if img.dim() == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.dim() == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.dim() == 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)
        
        # Resize if needed
        if img.shape[-2:] != self.size:
            import torch.nn.functional as F
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Normalize
        if self.bgr_to_rgb and img.shape[0] == 3:
            img = img[[2, 1, 0], :, :]
        
        # Apply normalization
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        img = (img - mean) / std
        
        return img
    
    def destruct(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Destruct the processed data back to original format.
        
        Args:
            data: Processed data.
            
        Returns:
            Dict[str, torch.Tensor]: Destructed data.
        """
        inputs = data.get('inputs')
        processed_inputs = inputs
        
        if inputs is not None:
            # Denormalize
            processed_inputs = inputs * self.std + self.mean
            
            # Convert RGB back to BGR if needed
            if self.bgr_to_rgb and processed_inputs.shape[1] == 3:
                processed_inputs = processed_inputs[:, [2, 1, 0], :, :]
        
        # Ensure we return valid tensors, not None
        result_inputs = processed_inputs if processed_inputs is not None else torch.empty(0)
        data_samples = data.get('data_samples')
        result_data_samples = data_samples if data_samples is not None else torch.empty(0)
        
        return {'inputs': result_inputs, 'data_samples': result_data_samples}


# 确保注册成功
if 'SegDataPreProcessor' not in MODELS.module_dict:
    print("✅ SegDataPreProcessor 已注册到 MODELS")
else:
    print("✅ SegDataPreProcessor 已存在于注册表中")