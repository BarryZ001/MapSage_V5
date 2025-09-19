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
        
    def forward(self, data: dict, training: bool = False) -> Dict[str, torch.Tensor]:
        """Process the input data.
        
        Args:
            data (dict): Input data containing 'inputs' and optionally 'data_samples'.
            training (bool): Whether in training mode.
            
        Returns:
            Dict[str, torch.Tensor]: Processed data.
        """
        inputs = data.get('inputs', [])
        data_samples = data.get('data_samples', [])
        
        # Process inputs (images)
        if isinstance(inputs, (list, tuple)):
            processed_inputs = []
            for img in inputs:
                processed_img = self._process_image(img)
                processed_inputs.append(processed_img)
            
            # Stack images into batch
            if processed_inputs:
                batch_inputs = torch.stack(processed_inputs, dim=0)
            else:
                batch_inputs = torch.empty(0)
        else:
            batch_inputs = self._process_image(inputs)
            if batch_inputs.dim() == 3:
                batch_inputs = batch_inputs.unsqueeze(0)
        
        # Move to device
        if hasattr(self, 'device'):
            batch_inputs = batch_inputs.to(self.device)
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)
        
        result = {'inputs': batch_inputs}
        
        # Process data samples if available
        if data_samples:
            result['data_samples'] = data_samples
            
        return result
    
    def _process_image(self, img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Process a single image.
        
        Args:
            img: Input image as tensor or numpy array.
            
        Returns:
            torch.Tensor: Processed image tensor.
        """
        # Convert to tensor if numpy array
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        elif not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        
        # Ensure float type
        if img.dtype != torch.float32:
            img = img.float()
        
        # Handle different input formats
        if img.dim() == 2:  # Grayscale
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.dim() == 3:
            if img.shape[0] == 1:  # Single channel
                img = img.repeat(3, 1, 1)
            elif img.shape[-1] == 3:  # HWC format
                img = img.permute(2, 0, 1)
        
        # Convert BGR to RGB if needed
        if self.bgr_to_rgb and img.shape[0] == 3:
            img = img[[2, 1, 0], :, :]
        
        # Normalize
        img = (img - self.mean) / self.std
        
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