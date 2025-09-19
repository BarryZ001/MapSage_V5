"""FCN Head for semantic segmentation.

This module implements a simple FCN (Fully Convolutional Network) head
for semantic segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from mmcv.cnn import build_norm_layer, build_activation_layer, ConvModule
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class FCNHead(BaseModule):
    """Fully Convolutional Network Head.
    
    This head applies a series of convolution layers followed by a classifier
    to produce segmentation predictions.
    
    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of intermediate channels.
        in_index (int): Index of input feature from backbone.
        num_convs (int): Number of convolution layers.
        concat_input (bool): Whether to concatenate input feature.
        dropout_ratio (float): Dropout ratio.
        num_classes (int): Number of classes for segmentation.
        norm_cfg (dict): Config for normalization layers.
        act_cfg (dict): Config for activation layers.
        align_corners (bool): Whether to align corners in interpolation.
        loss_decode (dict): Config for decode loss.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 in_index: int = -1,
                 num_convs: int = 2,
                 concat_input: bool = True,
                 dropout_ratio: float = 0.1,
                 num_classes: int = 19,
                 norm_cfg: dict = dict(type='SyncBN', requires_grad=True),
                 act_cfg: dict = dict(type='ReLU'),
                 align_corners: bool = False,
                 loss_decode: dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.loss_decode_cfg = loss_decode
        
        # Build loss function
        self.loss_decode = MODELS.build(loss_decode)
        
        # Build the decode head layers
        self._build_decode_layers()
        
    def _build_decode_layers(self):
        """Build decode layers."""
        # Convolution layers
        conv_layers = []
        for i in range(self.num_convs):
            in_ch = self.in_channels if i == 0 else self.channels
            conv_layers.append(
                ConvModule(
                    in_ch,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        self.convs = nn.ModuleList(conv_layers)
        
        # Concatenation layer
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        
        # Dropout layer
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
            
        # Classifier
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        
    def forward(self, inputs):
        """Forward function.
        
        Args:
            inputs (list[Tensor] | Tensor): Input features.
            
        Returns:
            Tensor: Output segmentation map.
        """
        if isinstance(inputs, (list, tuple)):
            x = inputs[self.in_index]
        else:
            x = inputs
            
        # Apply convolution layers
        output = x
        for conv in self.convs:
            output = conv(output)
            
        # Concatenate input if needed
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
            
        # Apply dropout
        if self.dropout is not None:
            output = self.dropout(output)
            
        # Apply classifier
        output = self.conv_seg(output)
        
        return output
        
    def loss(self, inputs, batch_data_samples, train_cfg):
        """Compute segmentation loss.
        
        Args:
            inputs (list[Tensor] | Tensor): Input features.
            batch_data_samples (list): Batch data samples.
            train_cfg (dict): Training config.
            
        Returns:
            dict: Loss dict.
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses
        
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """Compute loss by features.
        
        Args:
            seg_logits (Tensor): Segmentation logits.
            batch_data_samples (list): Batch data samples.
            
        Returns:
            dict: Loss dict.
        """
        # Extract ground truth labels
        seg_labels = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'gt_sem_seg'):
                seg_labels.append(data_sample.gt_sem_seg.data)
            else:
                # Fallback for different data sample formats
                seg_labels.append(data_sample['gt_semantic_seg'])
                
        seg_label = torch.stack(seg_labels, dim=0)
        
        # Resize logits to match label size if needed
        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=seg_label.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            
        # Compute loss
        loss = self.loss_decode(seg_logits, seg_label)
        
        return {'loss_seg': loss}
        
    def predict(self, inputs, batch_img_metas, test_cfg):
        """Predict segmentation results.
        
        Args:
            inputs (list[Tensor] | Tensor): Input features.
            batch_img_metas (list): Batch image metas.
            test_cfg (dict): Test config.
            
        Returns:
            list: Segmentation results.
        """
        seg_logits = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas)
        
    def predict_by_feat(self, seg_logits, batch_img_metas):
        """Predict by features.
        
        Args:
            seg_logits (Tensor): Segmentation logits.
            batch_img_metas (list): Batch image metas.
            
        Returns:
            list: Segmentation results.
        """
        # Apply softmax to get probabilities
        seg_pred = F.softmax(seg_logits, dim=1)
        
        # Get predictions
        seg_pred = seg_pred.argmax(dim=1)
        
        # Convert to list of results
        results = []
        for i in range(seg_pred.shape[0]):
            results.append(seg_pred[i].cpu().numpy())
            
        return results