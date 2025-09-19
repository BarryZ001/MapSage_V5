#!/usr/bin/env python3
"""
自定义的EncoderDecoder模型，用于解决T20服务器上的注册表问题
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class EncoderDecoder(BaseModel):
    """自定义的EncoderDecoder模型，兼容T20服务器环境"""
    
    def __init__(self, 
                 backbone: Optional[Dict] = None,
                 decode_head: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 auxiliary_head: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        
        # 存储配置以供后续使用
        self.backbone_cfg = backbone
        self.decode_head_cfg = decode_head
        self.neck_cfg = neck
        self.auxiliary_head_cfg = auxiliary_head
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 构建模型组件
        if backbone is not None:
            self.backbone = MODELS.build(backbone)
        else:
            self.backbone = nn.Identity()
            
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None
            
        if decode_head is not None:
            self.decode_head = MODELS.build(decode_head)
        else:
            self.decode_head = nn.Identity()
            
        if auxiliary_head is not None:
            self.auxiliary_head = MODELS.build(auxiliary_head)
        else:
            self.auxiliary_head = None
    
    def extract_feat(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """提取特征"""
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x if isinstance(x, (list, tuple)) else [x]
    
    def encode_decode(self, inputs: torch.Tensor, batch_img_metas: List[Dict]) -> torch.Tensor:
        """编码-解码过程"""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head(x)
        return seg_logits
    
    def forward(self, 
                inputs: torch.Tensor, 
                data_samples: Optional[Any] = None, 
                mode: str = 'tensor') -> Union[Dict, List, torch.Tensor]:
        """前向传播"""
        
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.encode_decode(inputs, [])
        else:
            raise ValueError(f"Invalid mode '{mode}'. "
                           "Only supports loss, predict and tensor mode")
    
    def loss(self, inputs: torch.Tensor, data_samples: Any) -> Dict[str, torch.Tensor]:
        """计算损失"""
        x = self.extract_feat(inputs)
        
        losses = dict()
        
        # 主解码头损失
        if hasattr(self.decode_head, 'loss_by_feat'):
            loss_decode = self.decode_head.loss_by_feat(x, data_samples)
        elif hasattr(self.decode_head, 'loss'):
            loss_decode = self.decode_head.loss(x, data_samples)
        else:
            # 简单的占位符损失
            loss_decode = {'loss_seg': torch.tensor(0.0, requires_grad=True, device=inputs.device)}
        
        losses.update(loss_decode)
        
        # 辅助解码头损失
        if self.auxiliary_head is not None:
            if hasattr(self.auxiliary_head, 'loss_by_feat'):
                loss_aux = self.auxiliary_head.loss_by_feat(x, data_samples)
            elif hasattr(self.auxiliary_head, 'loss'):
                loss_aux = self.auxiliary_head.loss(x, data_samples)
            else:
                loss_aux = {'loss_aux': torch.tensor(0.0, requires_grad=True, device=inputs.device)}
            
            losses.update(loss_aux)
        
        return losses
    
    def predict(self, inputs: torch.Tensor, data_samples: Any) -> Any:
        """预测"""
        batch_img_metas = []
        if data_samples is not None:
            if hasattr(data_samples, '__iter__'):
                for sample in data_samples:
                    if hasattr(sample, 'metainfo'):
                        batch_img_metas.append(sample.metainfo)
                    else:
                        batch_img_metas.append({})
            else:
                batch_img_metas = [{}]
        
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        
        # 简单的预测结果处理
        if data_samples is not None:
            # 将预测结果添加到data_samples中
            if hasattr(data_samples, '__iter__'):
                for i, sample in enumerate(data_samples):
                    if hasattr(sample, 'pred_sem_seg'):
                        # 假设seg_logits的形状为[B, C, H, W]
                        if i < seg_logits.shape[0]:
                            pred_mask = seg_logits[i].argmax(dim=0)
                            sample.pred_sem_seg.data = pred_mask
            return data_samples
        else:
            return seg_logits
    
    def _forward(self, inputs: torch.Tensor, data_samples: Optional[Any] = None) -> torch.Tensor:
        """内部前向传播（用于推理）"""
        return self.encode_decode(inputs, [])