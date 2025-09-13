import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any


class SegmentationDistiller(nn.Module):
    """Knowledge Distillation for Semantic Segmentation.
    
    This model implements knowledge distillation where a teacher model
    (e.g., DINOv3) transfers knowledge to a student model (e.g., SegFormer).
    
    Args:
        teacher (dict): Teacher model configuration
        student (dict): Student model configuration  
        feature_adapters (list): Feature adaptation modules
        distill_losses (dict): Distillation loss configurations
        temperature (float): Temperature for knowledge distillation
        alpha (float): Balance factor between task loss and distillation loss
    """
    
    def __init__(self,
                 teacher: dict,
                 student: dict,
                 feature_adapters: List[dict],
                 distill_losses: dict,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 **kwargs):
        super().__init__()
        
        # Import here to avoid circular imports
        try:
            from mmseg.registry import MODELS
            self.MODELS = MODELS
        except ImportError:
            raise ImportError("MMSegmentation is required for this model")
        
        # Build teacher and student models
        self.teacher = self.MODELS.build(teacher)
        self.student = self.MODELS.build(student)
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Build feature adapters (simple 1x1 convolutions)
        self.feature_adapters = nn.ModuleList()
        for adapter_cfg in feature_adapters:
            adapter = nn.Conv2d(
                in_channels=adapter_cfg['in_channels'],
                out_channels=adapter_cfg['out_channels'],
                kernel_size=adapter_cfg['kernel_size'],
                bias=adapter_cfg.get('bias', True)
            )
            self.feature_adapters.append(adapter)
        
        # Distillation parameters
        self.temperature = temperature
        self.alpha = alpha
        
        # Build distillation losses
        self.feature_loss_weight = distill_losses['feature_loss']['loss_weight']
        self.attention_loss_weight = distill_losses.get('attention_loss', {}).get('loss_weight', 0.0)
        self.task_loss_weight = distill_losses['task_loss']['loss_weight']
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, inputs: torch.Tensor, data_samples: List[Any], mode: str = 'loss') -> Union[Dict[str, torch.Tensor], List[Any]]:
        """Forward function.
        
        Args:
            inputs (torch.Tensor): Input images
            data_samples (List[Any]): Data samples with ground truth
            mode (str): Forward mode ('loss' or 'predict')
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def loss(self, inputs: torch.Tensor, data_samples: List[Any]) -> Dict[str, torch.Tensor]:
        """Calculate losses for knowledge distillation."""
        losses = {}
        
        # Extract ground truth
        gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        gt_semantic_seg = torch.stack(gt_semantic_segs, dim=0)
        
        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_features = self.teacher.backbone(inputs)
            teacher_logits = self.teacher.decode_head(teacher_features)
        
        # Student forward
        student_features = self.student.backbone(inputs)
        student_logits = self.student.decode_head(student_features)
        
        # Task loss (segmentation loss)
        task_loss = self.ce_loss(student_logits, gt_semantic_seg.squeeze(1).long())
        losses['loss_task'] = task_loss * self.task_loss_weight
        
        # Feature distillation loss
        feature_loss = self._compute_feature_loss(teacher_features, student_features)
        losses['loss_feature'] = feature_loss * self.feature_loss_weight
        
        # Knowledge distillation loss (logits)
        kd_loss = self._compute_kd_loss(teacher_logits, student_logits)
        losses['loss_kd'] = kd_loss * self.alpha
        
        # Total loss
        total_loss = losses['loss_task'] + losses['loss_feature'] + losses['loss_kd']
        losses['loss'] = total_loss
        
        return losses
    
    def _compute_feature_loss(self, teacher_features: List[torch.Tensor], 
                            student_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute feature distillation loss."""
        feature_losses = []
        
        # Only use the number of features we have adapters for
        num_features = min(len(teacher_features), len(student_features), len(self.feature_adapters))
        
        for i in range(num_features):
            t_feat = teacher_features[i]
            s_feat = student_features[i]
            
            # Adapt student features to teacher feature dimensions
            s_feat_adapted = self.feature_adapters[i](s_feat)
            
            # Resize teacher features to match student spatial dimensions
            if t_feat.shape[2:] != s_feat_adapted.shape[2:]:
                t_feat = F.interpolate(t_feat, size=s_feat_adapted.shape[2:], 
                                     mode='bilinear', align_corners=False)
            
            # Normalize features before computing loss
            t_feat_norm = F.normalize(t_feat, p=2, dim=1)
            s_feat_norm = F.normalize(s_feat_adapted, p=2, dim=1)
            
            # Compute MSE loss
            feat_loss = self.mse_loss(s_feat_norm, t_feat_norm)
            feature_losses.append(feat_loss)
        
        if feature_losses:
            return sum(feature_losses) / len(feature_losses)
        else:
            device = teacher_features[0].device if teacher_features else torch.device('cpu')
            return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    def _compute_kd_loss(self, teacher_logits: torch.Tensor, 
                        student_logits: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss on logits."""
        # Resize logits to same size if needed
        if teacher_logits.shape != student_logits.shape:
            teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[2:],
                                         mode='bilinear', align_corners=False)
        
        # Apply temperature scaling
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
        kd_loss = kd_loss * (self.temperature ** 2)
        
        return kd_loss
    
    def predict(self, inputs: torch.Tensor, data_samples: List[Any]) -> List[Any]:
        """Predict segmentation results using student model."""
        # Use student model for inference
        student_features = self.student.backbone(inputs)
        student_logits = self.student.decode_head(student_features)
        
        # Convert logits to predictions
        seg_pred = student_logits.argmax(dim=1)
        
        # Create output data samples
        try:
            from mmengine.structures import PixelData
            from mmseg.structures import SegDataSample
            
            results = []
            for i, data_sample in enumerate(data_samples):
                result = SegDataSample()
                result.pred_sem_seg = PixelData(data=seg_pred[i:i+1])
                results.append(result)
            return results
        except ImportError:
            # Fallback: return raw predictions
            return [{'pred_sem_seg': seg_pred[i:i+1]} for i in range(len(data_samples))]
    
    def extract_feat(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Extract features using student model."""
        return self.student.extract_feat(inputs)
    
    def encode_decode(self, inputs: torch.Tensor, batch_img_metas: List[dict]) -> torch.Tensor:
        """Encode images with backbone and decode into a semantic segmentation map."""
        return self.student.encode_decode(inputs, batch_img_metas)
    
    def slide_inference(self, inputs: torch.Tensor, batch_img_metas: List[dict]) -> torch.Tensor:
        """Inference by sliding-window with overlap."""
        return self.student.slide_inference(inputs, batch_img_metas)
    
    def whole_inference(self, inputs: torch.Tensor, batch_img_metas: List[dict]) -> torch.Tensor:
        """Inference with full image."""
        return self.student.whole_inference(inputs, batch_img_metas)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        # Always keep teacher in eval mode
        self.teacher.eval()
        return self


# Register the model if MODELS registry is available
try:
    from mmseg.registry import MODELS
    MODELS.register_module()(SegmentationDistiller)
except ImportError:
    pass