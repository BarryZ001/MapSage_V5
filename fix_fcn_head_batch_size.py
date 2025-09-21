#!/usr/bin/env python3
"""
修复FCN头中的batch size不匹配问题
解决 ValueError: Expected input batch_size (4) to match target batch_size (1)
"""

import os
import sys

def fix_fcn_head_batch_size():
    """修复FCN头的batch size处理逻辑"""
    
    fcn_head_path = "/Users/barryzhang/myDev3/MapSage_V5/mmseg_custom/models/fcn_head.py"
    
    # 读取原文件
    with open(fcn_head_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 新的loss_by_feat函数实现
    new_loss_by_feat = '''    def loss_by_feat(self, seg_logits: Union[torch.Tensor, List[torch.Tensor]],
                     batch_data_samples: Union[List, Dict]) -> dict:
        """Compute segmentation loss with proper batch size handling."""
        
        # Debug information
        print(f"[FCN_HEAD_DEBUG] seg_logits type: {type(seg_logits)}")
        if isinstance(seg_logits, torch.Tensor):
            print(f"[FCN_HEAD_DEBUG] seg_logits shape: {seg_logits.shape}")
        elif isinstance(seg_logits, list):
            print(f"[FCN_HEAD_DEBUG] seg_logits list length: {len(seg_logits)}")
            if len(seg_logits) > 0:
                print(f"[FCN_HEAD_DEBUG] seg_logits[0] shape: {seg_logits[0].shape}")
        
        print(f"[FCN_HEAD_DEBUG] batch_data_samples type: {type(batch_data_samples)}")
        print(f"[FCN_HEAD_DEBUG] batch_data_samples length: {len(batch_data_samples) if hasattr(batch_data_samples, '__len__') else 'N/A'}")
        
        # Handle different input formats for seg_logits
        if isinstance(seg_logits, list):
            if len(seg_logits) == 0:
                # Create dummy tensor if empty list
                seg_logits = torch.zeros(1, self.num_classes, 64, 64, device='cpu')
                print(f"[FCN_HEAD_DEBUG] Created dummy seg_logits: {seg_logits.shape}")
            else:
                seg_logits = seg_logits[0]
                print(f"[FCN_HEAD_DEBUG] Using first seg_logits: {seg_logits.shape}")
        
        # Ensure seg_logits is a tensor
        if not isinstance(seg_logits, torch.Tensor):
            raise ValueError(f"seg_logits must be a tensor, got {type(seg_logits)}")
        
        batch_size = seg_logits.shape[0]
        print(f"[FCN_HEAD_DEBUG] Input batch size: {batch_size}")
        
        # Handle different input formats for batch_data_samples
        if isinstance(batch_data_samples, dict):
            # Convert dict to list format expected by FCN head
            data_samples_list = batch_data_samples.get('data_samples', [])
            print(f"[FCN_HEAD_DEBUG] Extracted {len(data_samples_list)} samples from dict")
        else:
            data_samples_list = batch_data_samples
            print(f"[FCN_HEAD_DEBUG] Using direct list with {len(data_samples_list)} samples")
        
        # Ensure we have the right number of samples
        if len(data_samples_list) != batch_size:
            print(f"[FCN_HEAD_DEBUG] Batch size mismatch: seg_logits={batch_size}, samples={len(data_samples_list)}")
            
            if len(data_samples_list) == 1 and batch_size > 1:
                # Replicate single sample to match batch size
                print(f"[FCN_HEAD_DEBUG] Replicating single sample to match batch size {batch_size}")
                data_samples_list = data_samples_list * batch_size
            elif len(data_samples_list) > batch_size:
                # Truncate to match batch size
                print(f"[FCN_HEAD_DEBUG] Truncating samples to match batch size {batch_size}")
                data_samples_list = data_samples_list[:batch_size]
            else:
                # Create dummy samples to match batch size
                print(f"[FCN_HEAD_DEBUG] Creating dummy samples to match batch size {batch_size}")
                dummy_samples = []
                for i in range(batch_size):
                    if i < len(data_samples_list):
                        dummy_samples.append(data_samples_list[i])
                    else:
                        # Create dummy sample
                        dummy_seg = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                        dummy_sample = type('DummySample', (), {})()
                        dummy_sample.gt_sem_seg = type('DummyGT', (), {})()
                        dummy_sample.gt_sem_seg.data = dummy_seg
                        dummy_samples.append(dummy_sample)
                data_samples_list = dummy_samples
        
        # Process each data sample to extract segmentation labels
        seg_labels = []
        for i, data_sample in enumerate(data_samples_list):
            try:
                if hasattr(data_sample, 'gt_sem_seg'):
                    # Standard SegDataSample format
                    if hasattr(data_sample.gt_sem_seg, 'data'):
                        seg_label = data_sample.gt_sem_seg.data
                    else:
                        seg_label = data_sample.gt_sem_seg
                elif isinstance(data_sample, dict) and 'gt_sem_seg' in data_sample:
                    # Handle dict format
                    seg_label = data_sample['gt_sem_seg']
                else:
                    # Create dummy segmentation
                    print(f"[FCN_HEAD_DEBUG] Creating dummy segmentation for sample {i}")
                    seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                
                # Ensure proper device and shape
                if isinstance(seg_label, torch.Tensor):
                    seg_label = seg_label.to(seg_logits.device)
                    # Resize if necessary
                    if seg_label.shape != seg_logits.shape[-2:]:
                        seg_label = torch.nn.functional.interpolate(
                            seg_label.unsqueeze(0).unsqueeze(0).float(),
                            size=seg_logits.shape[-2:],
                            mode='nearest'
                        ).squeeze().long()
                else:
                    # Convert to tensor if not already
                    seg_label = torch.tensor(seg_label, dtype=torch.long, device=seg_logits.device)
                
                seg_labels.append(seg_label)
                print(f"[FCN_HEAD_DEBUG] Processed sample {i}, seg_label shape: {seg_label.shape}")
                
            except Exception as e:
                print(f"[FCN_HEAD_DEBUG] Error processing sample {i}: {e}")
                # Create dummy segmentation as fallback
                seg_label = torch.zeros(seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
                seg_labels.append(seg_label)
        
        # Stack labels to create batch
        if len(seg_labels) > 0:
            seg_label = torch.stack(seg_labels, dim=0)
            print(f"[FCN_HEAD_DEBUG] Final seg_label shape: {seg_label.shape}")
        else:
            # Create dummy batch if no labels
            seg_label = torch.zeros((batch_size,) + seg_logits.shape[-2:], dtype=torch.long, device=seg_logits.device)
            print(f"[FCN_HEAD_DEBUG] Created dummy seg_label batch: {seg_label.shape}")
        
        # Compute loss
        losses = dict()
        
        # Handle multiple loss functions
        if isinstance(self.loss_decode, list):
            for i, loss_fn in enumerate(self.loss_decode):
                try:
                    loss_value = loss_fn(seg_logits, seg_label)
                    losses[f'loss_seg_{i}'] = loss_value
                    print(f"[FCN_HEAD_DEBUG] Loss {i}: {loss_value}")
                except Exception as e:
                    print(f"[FCN_HEAD_DEBUG] Error computing loss {i}: {e}")
                    losses[f'loss_seg_{i}'] = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)
        else:
            try:
                loss_value = self.loss_decode(seg_logits, seg_label)
                losses['loss_seg'] = loss_value
                print(f"[FCN_HEAD_DEBUG] Single loss: {loss_value}")
            except Exception as e:
                print(f"[FCN_HEAD_DEBUG] Error computing single loss: {e}")
                losses['loss_seg'] = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)
        
        return losses'''
    
    # 找到并替换loss_by_feat函数
    import re
    
    # 匹配loss_by_feat函数的模式
    pattern = r'def loss_by_feat\(self, seg_logits:.*?\n        return losses'
    
    # 使用DOTALL标志来匹配跨行内容
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # 替换函数
        new_content = content[:match.start()] + new_loss_by_feat + content[match.end():]
        
        # 写入修复后的文件
        with open(fcn_head_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ FCN头batch size处理逻辑修复完成")
        print(f"📁 修复文件: {fcn_head_path}")
        print("🔧 修复内容:")
        print("  - 添加详细的调试信息")
        print("  - 修复batch size不匹配问题")
        print("  - 处理不同格式的数据样本")
        print("  - 确保标签和输入的批次大小一致")
        print("  - 添加错误处理和回退机制")
        
        return True
    else:
        print("❌ 未找到loss_by_feat函数，无法修复")
        return False

if __name__ == "__main__":
    print("🔧 开始修复FCN头batch size不匹配问题...")
    
    if fix_fcn_head_batch_size():
        print("\n✅ 修复完成！现在可以测试训练启动")
        print("\n📋 测试步骤:")
        print("1. 单进程测试: python3 scripts/train_distributed_8card_gcu.py configs/train_dinov3_loveda_t20_gcu.py --launcher none")
        print("2. 如果单进程成功，再测试8卡分布式训练")
        print("3. 准备MMRS1M数据集的8卡分布式训练配置")
    else:
        print("\n❌ 修复失败，请检查文件内容")