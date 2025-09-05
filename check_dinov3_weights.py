import torch
import os

# 检查DINOv3权重文件
checkpoint_path = 'checkpoints/regular_checkpoint_stage_4_epoch_26.pth'

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint type: {type(ckpt)}")
    print(f"Keys in checkpoint: {list(ckpt.keys())}")
    
    # 检查model_state_dict
    if 'model_state_dict' in ckpt:
        model_keys = list(ckpt['model_state_dict'].keys())
        print(f"Model state dict keys (first 10): {model_keys[:10]}")
        print(f"Total model parameters: {len(model_keys)}")
        
        # 检查是否包含ViT相关的键
        vit_keys = [k for k in model_keys if 'patch_embed' in k or 'blocks' in k or 'norm' in k]
        print(f"ViT-related keys found: {len(vit_keys)}")
        if vit_keys:
            print(f"Sample ViT keys: {vit_keys[:5]}")
        
        # 检查是否包含分割头相关的键
        seg_keys = [k for k in model_keys if 'decode_head' in k or 'head' in k or 'classifier' in k]
        print(f"Segmentation head keys found: {len(seg_keys)}")
        if seg_keys:
            print(f"Sample seg keys: {seg_keys[:5]}")
            
        # 检查backbone相关的键
        backbone_keys = [k for k in model_keys if 'backbone' in k]
        print(f"Backbone keys found: {len(backbone_keys)}")
        if backbone_keys:
            print(f"Sample backbone keys: {backbone_keys[:5]}")
    
    # 检查模型配置
    if 'model_config' in ckpt:
        print(f"Model config: {ckpt['model_config']}")
        
    # 检查训练配置
    if 'training_config' in ckpt:
        print(f"Training config keys: {list(ckpt['training_config'].keys())}")
        
else:
    print(f"Checkpoint file not found: {checkpoint_path}")

# 同时检查官方DINOv3权重
official_dinov3_path = 'checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
if os.path.exists(official_dinov3_path):
    print(f"\n=== Checking official DINOv3 weights ===")
    print(f"Loading checkpoint: {official_dinov3_path}")
    ckpt_official = torch.load(official_dinov3_path, map_location='cpu')
    
    print(f"Official checkpoint type: {type(ckpt_official)}")
    print(f"Official keys in checkpoint: {list(ckpt_official.keys())}")
    
    if isinstance(ckpt_official, dict):
        if 'model' in ckpt_official:
            official_keys = list(ckpt_official['model'].keys())
        elif 'state_dict' in ckpt_official:
            official_keys = list(ckpt_official['state_dict'].keys())
        else:
            official_keys = list(ckpt_official.keys())
        
        print(f"Official model keys (first 10): {official_keys[:10]}")
        print(f"Total official parameters: {len(official_keys)}")
        
        # 检查ViT相关键
        official_vit_keys = [k for k in official_keys if 'patch_embed' in k or 'blocks' in k or 'norm' in k]
        print(f"Official ViT-related keys found: {len(official_vit_keys)}")
        if official_vit_keys:
            print(f"Sample official ViT keys: {official_vit_keys[:5]}")
else:
    print(f"Official DINOv3 checkpoint not found: {official_dinov3_path}")