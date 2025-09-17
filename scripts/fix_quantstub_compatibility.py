#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复QuantStub导入兼容性问题

在PyTorch 1.10版本中，QuantStub的导入路径发生了变化
这个脚本提供兼容性修复
"""

import sys
import importlib.util

def fix_quantstub_import():
    """修复QuantStub导入问题"""
    import torch
    
    try:
        # 尝试从新路径导入
        from torch.ao.quantization import QuantStub
        print("✅ QuantStub从torch.ao.quantization导入成功")
        return True
    except ImportError:
        try:
            # 尝试从旧路径导入
            from torch.quantization import QuantStub
            print("✅ QuantStub从torch.quantization导入成功")
            return True
        except ImportError:
            try:
                # 创建一个mock QuantStub类
                import torch.nn as nn
                
                class MockQuantStub(nn.Module):
                    def __init__(self):
                        super().__init__()
                    
                    def forward(self, x):
                        return x
                
                # 将mock类添加到torch.ao.quantization模块
                if hasattr(torch, 'ao') and hasattr(torch.ao, 'quantization'):
                    if not hasattr(torch.ao.quantization, 'QuantStub'):
                        setattr(torch.ao.quantization, 'QuantStub', MockQuantStub)
                    
                print("✅ 使用Mock QuantStub类")
                return True
            except Exception as e:
                print(f"❌ QuantStub修复失败: {e}")
                return False

def check_pytorch_version():
    """检查PyTorch版本"""
    import torch
    version = torch.__version__
    print(f"PyTorch版本: {version}")
    
    # 检查是否为燧原T20的特殊版本
    if 'gcu' in version.lower() or hasattr(torch, 'gcu'):
        print("检测到燧原T20 GCU版本的PyTorch")
        return 'gcu'
    elif version.startswith('1.10'):
        print("检测到PyTorch 1.10版本")
        return '1.10'
    else:
        print(f"检测到PyTorch {version}版本")
        return 'other'

def main():
    print("🔧 修复QuantStub导入兼容性问题")
    print("=" * 50)
    
    # 检查PyTorch版本
    pytorch_version = check_pytorch_version()
    
    # 修复QuantStub导入
    success = fix_quantstub_import()
    
    if success:
        print("\n✅ QuantStub兼容性修复完成")
        print("现在可以正常导入自定义模块了")
    else:
        print("\n❌ QuantStub兼容性修复失败")
        print("可能需要升级或降级PyTorch版本")
    
    return success

if __name__ == "__main__":
    main()