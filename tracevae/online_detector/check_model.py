#!/usr/bin/env python3
"""
检查模型文件内容
"""

import torch
import pprint

model_path = "../results/train/models/final.pt"

print("🔍 检查模型文件内容...")
checkpoint = torch.load(model_path, map_location='cpu')

print("📋 模型文件结构:")
if isinstance(checkpoint, dict):
    print("类型: 字典")
    print("键:")
    for key in checkpoint.keys():
        print(f"  - {key}: {type(checkpoint[key])}")
        if hasattr(checkpoint[key], 'shape'):
            print(f"    形状: {checkpoint[key].shape}")
elif isinstance(checkpoint, torch.nn.Module):
    print("类型: PyTorch模型")
    print("模型结构:")
    print(checkpoint)
else:
    print(f"类型: {type(checkpoint)}")
    if hasattr(checkpoint, 'keys'):
        print("键:", list(checkpoint.keys()))

print("\n🔧 建议的加载方式:")
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        print("使用: model.load_state_dict(checkpoint['model_state_dict'])")
    elif 'state_dict' in checkpoint:
        print("使用: model.load_state_dict(checkpoint['state_dict'])")
    else:
        print("使用: model.load_state_dict(checkpoint)")