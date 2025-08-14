import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights, vit_l_16, ViT_L_16_Weights, vit_l_32, ViT_L_32_Weights, vit_h_14, ViT_H_14_Weights

from torchvision.transforms import Compose
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from torchvision import datasets
import argparse

import timm
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-model_idx', type=int, default=8, help="Model index: 1=vit_b_16, 2=vit_b_32, 3=vit_l_16, 4=vit_l_32, 5=vit_h_14")
parser.add_argument('-quant',  help="Enable int8 quantization for CPU")
args = parser.parse_args()


if args.quant == 'int8':
    device = torch.device("cpu")
if args.quant == 'fp32' or args.quant == 'fp16':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit_models = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, 224),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, 224),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1, 224),
    "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1, 224),
    "vit_h_14": (vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1, 224),  
}

# DeiT models from timm
deit_models = {
    "deit_tiny_patch16_224": "deit_tiny_patch16_224",
    "deit_small_patch16_224": "deit_small_patch16_224",
    "deit_base_patch16_224": "deit_base_patch16_224",
    "deit3_base_patch16_224": "deit3_base_patch16_224",
    "deit3_huge_patch14_224": "deit3_huge_patch14_224",
    "deit3_large_patch16_224": "deit3_large_patch16_224",
    "deit3_medium_patch16_224": "deit3_medium_patch16_224",
    "deit3_small_patch16_224": "deit3_small_patch16_224",
}

model_idx_to_name = {
    1: "vit_b_16",
    2: "vit_b_32",
    3: "vit_l_16",
    4: "vit_l_32",
    5: "vit_h_14",
    6: "deit_tiny_patch16_224",
    7: "deit_small_patch16_224",
    8: "deit_base_patch16_224",
    9: "deit3_base_patch16_224",
    10: "deit3_huge_patch14_224",
    11: "deit3_large_patch16_224",
    12: "deit3_medium_patch16_224",
    13: "deit3_small_patch16_224",
    14: "deit-tiny-distilled-patch16-224",  
}

# Validate model_idx
if args.model_idx not in model_idx_to_name:
    raise ValueError(f"Invalid model_idx: {args.model_idx}. Choose from 1 to 13.")

model_name = model_idx_to_name[args.model_idx]

if model_name.startswith('deit') and args.model_idx != 14:
    model = timm.create_model(model_name, pretrained=True)
    # data_config = timm.data.resolve_model_data_config(model)
    # transform = timm.data.create_transform(**data_config)
elif  model_name.startswith('vit'): 
    model_fn, weights, input_res = vit_models[model_name]
    model = model_fn(weights=weights)
    transform = weights.transforms()
elif args.model_idx == 14:
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)


  

# Move model to device
model = model.to('cuda')
model.eval()
print(model)

# Save the model state_dict to a .pth file
save_path = f"{model_name}.pth"
torch.save(model, save_path)
print(f"Model saved to {save_path}")
