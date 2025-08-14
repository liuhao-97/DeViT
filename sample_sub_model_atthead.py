# 这个程序有点问题
# patch embedding从768->192是unstructured剪枝，这块已经对attention head进行了unstructured剪枝打乱了按照head剪枝的顺序,比如head 1 剪了多,head 2 剪了少, head 3 剪了少, head 4 剪了多, 这样就会导致每个head的维度不一致
# 但是在patch剪枝之后的attention head是structured剪枝,新的head的权重矩阵不一定是均分的(因为前面是unstructure剪枝)，这样按照npy进行剪枝，比如删除head1的时候也可能删除了head2的权重
# 因此patch embedding不应该是unstructured剪枝，而是structured剪枝，按照head的npy进行剪枝，这样才能保证每个head的维度一致



import torch
import torch.nn as nn
import argparse
import numpy as np
import os
import models.de_vit
from models.de_vit import model_config
from models.de_vit import VisionTransformer
from functools import partial

parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
parser.add_argument('--output_dir', default=r'./output',
                    help='path where to save, empty for no saving')
# Model parameters
parser.add_argument('--model', default='deit_base_distilled_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'IMNET', 'cars', 'pets', 'flowers'],
                    type=str, help='Image Net dataset path')

# Division parameters
parser.add_argument('--num_division', metavar='N',
                    type=int,
                    default=4,
                    help='The number of sub models')
parser.add_argument('--start-division', metavar='N',
                    type=int,
                    default=0,
                    help='The number of sub models')

parser.set_defaults(pin_mem=True)
args = parser.parse_args()


args.method = f'shrink'
args.shrink_checkpoint = os.path.join(args.output_dir, f'{args.dataset}_div{args.num_division}', f'{args.model}',
                                args.method, f'sub-dataset{args.start_division}')

args.model_path = os.path.join(args.model_path, f'sub-dataset{args.start_division}', 'checkpoint.pth')
checkpoint = torch.load(args.model_path, map_location='cpu')
print("Loading checkpoint from: ", args.model_path)
checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint


reduce_neck = []
neck_shrink_path = os.path.join(args.shrink_checkpoint, "Deit_base_12_neck_768_kl_5k_192.txt")
with open(neck_shrink_path, 'r') as f:
    for i in f:
        reduce_neck.append(int(i))

new_dict  = {}
cnt = 1
for k, v in checkpoint.items():
    # print(k,end= ", ")
    # print(v.shape)
    new_dict[ k] = v

for k, v in checkpoint.items():
    print(k,end= ", ")
    if "qkv.weight" in k or "head.weight" in k or "mlp.fc1.weight" in k:
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(1))]
        new_v = v[:,new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    elif "cls_token" in k or "pos_embed" in k:
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(2))]
        new_v = v[:,:,new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    elif "patch_embed" in k or "norm" in k  or "fc2" in k or "attn.proj" in k:
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(0))]
        new_v = v[new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    elif 'head_dist.weight' in k :
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(1))]
        new_v = v[:,new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    elif 'dist_token' in k:
        new_index = [i not in  reduce_neck for i in torch.arange(v.size(2))]
        new_v = v[:,:,new_index]
        print(new_v.shape)
        new_dict[ k] = new_v
    else:
        print(v.shape)
        new_dict[ k] = v


model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled=True)
model.load_state_dict(new_dict)
dummy_input = torch.randn(1, 3, 32, 32)
print(f"输入张量的形状: {dummy_input.shape}\n")
with torch.no_grad():
    output = model(dummy_input)
print(f"模型输出的形状: {output.shape}")





num_heads = 12
head_dim = 64

head_mask_np = np.load(os.path.join(args.shrink_checkpoint, 'head_mask.npy'))
head_mask = [list(arr) for arr in head_mask_np]
heads_to_prune_per_block = []
for i in range(len(head_mask)):
    zero_indices = [i for i, val in enumerate(head_mask[i]) if val == 0]
    heads_to_prune_per_block.append(zero_indices)

# heads_to_prune_per_block = [[0, 1, 2, 3, 4, 5, 6, 7, 8]] * 12

for block_idx in range(12):
    heads_to_prune = heads_to_prune_per_block[block_idx]
    if not heads_to_prune:
        continue

    print(f"Block {block_idx}: Pruning heads {heads_to_prune}")
    heads_to_keep = [h for h in range(num_heads) if h not in heads_to_prune]

    # --- 处理 QKV ---
    qkv_weight_key = f"blocks.{block_idx}.attn.qkv.weight"
    qkv_bias_key = f"blocks.{block_idx}.attn.qkv.bias"
    
    # 从 new_dict (已剪枝) 而不是原始 checkpoint 中获取张量
    v_w = new_dict[qkv_weight_key]
    v_b = new_dict[qkv_bias_key]

    # (剪枝逻辑不变)
    q_w, k_w, v_w_parts = torch.chunk(v_w, 3, dim=0)
    q_b, k_b, v_b_parts = torch.chunk(v_b, 3, dim=0)
    q_w_reshaped, k_w_reshaped, v_w_reshaped = q_w.reshape(num_heads, head_dim, -1), k_w.reshape(num_heads, head_dim, -1), v_w_parts.reshape(num_heads, head_dim, -1)
    q_b_reshaped, k_b_reshaped, v_b_reshaped = q_b.reshape(num_heads, head_dim), k_b.reshape(num_heads, head_dim), v_b_parts.reshape(num_heads, head_dim)
    new_q_w, new_k_w, new_v_w = q_w_reshaped[heads_to_keep].reshape(-1, q_w.shape[1]), k_w_reshaped[heads_to_keep].reshape(-1, k_w.shape[1]), v_w_reshaped[heads_to_keep].reshape(-1, v_w_parts.shape[1])
    new_q_b, new_k_b, new_v_b = q_b_reshaped[heads_to_keep].reshape(-1), k_b_reshaped[heads_to_keep].reshape(-1), v_b_reshaped[heads_to_keep].reshape(-1)
    
    # 将剪枝后的结果更新回 new_dict
    new_dict[qkv_weight_key] = torch.cat([new_q_w, new_k_w, new_v_w], dim=0)
    new_dict[qkv_bias_key] = torch.cat([new_q_b, new_k_b, new_v_b], dim=0)
    print(f"Updated {qkv_weight_key} and {qkv_bias_key} with shape {new_dict[qkv_weight_key].shape} and {new_dict[qkv_bias_key].shape}")

    # --- 处理 Proj ---
    proj_weight_key = f"blocks.{block_idx}.attn.proj.weight"
    v_proj = new_dict[proj_weight_key]
    v_proj_reshaped = v_proj.reshape(v_proj.shape[0], num_heads, head_dim)
    new_proj_w = v_proj_reshaped[:, heads_to_keep, :].reshape(v_proj.shape[0], -1)
    new_dict[proj_weight_key] = new_proj_w

print("--- Head pruning finished ---")




neuron_mask_np = np.load(os.path.join(args.shrink_checkpoint, 'neuron_mask.npy'))
neuron_mask = [list(arr) for arr in neuron_mask_np]
neuron_to_prune_per_block = []
for i in range(len(neuron_mask)):
    zero_indices = [i for i, val in enumerate(neuron_mask[i]) if val == 0]
    neuron_to_prune_per_block.append(zero_indices)


for reduce in range(0,12):                                                
    # MLP fc1
    block_ind_w = f"blocks.{reduce}.mlp.fc1.weight"
    block_ind_b = f"blocks.{reduce}.mlp.fc1.bias"
    v_w = new_dict[block_ind_w] # 从 new_dict 读取
    v_b = new_dict[block_ind_b] # 从 new_dict 读取
    new_index = [i not in neuron_to_prune_per_block[reduce] for i in torch.arange(v_w.size(0))]
    new_dict[block_ind_w] = v_w[new_index] # 更新 new_dict
    new_dict[block_ind_b] = v_b[new_index] # 更新 new_dict
    print(f"blocks.{reduce}.mlp.fc1.weight")
    print(f"blocks.{reduce}.mlp.fc1.bias")
    
    # MLP fc2
    block_ind_w = f"blocks.{reduce}.mlp.fc2.weight"
    v_w = new_dict[block_ind_w] # 从 new_dict 读取
    new_index = [i not in neuron_to_prune_per_block[reduce] for i in torch.arange(v_w.size(1))]
    new_dict[block_ind_w] = v_w[:, new_index] # 更新 new_dict
    print(f"blocks.{reduce}.mlp.fc2.weight")

print("--- MLP pruning finished ---")

model_checkpoint = os.path.join(args.shrink_checkpoint, f"checkpoint.pth")
torch.save(new_dict, model_checkpoint)


