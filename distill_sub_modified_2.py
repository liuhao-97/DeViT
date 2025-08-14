# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 15:43
# @Author  : Falcon
# @FileName: distill_sub.py
import os
import json
import argparse
import datetime
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma, get_state_dict
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from engine import evaluate, train_1epoch_qkv
from data.get_dataset import build_division_dataset

import models.de_vit
from models.de_vit import model_config
from core.imp_rank import *
from utils import dist_utils
from utils.samplers import RASampler
from utils.logger import create_logger
from utils.losses import DistillLoss
from utils.dist_utils import get_rank, get_world_size


def get_args_parser():
    parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--eval-batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--output_dir', default=r'./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune")

    # Model parameters
    parser.add_argument('--model', default='dedeit', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--no_aug', action='store_true', help="no aug")

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='vit_large_patch16_224', type=str, metavar='MODEL',
                        help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str,
                        default=r'./teacher_ckpt')
    parser.add_argument('--distillation-type', default='hard', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-inter', type=bool, default=True,
                        help="Whether to distill intermediate features")
    # parser.add_argument('--distillation-token', type=bool, default=True, help="Whether to distill token")
    parser.add_argument('--distillation-token', action='store_true', help="Whether to distill token")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    # Loss parameters
    parser.add_argument('--gama', nargs='+', default=[0.2, 0.1, 0.3], help="paras to adjust loss")

    # Dataset parameters
    parser.add_argument('--data-path', default=r'./datasets',
                        type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'IMNET', 'cars', 'pets', 'flowers'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    # Division parameters
    parser.add_argument('--num_division', metavar='N',
                        type=int,
                        default=4,
                        help='The number of sub models')
    parser.add_argument('--start-division', metavar='N',
                        type=int,
                        default=0,
                        help='The number of sub models')

    # Others
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # shrink
    parser.add_argument('--load_shrink', action='store_true', default=False)
    parser.add_argument('--shrink_checkpoint', type=str, default='')
    parser.add_argument('--neuron_shrinking', action='store_true', default=False)
    parser.add_argument('--head_shrinking', action='store_true', default=False)

    # args = parser.parse_args()
    return parser


def get_models(args, num_classes, num_sub, log):
    stu_nb = 1000 if args.model_path != '' else num_classes
    model_path = args.model_path if args.finetune else None
    resize_dim = model_config[args.teacher_model]["embed_dim"] if args.distillation_token else None
    model = create_model(
        args.model,
        pretrained=True,
        checkpoint_path=model_path,
        num_classes=stu_nb,
        resize_dim=resize_dim,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    log.info(f'Create {args.model} model\n Load ckpt from [PATH]: {model_path}')
    if args.model_path != '':
        model.reset_classifier(num_classes=num_classes)
    model.to(args.device)

    teacher_model = None
    if args.distillation_type != 'none':
        teacher_path = os.path.join(args.teacher_path, f'sub-dataset{num_sub}', 'checkpoint.pth')
        if not os.path.exists(teacher_path):
            raise ValueError(f'Teacher model path {teacher_path} does not exist. Please check the path.')
        else:
            print(f'Load teacher model from [PATH]: {teacher_path}')

        teacher_ckpt = torch.load(teacher_path, map_location='cpu')
        teacher_model = create_model(args.teacher_model,
                                     num_classes=num_classes,
                                     drop_rate=args.drop,
                                     drop_path_rate=args.drop_path,
                                     drop_block_rate=None, )
        teacher_model.load_state_dict(teacher_ckpt)
        teacher_model.to(args.device)
        teacher_model.eval()

    return model, teacher_model



def prune_and_transfer_weights(source_model_path, target_model_path=None, 
                               heads_to_keep=None, neurons_to_keep=None):
    """
    将deit_base_distilled_patch16_224模型剪枝并迁移到dedeit模型
    
    Args:
        source_model_path: 原始deit_base_distilled_patch16_224模型权重路径
        target_model_path: 保存剪枝后的dedeit模型权重路径
        heads_to_keep: 每层要保留的head索引列表，默认为[0, 1, 2]（保留前3个）
        neurons_to_keep: 每层要保留的neuron索引列表，默认为前768个
    """
    
    # 默认保留的heads和neurons
    if heads_to_keep is None:
        heads_to_keep = [0, 1, 2]  # 保留12个head中的前3个
    if neurons_to_keep is None:
        neurons_to_keep = list(range(768))  # 保留3072个neuron中的前768个
    
    # # 加载源模型
    # source_model = deit_base_distilled_patch16_224(pretrained=True, pretrained_path=source_model_path)
    # source_state_dict = source_model.state_dict()
    

    source_ckpt = torch.load(source_model_path, map_location='cpu')
    source_model = create_model('deit_base_distilled_patch16_224',
                                    num_classes=25,
                                    drop_rate=0,
                                    drop_path_rate=0.1,
                                    drop_block_rate=None, )
    source_model.load_state_dict(source_ckpt)
    source_state_dict = source_model.state_dict()
    # print(source_model)

    # 创建目标模型
    # target_model = dedeit(pretrained=False)
    # target_state_dict = target_model.state_dict()

    target_model = create_model('dedeit',
                                    num_classes=25,
                                    drop_rate=0,
                                    drop_path_rate=0.1,
                                    drop_block_rate=None, )
    target_state_dict = target_model.state_dict()

    
    # 原始模型参数
    source_embed_dim = 768
    source_num_heads = 12
    source_mlp_dim = 3072
    
    # 目标模型参数
    target_embed_dim = 192
    target_num_heads = 3
    target_mlp_dim = 768
    
    # 计算head维度
    source_head_dim = source_embed_dim // source_num_heads  # 64
    target_head_dim = target_embed_dim // target_num_heads   # 64
    
    print(f"源模型: embed_dim={source_embed_dim}, num_heads={source_num_heads}, mlp_dim={source_mlp_dim}")
    print(f"目标模型: embed_dim={target_embed_dim}, num_heads={target_num_heads}, mlp_dim={target_mlp_dim}")
    
    # 遍历所有参数
    for key in target_state_dict.keys():
        print(f"\n处理: {key}")
        
        # 1. 处理patch embedding层
        if 'patch_embed' in key:
            if 'proj.weight' in key:
                # 只复制前192个输出通道
                target_state_dict[key] = source_state_dict[key][:target_embed_dim, :, :, :]
                print(f"  Patch embed weight: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
            elif 'proj.bias' in key:
                target_state_dict[key] = source_state_dict[key][:target_embed_dim]
                print(f"  Patch embed bias: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
        
        # 2. 处理position embedding和tokens
        elif key in ['pos_embed', 'cls_token', 'dist_token']:
            target_state_dict[key] = source_state_dict[key][:, :, :target_embed_dim]
            print(f"  {key}: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
        
        # 3. 处理LayerNorm层
        elif 'norm' in key and 'blocks' not in key:
            # 最后的norm层
            target_state_dict[key] = source_state_dict[key][:target_embed_dim]
            print(f"  Final norm: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
        
        # 4. 处理分类头
        elif 'head' in key or 'head_dist' in key:
            if 'weight' in key:
                target_state_dict[key] = source_state_dict[key][:, :target_embed_dim]
                print(f"  Head weight: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
            elif 'bias' in key:
                target_state_dict[key] = source_state_dict[key].clone()
                print(f"  Head bias: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
        
        # 5. 处理Transformer blocks
        elif 'blocks' in key:
            block_idx = int(key.split('.')[1])
            
            # 5.1 处理Attention层
            if 'attn.qkv' in key:
                source_qkv = source_state_dict[key]
                
                if 'weight' in key:
                    # QKV权重分割和重组
                    source_q, source_k, source_v = source_qkv.chunk(3, dim=0)
                    
                    # 每个是 [768, 768] -> 需要提取 [192, 192]
                    # 只保留前3个head和前192个输入维度
                    target_q = extract_head_weights(source_q, source_num_heads, heads_to_keep, 
                                                  source_embed_dim, target_embed_dim)
                    target_k = extract_head_weights(source_k, source_num_heads, heads_to_keep,
                                                  source_embed_dim, target_embed_dim)
                    target_v = extract_head_weights(source_v, source_num_heads, heads_to_keep,
                                                  source_embed_dim, target_embed_dim)
                    
                    target_state_dict[key] = torch.cat([target_q, target_k, target_v], dim=0)
                    print(f"  QKV weight: {source_qkv.shape} -> {target_state_dict[key].shape}")
                
                elif 'bias' in key:
                    source_q_bias, source_k_bias, source_v_bias = source_qkv.chunk(3, dim=0)
                    
                    # 只保留前3个head的bias
                    target_q_bias = extract_head_bias(source_q_bias, source_num_heads, heads_to_keep)
                    target_k_bias = extract_head_bias(source_k_bias, source_num_heads, heads_to_keep)
                    target_v_bias = extract_head_bias(source_v_bias, source_num_heads, heads_to_keep)
                    
                    target_state_dict[key] = torch.cat([target_q_bias, target_k_bias, target_v_bias], dim=0)
                    print(f"  QKV bias: {source_qkv.shape} -> {target_state_dict[key].shape}")
            
            elif 'attn.proj' in key:
                if 'weight' in key:
                    # 投影层权重 [768, 768] -> [192, 192]
                    source_proj = source_state_dict[key]
                    target_proj = extract_proj_weights(source_proj, source_num_heads, heads_to_keep,
                                                      source_embed_dim, target_embed_dim)
                    target_state_dict[key] = target_proj
                    print(f"  Proj weight: {source_proj.shape} -> {target_state_dict[key].shape}")
                elif 'bias' in key:
                    target_state_dict[key] = source_state_dict[key][:target_embed_dim]
                    print(f"  Proj bias: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
            
            # 5.2 处理MLP层
            elif 'mlp.fc1' in key:
                if 'weight' in key:
                    # [3072, 768] -> [768, 192]
                    target_state_dict[key] = source_state_dict[key][neurons_to_keep, :target_embed_dim]
                    print(f"  FC1 weight: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
                elif 'bias' in key:
                    target_state_dict[key] = source_state_dict[key][neurons_to_keep]
                    print(f"  FC1 bias: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
            
            elif 'mlp.fc2' in key:
                if 'weight' in key:
                    # [768, 3072] -> [192, 768]
                    target_state_dict[key] = source_state_dict[key][:target_embed_dim][:, neurons_to_keep]
                    print(f"  FC2 weight: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
                elif 'bias' in key:
                    target_state_dict[key] = source_state_dict[key][:target_embed_dim]
                    print(f"  FC2 bias: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
            
            # 5.3 处理LayerNorm
            elif 'norm1' in key or 'norm2' in key:
                target_state_dict[key] = source_state_dict[key][:target_embed_dim]
                print(f"  Block norm: {source_state_dict[key].shape} -> {target_state_dict[key].shape}")
            
            # 5.4 处理gate参数（如果存在）
            elif 'gate' in key:
                if 'attn.gate' in key:
                    # 只保留前3个head的gate
                    target_state_dict[key] = torch.ones(target_num_heads)
                    print(f"  Attention gate initialized: {target_state_dict[key].shape}")
                elif 'mlp.gate' in key:
                    # 只保留前768个neuron的gate
                    target_state_dict[key] = torch.ones(target_mlp_dim)
                    print(f"  MLP gate initialized: {target_state_dict[key].shape}")
    
    # 加载剪枝后的权重到目标模型
    target_model.load_state_dict(target_state_dict)
    
    # 设置gate masks
    set_pruning_masks(target_model, heads_to_keep, neurons_to_keep)
    
    # 保存模型
    if target_model_path:
        torch.save({'model': target_state_dict}, target_model_path)
        print(f"\n模型已保存到: {target_model_path}")
    
    return target_model


def extract_head_weights(weight, source_num_heads, heads_to_keep, source_dim, target_dim):
    """提取指定head的权重"""
    # weight shape: [out_dim, in_dim]
    head_dim = source_dim // source_num_heads
    
    # 重塑为 [num_heads, head_dim, in_dim]
    weight_reshaped = weight.view(source_num_heads, head_dim, source_dim)
    
    # 选择要保留的heads
    selected_heads = weight_reshaped[heads_to_keep]  # [3, 64, 768]
    
    # 重塑回 [out_dim, in_dim] 并截取输入维度
    result = selected_heads.reshape(-1, source_dim)[:, :target_dim]  # [192, 192]
    
    return result


def extract_head_bias(bias, source_num_heads, heads_to_keep):
    """提取指定head的bias"""
    head_dim = bias.shape[0] // source_num_heads
    
    # 重塑为 [num_heads, head_dim]
    bias_reshaped = bias.view(source_num_heads, head_dim)
    
    # 选择要保留的heads
    selected_heads = bias_reshaped[heads_to_keep]  # [3, 64]
    
    # 重塑回 [out_dim]
    result = selected_heads.reshape(-1)  # [192]
    
    return result


def extract_proj_weights(weight, source_num_heads, heads_to_keep, source_dim, target_dim):
    """提取投影层权重"""
    head_dim = source_dim // source_num_heads
    
    # weight shape: [out_dim, in_dim]
    # 需要从 [768, 768] 提取 [192, 192]
    
    # 重塑为 [out_dim, num_heads, head_dim]
    weight_reshaped = weight.view(source_dim, source_num_heads, head_dim)
    
    # 选择要保留的heads并截取输出维度
    selected_heads = weight_reshaped[:target_dim, heads_to_keep, :]  # [192, 3, 64]
    
    # 重塑回 [out_dim, in_dim]
    result = selected_heads.reshape(target_dim, -1)  # [192, 192]
    
    return result


def set_pruning_masks(model, heads_to_keep, neurons_to_keep):
    """设置模型的剪枝mask"""
    for block in model.blocks:
        # 设置attention head mask
        if hasattr(block.attn, 'gate'):
            block.attn.gate.data.fill_(1.0)  # 所有保留的head都设为1
            
        # 设置MLP neuron mask  
        if hasattr(block.mlp, 'gate'):
            block.mlp.gate.data.fill_(1.0)  # 所有保留的neuron都设为1


def verify_model_architecture(model):
    """验证模型架构"""
    print("\n=== 模型架构验证 ===")
    print(f"模型类型: {type(model).__name__}")
    print(f"Embedding维度: {model.embed_dim}")
    print(f"层数: {len(model.blocks)}")
    
    # 检查第一个block
    first_block = model.blocks[0]
    print(f"\n第一个Block的结构:")
    print(f"  - Attention heads: {first_block.attn.num_heads}")
    print(f"  - MLP hidden dim: {first_block.mlp.hidden_features}")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")





def rebuild_small_model_from_pruned(large_model: nn.Module, small_model:nn.Module,  neuron_mask: list, head_mask: list, num_classes: int) -> nn.Module:
    """
    根据剪枝掩码 (mask) 从一个大的 ViT 模型重建一个新的、结构更小的模型，并复制权重。

    Args:
        large_model (nn.Module): 经过训练和剪枝的大模型 (例如 deit_base_distilled_patch16_224)。
        neuron_mask (list): MLP 神经元掩码列表。
        head_mask (list): 注意力头掩码列表。
        num_classes (int): 分类任务的类别数。

    Returns:
        nn.Module: 一个全新的、更小的模型实例，其中填充了来自大模型的权重。
    """
    print(" Starting to rebuild a smaller model from the pruned large model...")

    # 1. 从掩码确定新模型的配置
    # ----
    # 将 mask 移到 CPU 以便使用 numpy/torch 的 CPU 操作
    head_indices = torch.where(head_mask[0].cpu() == 1)[0]
    neuron_indices = torch.where(neuron_mask[0].cpu() == 1)[0]

    # 原始大模型配置 (deit_base)
    large_dim = large_model.embed_dim
    large_heads = large_model.blocks[0].attn.num_heads
    mlp_hidden_dim_large = large_model.blocks[0].mlp.fc1.out_features
    head_dim = large_dim // large_heads  # 每个头的维度，通常是 64

    # 计算新小模型的配置
    small_heads = len(head_indices)
    small_dim = small_heads * head_dim
    small_mlp_hidden_dim = len(neuron_indices)
    
    print("\nModel Configuration:")
    print(f"  - Attention Heads: {large_heads} -> {small_heads}")
    print(f"  - Embedding Dim: {large_dim} -> {small_dim}")
    print(f"  - MLP Hidden Dim:  {mlp_hidden_dim_large} -> {small_mlp_hidden_dim}")

    # 2. 创建小模型实例
    # ----
    # `dedeit_pruned` 模型在你的代码中被定义为 embed_dim=192, num_heads=3，这正好匹配
    # 从 deit_base (embed_dim=768, num_heads=12) 剪枝9个头（保留3个）后的尺寸。
    # small_model = create_model(
    #     'dedeit_pruned',  # 使用预定义的小模型结构
    #     num_classes=25,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    # )
    
    # 3. 准备权重迁移
    # ----
    large_state_dict = large_model.state_dict()
    new_state_dict = small_model.state_dict()

    # `embed_dim_indices` 是最重要的索引，用于切片所有与 embed_dim 相关的权重
    embed_dim_indices = torch.cat([
        torch.arange(idx * head_dim, (idx + 1) * head_dim) for idx in head_indices
    ]).long()

    # 4. 遍历权重并进行切片复制
    # ----
    print("\n🔄 Transferring weights...")
    
    # 辅助函数，用于获取层名中的 block 索引
    def get_block_num(name):
        return int(name.split('.')[1])

    # CLS, Distill, Position Embeddings
    new_state_dict['cls_token'] = large_state_dict['cls_token'][:, :, embed_dim_indices]
    new_state_dict['dist_token'] = large_state_dict['dist_token'][:, :, embed_dim_indices]
    new_state_dict['pos_embed'] = large_state_dict['pos_embed'][:, :, embed_dim_indices]

    # Patch Embedding
    new_state_dict['patch_embed.proj.weight'] = large_state_dict['patch_embed.proj.weight'][embed_dim_indices, ...]
    new_state_dict['patch_embed.proj.bias'] = large_state_dict['patch_embed.proj.bias'][embed_dim_indices]

    # Transformer Blocks
    for i in range(len(small_model.blocks)):
        # LayerNorms
        new_state_dict[f'blocks.{i}.norm1.weight'] = large_state_dict[f'blocks.{i}.norm1.weight'][embed_dim_indices]
        new_state_dict[f'blocks.{i}.norm1.bias'] = large_state_dict[f'blocks.{i}.norm1.bias'][embed_dim_indices]
        new_state_dict[f'blocks.{i}.norm2.weight'] = large_state_dict[f'blocks.{i}.norm2.weight'][embed_dim_indices]
        new_state_dict[f'blocks.{i}.norm2.bias'] = large_state_dict[f'blocks.{i}.norm2.bias'][embed_dim_indices]

        # Attention QKV
        qkv_row_indices = torch.cat([
            embed_dim_indices,
            embed_dim_indices + large_dim,
            embed_dim_indices + 2 * large_dim
        ]).long()
        # weight: [out_features, in_features] -> [small_dim*3, small_dim]
        w_qkv = large_state_dict[f'blocks.{i}.attn.qkv.weight']
        new_state_dict[f'blocks.{i}.attn.qkv.weight'] = w_qkv[qkv_row_indices, :][:, embed_dim_indices]
        # bias
        b_qkv = large_state_dict[f'blocks.{i}.attn.qkv.bias']
        new_state_dict[f'blocks.{i}.attn.qkv.bias'] = b_qkv[qkv_row_indices]

        # Attention Proj
        # weight: [out_features, in_features] -> [small_dim, small_dim]
        w_proj = large_state_dict[f'blocks.{i}.attn.proj.weight']
        new_state_dict[f'blocks.{i}.attn.proj.weight'] = w_proj[embed_dim_indices, :][:, embed_dim_indices]
        # bias
        b_proj = large_state_dict[f'blocks.{i}.attn.proj.bias']
        new_state_dict[f'blocks.{i}.attn.proj.bias'] = b_proj[embed_dim_indices]

        # MLP fc1
        w_fc1 = large_state_dict[f'blocks.{i}.mlp.fc1.weight']
        new_state_dict[f'blocks.{i}.mlp.fc1.weight'] = w_fc1[neuron_indices, :][:, embed_dim_indices]
        b_fc1 = large_state_dict[f'blocks.{i}.mlp.fc1.bias']
        new_state_dict[f'blocks.{i}.mlp.fc1.bias'] = b_fc1[neuron_indices]
        
        # MLP fc2
        w_fc2 = large_state_dict[f'blocks.{i}.mlp.fc2.weight']
        new_state_dict[f'blocks.{i}.mlp.fc2.weight'] = w_fc2[embed_dim_indices, :][:, neuron_indices]
        b_fc2 = large_state_dict[f'blocks.{i}.mlp.fc2.bias']
        new_state_dict[f'blocks.{i}.mlp.fc2.bias'] = b_fc2[embed_dim_indices]

    # Final LayerNorm
    new_state_dict['norm.weight'] = large_state_dict['norm.weight'][embed_dim_indices]
    new_state_dict['norm.bias'] = large_state_dict['norm.bias'][embed_dim_indices]

    # Classifier Heads
    new_state_dict['head.weight'] = large_state_dict['head.weight'][:, embed_dim_indices]
    new_state_dict['head.bias'] = large_state_dict['head.bias']
    new_state_dict['head_dist.weight'] = large_state_dict['head_dist.weight'][:, embed_dim_indices]
    new_state_dict['head_dist.bias'] = large_state_dict['head_dist.bias']

    # 5. 加载新的权重字典
    # ----
    small_model.load_state_dict(new_state_dict)
    print("\n Weight transfer complete. The new small model is ready!")
    
    return small_model



def debug_weight_copy(large_model, small_model, neuron_mask, head_mask):
    """
    逐一比较小模型的每个参数，验证它是否与大模型参数切片后的结果完全相等。

    Args:
        large_model: 干净的、未经修改的大模型。
        small_model: 重建后的小模型。
        neuron_mask: MLP神经元掩码。
        head_mask: 注意力头掩码。
    """
    print("\n" + "="*50)
    print("🕵️  Verifying weight copy, parameter by parameter...")
    print("="*50)

    # 1. 准备所有需要的索引，这部分逻辑必须和 rebuild 函数完全一致
    head_indices = torch.where(head_mask[0].cpu() == 1)[0]
    neuron_indices = torch.where(neuron_mask[0].cpu() == 1)[0]
    large_dim = large_model.embed_dim
    large_heads = large_model.blocks[0].attn.num_heads
    head_dim = large_dim // large_heads
    embed_dim_indices = torch.cat(
        [torch.arange(idx * head_dim, (idx + 1) * head_dim) for idx in head_indices]
    ).long()
    qkv_row_indices = torch.cat([
        embed_dim_indices,
        embed_dim_indices + large_dim,
        embed_dim_indices + 2 * large_dim
    ]).long()

    # 2. 获取两个模型的 state_dict
    large_state_dict = large_model.state_dict()
    small_state_dict = small_model.state_dict()

    mismatched_params = []

    # 3. 遍历小模型的每一个参数进行校验
    for name, small_param in small_state_dict.items():
        if "num_batches_tracked" in name: # 跳过BN层的非参数项
            continue
            
        if name not in large_state_dict:
            print(f"❓ WARNING: Parameter '{name}' in small model not found in large model.")
            continue

        large_param = large_state_dict[name].cpu()
        small_param_cpu = small_param.cpu()
        large_param_sliced = None

        # 4. 应用与 rebuild 函数相同的切片逻辑
        if 'patch_embed.proj' in name:
            large_param_sliced = large_param[embed_dim_indices, ...] if 'weight' in name else large_param[embed_dim_indices]
        elif 'cls_token' in name or 'dist_token' in name or 'pos_embed' in name:
            large_param_sliced = large_param[:, :, embed_dim_indices]
        elif 'blocks' in name:
            if 'norm' in name:
                large_param_sliced = large_param[embed_dim_indices]
            elif 'attn.qkv' in name:
                if 'weight' in name:
                    large_param_sliced = large_param[qkv_row_indices, :][:, embed_dim_indices]
                else:  # bias
                    large_param_sliced = large_param[qkv_row_indices]
            elif 'attn.proj' in name:
                if 'weight' in name:
                    large_param_sliced = large_param[embed_dim_indices, :][:, embed_dim_indices]
                else:  # bias
                    large_param_sliced = large_param[embed_dim_indices]
            elif 'mlp.fc1' in name:
                if 'weight' in name:
                    large_param_sliced = large_param[neuron_indices, :][:, embed_dim_indices]
                else:  # bias
                    large_param_sliced = large_param[neuron_indices]
            elif 'mlp.fc2' in name:
                if 'weight' in name:
                    large_param_sliced = large_param[embed_dim_indices, :][:, neuron_indices]
                else:  # bias
                    large_param_sliced = large_param[embed_dim_indices]
        elif 'norm' in name:  # Final norm
            large_param_sliced = large_param[embed_dim_indices]
        elif 'head' in name:
            if 'weight' in name:
                large_param_sliced = large_param[:, embed_dim_indices]
            else:  # bias
                large_param_sliced = large_param # bias is independent of input dim
        
        # 5. 进行比较
        if large_param_sliced is not None:
            # torch.equal 要求形状和数值都完全相等
            if torch.equal(large_param_sliced, small_param_cpu):
                print(f"✅ MATCH: '{name}'")
            else:
                print(f"❌ MISMATCH: '{name}'")
                mismatched_params.append(name)
        else:
            print(f"INFO: No slicing rule for '{name}', assuming direct copy.")
            if torch.equal(large_param, small_param_cpu):
                 print(f"✅ MATCH: '{name}'")
            else:
                print(f"❌ MISMATCH: '{name}'")
                mismatched_params.append(name)

    print("\n" + "="*50)
    if not mismatched_params:
        print("🎉🎉🎉 All parameters verified successfully! Weight copy logic is CORRECT.")
    else:
        print("💔 Found mismatches in the following parameters:")
        for name in mismatched_params:
            print(f"  - {name}")
    print("="*50 + "\n")


def main(args):
    dist_utils.init_distributed_mode(args)

    # Create output path
    args.method = f'distill_sub'
    args.name = f'lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-grad{args.clip_grad}' \
                f'-wd{args.weight_decay}-wm{args.warmup_epochs}-gama{args.gama[0]}_{args.gama[1]}_{args.gama[2]}'
    args.output_dir = os.path.join(args.output_dir, f'{args.dataset}_div{args.num_division}', f'{args.model}',
                                   args.method, args.name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # log init
    logger = create_logger(output_dir=args.output_dir, dist_rank=dist_utils.get_rank(), name=f"{args.method}")
    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Load dataset
    sub_dataset_path = os.path.join(args.data_path, f'sub-dataset{args.start_division}')

    train_dataset, test_dataset, division_num_classes = build_division_dataset(dataset_path=sub_dataset_path, args=args)
    args.num_classes = division_num_classes

    num_tasks = get_world_size()
    global_rank = get_rank()

    if args.repeated_aug:
        sampler_train = RASampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if args.dist_eval:
        if len(test_dataset) % num_tasks != 0:
            logger.info(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=division_num_classes)

    model, teacher_model = get_models(args, division_num_classes, args.start_division, logger)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')



    # load shrink
    if args.shrink_checkpoint != '':
        shrinked_policy = np.load(os.path.join(args.shrink_checkpoint, 'shrinked_policy.npy'))
        shrinked_acc = np.load(os.path.join(args.shrink_checkpoint, 'shrinked_accuracy.npy'))
        # print('shrinked_policy:', shrinked_policy)
        # print('shrinked_policy shape:', shrinked_policy.shape) # shrinked_policy shape: (100, 24)
        # print('shrinked_acc:', shrinked_acc)
        # print('shrinked_acc shape:', shrinked_acc.shape) # shrinked_acc shape: (100,)
        max_index = np.argmax(shrinked_acc)
        
        # print('shrinked_policy[max_index, :12]:', shrinked_policy[max_index, :12])
        # print('shrinked_policy[max_index, 12:]:',shrinked_policy[max_index, 12:])

        neuron_sparsity = shrinked_policy[max_index, :12]
        head_sparsity = shrinked_policy[max_index, 12:]

    else:
        # modified: manually set neuron_spartisy and head_sparsity
        neuron_sparsity = np.full(12, 0.75)
        head_sparsity = np.full(12, 0.75)

    if args.neuron_shrinking:
        logger.info('Start shrink neuron')
        # print('mlp_neuron_rank(model_without_ddp, data_loader_train):',mlp_neuron_rank(model_without_ddp, data_loader_train))
        ## modified to generate unified mask 
        # neuron_mask = mlp_neuron_mask(model_without_ddp, neuron_sparsity, mlp_neuron_rank(model_without_ddp, data_loader_train))
        mlp_hidden_dim = model_without_ddp.blocks[0].mlp.fc1.out_features
        print(f'mlp_hidden_dim:', mlp_hidden_dim)
        # 调用新函数生成一致的掩码
        neuron_mask = generate_consistent_masks(
            model=model_without_ddp, 
            data_loader=data_loader_train,
            rank_function=mlp_neuron_rank,
            sparsity_ratios=neuron_sparsity,
            num_elements=mlp_hidden_dim,
            device=device
        )
        
        print(f'neuron_mask:', neuron_mask)
        for i, t in enumerate(neuron_mask):
            count = torch.sum(t == 1.).item()
            length = t.numel()  # or len(t) since they are 1D tensors
            print(f"neuron Tensor {i+1}: {count} ones out of {length} elements")
        mlp_neuron_shrink(model_without_ddp, neuron_mask)

    if args.head_shrinking:
        logger.info('Start shrink head')
        # print('attn_head_rank(model_without_ddp, data_loader_train):', attn_head_rank(model_without_ddp, data_loader_train))
         ## modified to generate unified mask 
        # head_mask = attn_head_mask(model_without_ddp, head_sparsity, attn_head_rank(model_without_ddp, data_loader_train))
        num_heads = model_without_ddp.blocks[0].attn.num_heads
        print(f'num_heads:', num_heads)
        # 调用新函数生成一致的掩码
        head_mask = generate_consistent_masks(
            model=model_without_ddp,
            data_loader=data_loader_train,
            rank_function=attn_head_rank,
            sparsity_ratios=head_sparsity,
            num_elements=num_heads,
            device=device
        )
        print(f'head_mask:', head_mask)
        for i, t in enumerate(head_mask):
            count = torch.sum(t == 1.).item()
            length = t.numel()  # or len(t) since they are 1D tensors
            print(f"head Tensor {i+1}: {count} ones out of {length} elements")
        attn_head_shrink(model_without_ddp, head_mask)
    
    neuron_mask_positions = torch.nonzero(neuron_mask[0] == 1, as_tuple=False).squeeze().tolist()
    head_positions = torch.nonzero(head_mask[0] == 1, as_tuple=False).squeeze().tolist()

    source_model_path = "output_imagenet/cifar100_division4/deit_base_distilled_patch16_224/sub_no_distill/lr0.0001-bs256-epochs100-gradNone-wd0-wm5/sub-dataset0/checkpoint.pth"
    target_model_path = "output_imagenet/cifar100_div4/deit_base_distilled_patch16_224/distill_sub/lr8e-05-bs256-epochs100-grad1.0-wd0-wm5-gama0.2_0.1_0.3/sub-dataset0/dedeit_pruned.pth"
    print('positions:', head_positions)
    rebuild_small_model = prune_and_transfer_weights(
        source_model_path=source_model_path,
        target_model_path=target_model_path,
        heads_to_keep=head_positions, 
        neurons_to_keep=neuron_mask_positions  
    )
    rebuild_small_model.to(args.device)
    
    linear_scaled_lr = args.lr * args.batch_size * get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, rebuild_small_model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillLoss(base_criterion=criterion, distillation_type=args.distillation_type,
                            alpha=args.distillation_alpha, tau=args.distillation_tau)


    # debug_weight_copy(
    #     large_model=model_without_ddp,
    #     small_model=rebuild_small_model,
    #     neuron_mask=neuron_mask,
    #     head_mask=head_mask
    # )
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            rebuild_small_model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    logger.info(f"Start training for {args.epochs} epochs in sub-dataset{args.start_division}")
    output_dir = Path(os.path.join(args.output_dir, f'sub-dataset{args.start_division}'))
    os.makedirs(output_dir, exist_ok=True)
    
    neuron_mask_np = np.stack([mask.cpu().numpy() for mask in neuron_mask])
    head_mask_np = np.stack([mask.cpu().numpy() for mask in head_mask])
    np.save(os.path.join(output_dir, 'neuron_mask'), neuron_mask_np)
    np.save(os.path.join(output_dir, 'head_mask'), head_mask_np)

    # init tensorboard
    writer = SummaryWriter(log_dir=output_dir) if get_rank() == 0 else None

    # for samples, targets in data_loader_train:
    #     samples = samples.to(device, non_blocking=True)
    # student_outputs = model(samples, output_qkv=True)
    # print(student_outputs.keys())

    # for samples, targets in data_loader_train:
    #     samples = samples.to(device, non_blocking=True)
    # teacher_outputs = teacher_model(samples, output_qkv=True)
    # print(teacher_outputs.keys())
    
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_1epoch_qkv(model=rebuild_small_model, teacher_model=teacher_model, criterion=criterion, args=args,
                                       data_loader=data_loader_train, optimizer=optimizer, device=args.device,
                                       epoch=epoch, loss_scaler=loss_scaler, log=logger, max_norm=args.clip_grad,
                                       model_ema=model_ema, mixup_fn=mixup_fn)

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_temp.pth']
            for checkpoint_path in checkpoint_paths:
                dist_utils.save_on_master({
                    'model': rebuild_small_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader=data_loader_val, model=rebuild_small_model, device=args.device)
        logger.info(f"Epoch: {epoch}/{args.epochs} \t [Train] Loss: {train_stats['loss']:.4f} \t ")
        logger.info(f"Epoch: {epoch}/{args.epochs} \t [Eval] Top-1: {test_stats['acc1']:.4f} \t "
                    f"Top-5: {test_stats['acc5']:.4f} \t Loss: {test_stats['loss']:.4f} \t ")
        if writer is not None:
            writer.add_scalar('Train/loss', train_stats['loss'], epoch)
            writer.add_scalar('Train/lr', train_stats['lr'], epoch)
            writer.add_scalar('Test/loss', test_stats['loss'], epoch)
            writer.add_scalar('Test/Top1', test_stats['acc1'], epoch)
            writer.add_scalar('Test/Top5', test_stats['acc5'], epoch)

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir and dist_utils.is_main_process():
                model_checkpoint = os.path.join(output_dir, f"checkpoint.pth")
                torch.save(rebuild_small_model.state_dict(), model_checkpoint)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                with open(os.path.join(output_dir, 'result.txt'), 'w') as f:
                    f.write(f'Final Accuracy: {max_accuracy}\n'
                            f'Model config: {model_config}')
                logger.info(f'Saving model in [PATH]: {output_dir}')
        logger.info(f'Max accuracy: {max_accuracy:.4f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and dist_utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f'Epochs: {epoch} \t Training time: {total_time_str} ')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str} on sub-dataset{args.start_division}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
