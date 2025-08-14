import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import PatchEmbed, DropPath
from timm.models.vision_transformer import _cfg
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

args = parser.parse_args()


# Assuming the VisionTransformer, Block, Attention, Mlp classes are defined as provided.
# No changes needed to them, as gates will be all 1s in the new model.

def build_small_model(old_model, neuron_mask, head_mask, num_classes, args):
    # Extract kept indices from masks (using the first layer's mask since all are identical)
    neuron_mask_0 = neuron_mask[0]  # Shape: (original_hidden,) e.g., (3072,)
    head_mask_0 = head_mask[0]      # Shape: (original_num_heads,) e.g., (12,)
    
    kept_neuron_ids = torch.nonzero(neuron_mask_0 == 1).squeeze(1)  # Tensor of 768 indices
    kept_head_ids = torch.nonzero(head_mask_0 == 1).squeeze(1)      # Tensor of 3 indices
    
    original_embed_dim = old_model.embed_dim  # 768
    original_num_heads = old_model.blocks[0].attn.num_heads  # 12
    original_head_dim = original_embed_dim // original_num_heads  # 64
    original_mlp_hidden = old_model.blocks[0].mlp.hidden_features  # 3072
    
    new_num_heads = len(kept_head_ids)  # 3
    new_embed_dim = new_num_heads * original_head_dim  # 192
    new_mlp_hidden = len(kept_neuron_ids)  # 768
    new_mlp_ratio = new_mlp_hidden / new_embed_dim  # 4.0
    
    # Create the small model
    new_model = create_model(
        'dedeit_pruned',  # 使用预定义的小模型结构
        num_classes=25,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    
    # Compute kept dimensions for embed_dim (chunks corresponding to kept heads)
    kept_dims = torch.cat([
        torch.arange(
            kept_head_ids[i].item() * original_head_dim, 
            (kept_head_ids[i].item() + 1) * original_head_dim
        ) for i in range(new_num_heads)
    ])
    
    # Copy patch_embed
    new_model.patch_embed.proj.weight.data = old_model.patch_embed.proj.weight.data[kept_dims]
    new_model.patch_embed.proj.bias.data = old_model.patch_embed.proj.bias.data[kept_dims]
    
    # Copy cls_token and dist_token
    new_model.cls_token.data = old_model.cls_token.data[..., kept_dims]
    if old_model.dist_token is not None:
        new_model.dist_token.data = old_model.dist_token.data[..., kept_dims]
    
    # Copy pos_embed
    new_model.pos_embed.data = old_model.pos_embed.data[..., kept_dims]
    
    # Copy blocks
    for i in range(12):
        old_block = old_model.blocks[i]
        new_block = new_model.blocks[i]
        
        # norm1
        new_block.norm1.weight.data = old_block.norm1.weight.data[kept_dims]
        new_block.norm1.bias.data = old_block.norm1.bias.data[kept_dims]
        
        # attn.qkv
        kept_q_dims = kept_dims.clone()  # 0 to 767 sliced
        kept_k_dims = kept_q_dims + original_embed_dim
        kept_v_dims = kept_q_dims + 2 * original_embed_dim
        kept_out_dims = torch.cat((kept_q_dims, kept_k_dims, kept_v_dims))
        new_block.attn.qkv.weight.data = old_block.attn.qkv.weight.data[kept_out_dims][:, kept_dims]
        new_block.attn.qkv.bias.data = old_block.attn.qkv.bias.data[kept_out_dims]
        
        # attn.proj
        new_block.attn.proj.weight.data = old_block.attn.proj.weight.data[kept_dims][:, kept_dims]
        new_block.attn.proj.bias.data = old_block.attn.proj.bias.data[kept_dims]
        
        # norm2
        new_block.norm2.weight.data = old_block.norm2.weight.data[kept_dims]
        new_block.norm2.bias.data = old_block.norm2.bias.data[kept_dims]
        
        # mlp.fc1
        new_block.mlp.fc1.weight.data = old_block.mlp.fc1.weight.data[kept_neuron_ids][:, kept_dims]
        new_block.mlp.fc1.bias.data = old_block.mlp.fc1.bias.data[kept_neuron_ids]
        
        # mlp.fc2
        new_block.mlp.fc2.weight.data = old_block.mlp.fc2.weight.data[kept_dims][:, kept_neuron_ids]
        new_block.mlp.fc2.bias.data = old_block.mlp.fc2.bias.data[kept_dims]
    
    # Final norm
    new_model.norm.weight.data = old_model.norm.weight.data[kept_dims]
    new_model.norm.bias.data = old_model.norm.bias.data[kept_dims]
    
    # head and head_dist
    new_model.head.weight.data = old_model.head.weight.data[:, kept_dims]
    new_model.head.bias.data = old_model.head.bias.data  # Bias remains the same size (num_classes)
    new_model.head_dist.weight.data = old_model.head_dist.weight.data[:, kept_dims]
    new_model.head_dist.bias.data = old_model.head_dist.bias.data
    
    # If there are other components like pre_logits, handle similarly (assuming none here)
    
    return new_model


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


output_dir = f'output_imagenet/cifar100_div4/deit_base_distilled_patch16_224/distill_sub/lr8e-05-bs256-epochs10-grad1.0-wd0-wm5-gama0.2_0.1_0.3/sub-dataset{args.start_division}'
# In the main script, replace the shrinking and evaluation with:
# Load model_to_retest as before
model_to_retest = create_model(
    'deit_base_distilled_patch16_224',
    pretrained=True,
    checkpoint_path=os.path.join(output_dir, 'checkpoint.pth'),
    num_classes=25,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None,
)
model_to_retest.to(args.device)
loaded_neuron_mask_np = np.load(os.path.join(output_dir, 'neuron_mask.npy'))
# recover back to PyTorch tensor list：
neuron_mask = [torch.from_numpy(arr) for arr in loaded_neuron_mask_np]
loaded_head_mask_np = np.load(os.path.join(output_dir, 'head_mask.npy'))
head_mask = [torch.from_numpy(arr) for arr in loaded_head_mask_np]
print('loaded_neuron_mask:', neuron_mask)
print('loaded_head_mask:', head_mask)

# Now build small model instead of shrinking
small_model = build_small_model(model_to_retest, neuron_mask, head_mask, 25, args)
small_model.to(args.device)

# Evaluate the small model
test_stats = evaluate(data_loader=data_loader_val, model=small_model, device=args.device)
print(test_stats)