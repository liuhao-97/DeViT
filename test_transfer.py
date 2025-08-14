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

from models.de_vit import VisionTransformer

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


def generate_compact_model(pruned_model: VisionTransformer):
    """
    根据一个带有gate掩码的大模型，生成一个物理上更紧凑的新模型。
    """
    print("Step 2: 正在分析 'gate' 并创建紧凑模型...")
    
    # --- 分析Gate，确定新架构 ---
    # 假设所有层的剪枝率相同，所以只看第一个block
    first_block = pruned_model.blocks[0]
    
    # 保留的注意力头索引
    head_mask = first_block.attn.gate.bool()
    print(f"注意力头掩码: {head_mask}\n")
    new_num_heads = int(head_mask.sum())
    
    # 保留的MLP神经元索引
    neuron_mask = first_block.mlp.gate.bool()
    new_mlp_hidden_dim = int(neuron_mask.sum())
    
    # 获取原始模型的配置
    embed_dim = 768
    depth = len(pruned_model.blocks)
    patch_size = pruned_model.patch_embed.patch_size[0]
    num_classes = pruned_model.num_classes
    
    # 计算新的 mlp_ratio
    # Mlp hidden_dim = embed_dim * mlp_ratio
    # new_mlp_ratio = new_mlp_hidden_dim / embed_dim
    new_mlp_ratio = 4

    print(f"新模型架构确定: embed_dim={embed_dim}, depth={depth}, num_heads={new_num_heads}, new_mlp_ratio={new_mlp_ratio:.4f}\n")

    # --- 创建新的紧凑模型实例 ---
    compact_model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=3,
        mlp_ratio=1,
        qkv_bias=True,
        norm_layer=torch.nn.LayerNorm,
        distilled=True,
        num_classes=num_classes
    )
    
    print("Step 3: 正在进行权重迁移...")
    
    # --- 权重迁移 ---
    pruned_sd = pruned_model.state_dict()
    compact_sd = compact_model.state_dict()
    # print(compact_sd)
    print(compact_sd.keys())
    # 迁移与剪枝无关的权重
    for name in compact_sd.keys():
        if 'attn.' not in name and 'mlp.' not in name:
            if name in pruned_sd:
                if compact_sd[name].shape != pruned_sd[name].shape:
                    print(f"警告: {name} 的形状不匹配！")
                else:
                    compact_sd[name].copy_(pruned_sd[name])

    # 迁移剪枝相关的权重（核心部分）
    head_dim = embed_dim // new_num_heads

    for i in range(depth):
        p_b = f'blocks.{i}.' # pruned_block_prefix
        c_b = f'blocks.{i}.' # compact_block_prefix
        
        # --- 迁移 Attention 权重 ---
        # qkv 权重和偏置
        # 原始qkv权重形状: (embed_dim*3, embed_dim)
        # 我们需要按头来选择权重
        qkv_weight = pruned_sd[p_b + 'attn.qkv.weight']
        qkv_bias = pruned_sd[p_b + 'attn.qkv.bias']
        
        # q, k, v 各自的权重和偏置
        q_w, k_w, v_w = qkv_weight.chunk(3, dim=0)
        q_b, k_b, v_b = qkv_bias.chunk(3, dim=0)

        # 按头进行切片
        # 原始每个头的维度
        old_head_dim = pruned_model.embed_dim // pruned_model.blocks[0].attn.num_heads
        
        # 选择有效的头
        compact_q_w = q_w.reshape(pruned_model.blocks[0].attn.num_heads, old_head_dim, embed_dim)[head_mask]
        compact_k_w = k_w.reshape(pruned_model.blocks[0].attn.num_heads, old_head_dim, embed_dim)[head_mask]
        compact_v_w = v_w.reshape(pruned_model.blocks[0].attn.num_heads, old_head_dim, embed_dim)[head_mask]
        
        compact_q_b = q_b.reshape(pruned_model.blocks[0].attn.num_heads, old_head_dim)[head_mask]
        compact_k_b = k_b.reshape(pruned_model.blocks[0].attn.num_heads, old_head_dim)[head_mask]
        compact_v_b = v_b.reshape(pruned_model.blocks[0].attn.num_heads, old_head_dim)[head_mask]

        # 重新组合成紧凑的qkv权重
        compact_sd[c_b + 'attn.qkv.weight'] = torch.cat([
            compact_q_w.reshape(-1, embed_dim), 
            compact_k_w.reshape(-1, embed_dim), 
            compact_v_w.reshape(-1, embed_dim)
        ], dim=0)
        compact_sd[c_b + 'attn.qkv.bias'] = torch.cat([
            compact_q_b.flatten(), 
            compact_k_b.flatten(), 
            compact_v_b.flatten()
        ], dim=0)

        # proj 权重和偏置
        # proj 的输入维度需要被剪枝
        compact_sd[c_b + 'attn.proj.weight'] = pruned_sd[p_b + 'attn.proj.weight'][:, head_mask.repeat_interleave(old_head_dim)]
        compact_sd[c_b + 'attn.proj.bias'] = pruned_sd[p_b + 'attn.proj.bias']
        
        # --- 迁移 MLP 权重 ---
        # fc1: 输出维度被剪枝 (选择行)
        compact_sd[c_b + 'mlp.fc1.weight'] = pruned_sd[p_b + 'mlp.fc1.weight'][neuron_mask]
        compact_sd[c_b + 'mlp.fc1.bias'] = pruned_sd[p_b + 'mlp.fc1.bias'][neuron_mask]

        # fc2: 输入维度被剪枝 (选择列)
        compact_sd[c_b + 'mlp.fc2.weight'] = pruned_sd[p_b + 'mlp.fc2.weight'][:, neuron_mask]
        compact_sd[c_b + 'mlp.fc2.bias'] = pruned_sd[p_b + 'mlp.fc2.bias']

    # 加载新的state_dict
    compact_model.load_state_dict(compact_sd)
    print("权重迁移完成！\n")
    return compact_model



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
    print('student deit model:', model)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
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


    output_dir = f'output_imagenet/cifar100_div4/deit_base_distilled_patch16_224/distill_sub/lr8e-05-bs256-epochs10-grad1.0-wd0-wm5-gama0.2_0.1_0.3/sub-dataset{args.start_division}'


    loaded_neuron_mask_np = np.load(os.path.join(output_dir, 'neuron_mask.npy'))
    # recover back to PyTorch tensor list：
    loaded_neuron_mask = [torch.from_numpy(arr) for arr in loaded_neuron_mask_np]
    loaded_head_mask_np = np.load(os.path.join(output_dir, 'head_mask.npy'))
    loaded_head_mask = [torch.from_numpy(arr) for arr in loaded_head_mask_np]
    print('loaded_neuron_mask:', loaded_neuron_mask)
    print('loaded_head_mask:', loaded_head_mask)
    

    model_to_retest = create_model(
        args.model,
        pretrained=True,
        checkpoint_path=os.path.join(output_dir, 'checkpoint.pth'),
        num_classes=25,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model_to_retest.to(args.device)

    mlp_neuron_shrink(model_to_retest, loaded_neuron_mask)
    attn_head_shrink(model_to_retest, loaded_head_mask)
    test_stats = evaluate(data_loader=data_loader_val, model=model_to_retest, device=args.device)
    print(test_stats)

    compact_model = generate_compact_model(model_to_retest)
    test_stats = evaluate(data_loader=data_loader_val, model=compact_model, device=args.device)
    print(test_stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
