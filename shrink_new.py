# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 17:32
# @Author  : Falcon
# @FileName: shrink.py

import os
import time
import json
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import models.de_vit
from models.de_vit import model_config
from data.get_dataset import build_division_dataset
from engine import evaluate

from core.shrink_imp import model_shrink
from core.imp_rank import *

from utils import dist_utils
from utils.logger import create_logger
from utils.losses import DistillLoss
from utils.samplers import RASampler

from models.de_vit_neck import VisionTransformer as neck_VisionTransformer
from functools import partial
from utils.pred_utils import ProgressMeter, accuracy, AverageMeter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def get_args_parser():
    parser = argparse.ArgumentParser('DeViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--eval-batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    # Model parameters
    parser.add_argument('--model', default='deit_base_distilled_patch16_224', type=str, metavar='MODEL',
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
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
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
    parser.add_argument('--no-aug', action='store_true', help='not use aug')

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
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default=r'./dataset', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'IMNET', 'cars', 'pets', 'flowers'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num_division', metavar='N',
                        type=int,
                        default=4,
                        help='The number of sub models')
    parser.add_argument('--start-division', metavar='N',
                        type=int,
                        default=0,
                        help='The number of sub models')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    # Resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # shrinking related
    parser.add_argument('--embedding_shrinking', action='store_true', default=False)
    parser.add_argument('--neuron_shrinking', action='store_true', default=False)
    parser.add_argument('--head_shrinking', action='store_true', default=False)
    parser.add_argument('--neuron_sparsity', type=float, default=0.)
    parser.add_argument('--head_sparsity', type=float, default=0.)

    # parser.add_argument('--classifier-choose', type=int, default=12, help='number of layers to shrink')

    parser.add_argument('--shrink_ratio', type=float, default=0.3, help='shrinking ratio')
    parser.add_argument('--bound', type=float, default=0.5, help='upper bound')
    parser.add_argument('--population', type=int, default=100)

    return parser


def main(args):
    dist_utils.init_distributed_mode(args)

    # Create output path
    args.method = f'shrink'
    args.output_dir = os.path.join(args.output_dir, f'{args.dataset}_div{args.num_division}', f'{args.model}',
                                   args.method, f'sub-dataset{args.start_division}')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

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

    num_tasks = dist_utils.get_world_size()
    global_rank = dist_utils.get_rank()

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
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    logger.info(f"Creating model: {args.model}")

    model_path = os.path.join(args.model_path, f'sub-dataset{args.start_division}', 'checkpoint.pth')
    model_ckpt = torch.load(model_path, map_location='cpu')
    model = create_model(
        args.model,
        # pretrained=True, # modified, change from False to True
        # checkpoint_path=model_path,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model.load_state_dict(model_ckpt)
    print(f'load model from {model_path}')
    
    print('args.num_classes', args.num_classes)

    model_ema = None
    if args.model_ema:  # Exponential Moving Average, for model smoothing
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_without_ddp = model
    # print(model_without_ddp)
    # print(args.distributed)
    if args.distributed: # false
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters / 1e6} M')

    linear_scaled_lr = args.lr * args.batch_size * dist_utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    
    # print(model_without_ddp)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.num_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillLoss(
        criterion, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    # print(model_without_ddp)
    print('args.resume', args.resume)
    if args.resume:
        logger.info(f'Load checkpoint from [PATH]: {args.resume}')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if args.data_set != 'IMNET':
            model_without_ddp.load_state_dict(checkpoint)
        else:
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and not args.finetune and \
                    'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if args.model_ema:
                    dist_utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
    

    # output_dir = Path(os.path.join(args.output_dir, f'sub-dataset{args.start_division}'))
    os.makedirs(output_dir, exist_ok=True)

    if args.embedding_shrinking:
        kls = []
        source_model_path = os.path.join(args.model_path, f'sub-dataset{args.start_division}', 'checkpoint.pth')
        source_checkpoint = torch.load(source_model_path, map_location='cpu')
        source_state_dict = source_checkpoint['model'] if 'model' in source_checkpoint else source_checkpoint

        source_model = create_model(
            args.model,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        source_model.load_state_dict(source_state_dict)
        source_model.cuda()
        source_model.eval()
        print(f'load model from {source_model_path}')
        
        test_stats = evaluate(data_loader_val, source_model, device)
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.4f}%")
        

        candidate_index = range(768) # need to modify
        for delete_ind in candidate_index:

            target_model_new_dict  = {}
            for k, v in source_state_dict.items():
                if "qkv.weight" in k or "head.weight" in k or "mlp.fc1.weight" in k or 'head_dist.weight' in k :
                    new_v = v[:,torch.arange(v.size(1))!=delete_ind]     
                elif "cls_token" in k or "pos_embed" in k or 'dist_token' in k:
                    new_v = v[:,:,torch.arange(v.size(2))!=delete_ind]
                elif "patch_embed" in k or "norm" in k  or "fc2" in k or "attn.proj" in k:
                    new_v = v[torch.arange(v.size(0))!=delete_ind]
                else:
                    new_v = v
                
                target_model_new_dict[k] = new_v
            
            target_model = neck_VisionTransformer(patch_size=16, embed_dim=767, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), distilled=True)
            target_model.reset_classifier(num_classes=25)
            target_model.load_state_dict(target_model_new_dict)
            target_model.cuda()
            target_model.eval()

                
            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.3f')
            top5 = AverageMeter('Acc@5', ':6.3f')
            kl = AverageMeter('KL', ':6.3f')
            cos = AverageMeter('Cosine', ':6.3f')

            progress = ProgressMeter(
                len(data_loader_train),
                [batch_time, losses, top1, top5],
                prefix='Test: ')


            kl_meter = AverageMeter('KL', ':6.3f')
            with torch.no_grad():
                for i, (images, target) in enumerate(data_loader_train):
                    images = images.cuda( non_blocking=True)
                    # target = target.cuda( non_blocking=True)

                    with autocast():
                        output = target_model(images)    
                        print(f'output shape: {output.shape}')
                        source_output = source_model(images)
                        print(f'source_output shape: {source_output.shape}')

                        logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
                        softmax = torch.nn.Softmax(dim=1).cuda()
                        distil_loss = torch.sum(
                            torch.sum(softmax(source_output) * (logsoftmax(source_output) - logsoftmax(output)), dim=1))
                    kl_meter.update(distil_loss.item(), 1)
                        
            total_kl_for_dim = kl_meter.sum
            print(f"Total KL divergence for pruned dim {delete_ind}: {total_kl_for_dim}")
            kls.append(total_kl_for_dim)

        with open(output_dir / "Deit_base_12_neck_768_kl_5k.txt", 'a') as f:
            for s in kls:
                f.write(str(s) + '\n')

        sorted_id = sorted(range(len(kls)), key=lambda k: kls[k])
        with open(output_dir / "Deit_base_12_neck_768_kl_5k_192.txt", 'w') as f:
            for s in sorted_id[:576]:
                f.write(str(s) + '\n')
                

    # modified: manually set neuron_spartisy and head_sparsity
    neuron_sparsity = np.full(12, 4/6)
    head_sparsity = np.full(12, 4/6)

    if args.neuron_shrinking:
        logger.info('Start shrink neuron')
        print('mlp_neuron_rank(model_without_ddp, data_loader_train):',mlp_neuron_rank(model_without_ddp, data_loader_train))
        # modified to generate unified mask 
        neuron_mask = mlp_neuron_mask(model_without_ddp, neuron_sparsity, mlp_neuron_rank(model_without_ddp, data_loader_train))
        mlp_hidden_dim = model_without_ddp.blocks[0].mlp.fc1.out_features
        print(f'mlp_hidden_dim:', mlp_hidden_dim)
        # 调用新函数生成一致的掩码
        # neuron_mask = generate_consistent_masks(
        #     model=model_without_ddp, 
        #     data_loader=data_loader_train,
        #     rank_function=mlp_neuron_rank,
        #     sparsity_ratios=neuron_sparsity,
        #     num_elements=mlp_hidden_dim,
        #     device=device
        # )
        
        print(f'neuron_mask:', neuron_mask)
        for i, t in enumerate(neuron_mask):
            count = torch.sum(t == 1.).item()
            length = t.numel()  # or len(t) since they are 1D tensors
            print(f"neuron Tensor {i+1}: {count} ones out of {length} elements")
        # mlp_neuron_shrink(model_without_ddp, neuron_mask)
    
        neuron_mask_np = np.stack([mask.cpu().numpy() for mask in neuron_mask])
        np.save(os.path.join(output_dir, 'neuron_mask'), neuron_mask_np)



    if args.head_shrinking:
        logger.info('Start shrink head')
        print('attn_head_rank(model_without_ddp, data_loader_train):', attn_head_rank(model_without_ddp, data_loader_train))
         # modified to generate unified mask 
        head_mask = attn_head_mask(model_without_ddp, head_sparsity, attn_head_rank(model_without_ddp, data_loader_train))
        num_heads = model_without_ddp.blocks[0].attn.num_heads
        print(f'num_heads:', num_heads)
        # 调用新函数生成一致的掩码
        # head_mask = generate_consistent_masks(
        #     model=model_without_ddp,
        #     data_loader=data_loader_train,
        #     rank_function=attn_head_rank,
        #     sparsity_ratios=head_sparsity,
        #     num_elements=num_heads,
        #     device=device
        # )
        print(f'head_mask:', head_mask)
        for i, t in enumerate(head_mask):
            count = torch.sum(t == 1.).item()
            length = t.numel()  # or len(t) since they are 1D tensors
            print(f"head Tensor {i+1}: {count} ones out of {length} elements")
        # attn_head_shrink(model_without_ddp, head_mask)

        head_mask_np = np.stack([mask.cpu().numpy() for mask in head_mask])
        np.save(os.path.join(output_dir, 'head_mask'), head_mask_np)

    logger.info(f"Start training for {args.epochs} epochs in sub-dataset{args.start_division}")
    
    





    logger.info(f'Finish shrinking on sub-dataset{args.start_division}')

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.4f}%")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeViT shrinking script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
