import argparse
import os
import os.path as osp
import copy
import pandas as pd

import torch
import torch.nn as nn
from torch.distributed.elastic.multiprocessing.errors import record

from collections import OrderedDict

import openslide

import sys
sys.path.append('./')
from celldetr.util.distributed import init_distributed_mode, save_on_master, is_main_process, get_rank
from celldetr.util.misc import seed_everything, nested_tensor_from_tensor_list
from celldetr.util.config import ConfigDict
from celldetr.data import build_dataset, build_loader
from celldetr.data.da import DomainAdaptationDataset
from celldetr.data.wsi import FolderPatchDataset
from celldetr.models import build_model, load_state_dict
from celldetr.engine import train_one_epoch, evaluate_detection, evaluate

import wandb

def build_tgt_dataset(cfg):
    transforms = build_tgt_transforms(cfg)
    dataset = FolderPatchDataset(root=cfg.dataset.train.root,
                                transform=transforms)
    return dataset

def build_tgt_transforms(cfg):
    import torchvision.transforms.v2 as v2
    from celldetr.data.transforms import build_augmentations
    transforms = v2.Compose([
        v2.ToImage(),
        build_augmentations(cfg),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cfg.transforms.normalize.mean, 
                     std=cfg.transforms.normalize.std)
    ])
    return transforms

def da_collate_fn(batch):
    srcs_imgs, src_lbls, tgt_imgs = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(srcs_imgs + tgt_imgs)
    return samples, src_lbls

@record
def train(cfg):
    # init distributed mode
    init_distributed_mode(cfg)
    device = torch.device("cuda:" + str(cfg.gpu) if torch.cuda.is_available() else "cpu")

    # init wandb
    if not cfg.experiment.has('wandb'):
        cfg.experiment.wandb = False
    if cfg.experiment.wandb and is_main_process():
        wandb.init(project=cfg.experiment.project,
                   name=cfg.experiment.name,
                   config=cfg.as_dict())
    
    # set seed
    seed = cfg.experiment.seed + get_rank()
    seed_everything(seed)

    # build source and target dataset
    src_dataset = build_dataset(cfg.domain.src, split='train')
    tgt_dataset = build_tgt_dataset(cfg.domain.tgt)
    print("Source dataset:", len(src_dataset))
    print("Target dataset:", len(tgt_dataset))
    # combine source and target dataset
    da_dataset = DomainAdaptationDataset(src_dataset, tgt_dataset,
                                        randperm=cfg.experiment.seed+get_rank())
    # build loader
    da_loader = build_loader(cfg, da_dataset, split='train', 
                             collate_fn=da_collate_fn)
    
    # create validation dataset and loader
    val_dataset = build_dataset(cfg.domain.tgt, split='val')
    val_loader = build_loader(cfg, val_dataset, split='val')

    # build model
    model, criterion, postprocessors = build_model(cfg)
    model = model.to(device)
    criterion = criterion.to(device)

    # load checkpoint
    assert cfg.model.has('checkpoint')
    ckpt = torch.load(cfg.model.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt,
                                                        strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    # autoscale lr
    lr_scale = 1.0
    if cfg.optimizer.lr_auto_scale:
        base_batch_size   = 16
        actual_batch_size = cfg.loader.train.batch_size * cfg.world_size
        lr_scale = actual_batch_size / base_batch_size

    # param kw -> lr map
    param_group_map = {
        name : param_group['lr_mult'] for param_group in cfg.optimizer.params\
                  for name in param_group['names']
    }
    # default parameters
    param_dicts = [
        {"params" : [p for n, p in model.named_parameters()\
                     if not any(name in n for name in param_group_map.keys())],
        "lr": cfg.optimizer.lr_base * lr_scale},
    ]
    # named params with specific lr
    param_dicts += [ 
        {"params": [p for n, p in model.named_parameters()\
                     if name in n and p.requires_grad],
         "lr": cfg.optimizer.lr_base * lr_mult * lr_scale}\
         for name, lr_mult in param_group_map.items()
    ]
    # optimizer
    optimizer = torch.optim.AdamW(param_dicts, 
                                  lr=cfg.optimizer.lr_base * lr_scale,
                                  weight_decay=cfg.optimizer.weight_decay)
    # multistep lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        cfg.optimizer.lr_drop_steps,
                                                        cfg.optimizer.lr_drop_factor)

    # distributed model
    if cfg.distributed:
        model = nn.parallel.DistributedDataParallel(model, 
                                                device_ids=[cfg.gpu])
        model_without_ddp = model.module

    # eval for source only performance
    val_stats = evaluate_detection(
            model, criterion, postprocessors, val_loader, device, thresholds=cfg.evaluation.thresholds)
    print(val_stats)
                    
    # log on wandb
    if is_main_process():
        # log to wandb
        val_stats = {'val_'+k:v for k,v in val_stats.items()}
        wandb.log(val_stats)

    #curr_epoch, best_stats = 1, dict(f=dict(th04=dict(detection=dict(f1=0.0))))
    curr_epoch = 1
    # resume
    if cfg.experiment.has('resume') and cfg.experiment.resume:
        assert cfg.experiment.has('output_dir') and\
               cfg.experiment.has('output_name')

        ckpt = osp.join(cfg.experiment.output_dir, cfg.experiment.output_name)
        ckpt = torch.load(ckpt, strict=True, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        curr_epoch = ckpt.get('epoch', 1)

    # train loop
    for epoch in range(curr_epoch, cfg.optimizer.epochs+1):
        # set epoch in sampler
        if cfg.distributed:
            da_loader.sampler.set_epoch(epoch)

        # fit epoch
        train_stats = train_one_epoch(model, criterion, da_loader, optimizer, 
                        device=device, epoch=epoch, 
                        max_norm=cfg.optimizer.clip_max_norm)
        lr_scheduler.step()

        # create and save checkpoint
        if cfg.experiment.has("output_dir") and \
           cfg.experiment.has("output_name"):
            ckpt = {
                    'model' : model_without_ddp.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.state_dict(),
                    'epoch' : epoch,
                }    
            ckpt_path = osp.join(cfg.experiment.output_dir, 
                                f'{cfg.experiment.output_name}')
            save_on_master(ckpt, ckpt_path)

        # evaluate
        val_stats = dict()
        if (epoch in [1, cfg.optimizer.epochs] or epoch % cfg.evaluation.interval==0):
            #val_stats, _ = evaluate(
            #    model, criterion, postprocessors, val_loader, val_dataset.coco, device, cfg.experiment.output_dir)
            val_stats = evaluate_detection(
                model, criterion, postprocessors, val_loader, device, thresholds=cfg.evaluation.thresholds)
                    
        # log on wandb
        if cfg.experiment.wandb and is_main_process():
            # log to wandb
            train_stats = {'train_'+k:v for k,v in train_stats.items()}
            val_stats = {'val_'+k:v for k,v in val_stats.items()}
            wandb.log({**train_stats, **val_stats})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CellDETR')

    # config-file
    parser.add_argument('--config-file', type=str, default=None,
                        help='config file')
    # options
    # override config options with key=value pairs
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config_file is not None, "Please provide a config file."
    cfg = ConfigDict.from_file(args.config_file)
    opts = ConfigDict.from_options(args.opts)
    cfg.update(opts)
    train(cfg)