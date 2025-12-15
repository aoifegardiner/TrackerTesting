import sys
import os
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- Imports (same sources as working WAFT) --------------------
from MFT_WAFT.MFT.WAFT.model import fetch_model
from MFT_WAFT.MFT.WAFT.utils.utils import load_ckpt
from MFT_WAFT.MFT.WAFT.dataloader.loader import fetch_dataloader
from MFT_WAFT.MFT.WAFT.criterion.loss import sequence_loss   # supports occlusion + uncertainty
from MFT_WAFT.MFT.WAFT.utils.ddp_utils import *
import wandb

# -------------------------------------------------------------------------------
os.environ["KMP_INIT_AT_FORK"] = "FALSE"


class AverageMeter:
    """Keeps running average for metrics"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear'
    )
    return optimizer, scheduler


def train(args, rank=0, world_size=1, use_ddp=False):
    """ Full training loop with occlusion + uncertainty """
    device_id = rank
    model = fetch_model(args).to(device_id)

    if rank == 0:
        avg_loss = AverageMeter()
        avg_epe = AverageMeter()
        avg_occl = AverageMeter()
        avg_uncert = AverageMeter()
        wandb.init(project=args.name)

    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"Restored checkpoint from {args.restore_ckpt}")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    model.train()

    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size, use_ddp=use_ddp)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    VAL_FREQ = 10000
    epoch = 0
    should_keep_training = True

    while should_keep_training:
        train_loader.sampler.set_epoch(epoch)
        epoch += 1

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            # Unpack batch (support occlusion if available)
            if len(data_blob) == 5:
                image1, image2, flow, valid, occl = [x.cuda(non_blocking=True) for x in data_blob]
            else:
                image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
                occl = None

            # Forward pass
            output = model(image1, image2, flow_gt=flow)

            # Compute combined loss
            loss, metrics = sequence_loss(output, flow, valid, occl_gt=occl, gamma=args.gamma, args=args)

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            # Logging
            if rank == 0:
                avg_loss.update(loss.item())
                avg_epe.update(metrics.get("train/epe", 0))
                avg_occl.update(metrics.get("train/cross_entropy_occl", 0))
                avg_uncert.update(metrics.get("train/uncert", 0))

                if total_steps % 100 == 0:
                    wandb.log({
                        "loss": avg_loss.avg,
                        "epe": avg_epe.avg,
                        "occlusion_loss": avg_occl.avg,
                        "uncertainty_loss": avg_uncert.avg
                    })
                    avg_loss.reset()
                    avg_epe.reset()
                    avg_occl.reset()
                    avg_uncert.reset()

            # Save checkpoint
            if total_steps % VAL_FREQ == VAL_FREQ - 1 and rank == 0:
                PATH = f'checkpoints/{total_steps+1}_{args.name}.pth'
                torch.save(model.module.state_dict(), PATH)

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    PATH = f'checkpoints/{args.name}.pth'
    if rank == 0:
        torch.save(model.module.state_dict(), PATH)
        wandb.finish()

    return PATH


def main(rank, world_size, args, use_ddp):
    if use_ddp:
        print(f"Using DDP [{rank=} {world_size=}]")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        setup_ddp(rank, world_size)

    train(args, rank=rank, world_size=world_size, use_ddp=use_ddp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment config file', required=True, type=str)
    parser.add_argument('--seed', help='seed', default=42, type=int)
    parser.add_argument('--restore_ckpt', help='restore checkpoint', default=None, type=str)
    args = parse_args(parser)

    smp, world_size = init_ddp()
    if world_size > 1:
        spwn_ctx = mp.spawn(main, nprocs=world_size, args=(world_size, args, True), join=False)
        spwn_ctx.join()
    else:
        main(0, 1, args, False)
    print("Done!")
