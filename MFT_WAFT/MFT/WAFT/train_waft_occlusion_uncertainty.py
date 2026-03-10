from __future__ import print_function, division
from asyncio.log import logger
import sys
from xml.parsers.expat import model
sys.path.append('core')
#from ipdb import iex

import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim


from MFT.waft import FlowOUTrackingResult
from MFT_WAFT.MFT.WAFT import evaluate
import MFT_WAFT.MFT.RAFT.core.datasets as datasets
from MFT_WAFT.MFT.WAFT.utils.utils import load_ckpt
from MFT_WAFT.MFT.WAFT.criterion.loss import sequence_loss

from torch.utils.tensorboard import SummaryWriter

from MFT.RAFT.core.utils.timer import Timer

from MFT.RAFT.core.utils.flow_viz import flow_to_color
from MFT_WAFT.MFT.WAFT.model import fetch_model
from MFT_WAFT.MFT.WAFT.model.waft_ou import WAFT_OU_FromFlow



try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 5 # 100
VAL_FREQ = 2000 # 5000


def sequence_loss(preds_dict, flow_gt, valid, occl_gt=None, gamma=0.8, max_flow=MAX_FLOW, args=None, **kwargs):

    alpha_flow = kwargs.get('alpha_flow', 1.0)
    alpha_occl = kwargs.get('alpha_occl', 5.0)
    alpha_uncertainty = kwargs.get('alpha_uncertainty', 1.0)

    uncertainty_loss_type = args.uncertainty_loss
    weighting_unc_loss = args.weighting_unc_loss
    flow_loss_type = args.optical_flow_loss

    total_loss = 0.0
    metrics = {}

    # Normalize prediction lists
    def ensure_list(x):
        if isinstance(x, torch.Tensor):
            return [x[i:i+1] for i in range(x.shape[0])]
        return x
    
    flow_preds = ensure_list(preds_dict['flow'])
    occl_preds = ensure_list(preds_dict.get('occlusion', []))
    uncertainty_preds = ensure_list(preds_dict.get('uncertainty', []))

    # store back normalized versions so downstream calls use them
    preds_dict['flow'] = flow_preds
    preds_dict['occlusion'] = occl_preds
    preds_dict['uncertainty'] = uncertainty_preds

    flow_loss_val = None
    occl_loss_val = None
    uncertainty_loss_val = None

    #print(f"Flow preds: {len(flow_preds)}, Occl preds: {len(occl_preds)}, Unc preds: {len(uncertainty_preds)}")
    #print(f"Flow[0]: {flow_preds[0].shape}")

    if not args.freeze_optical_flow_training:
        flow_loss, flow_metrics = sequence_flow_loss(flow_preds, flow_gt, valid, occl_gt=occl_gt,
                                                     gamma=gamma, max_flow=max_flow, flow_loss_type=flow_loss_type)
        metrics.update(flow_metrics)
        total_loss += (alpha_flow * flow_loss)
        flow_loss_val = flow_loss.detach().item()
        metrics['train/loss_flow'] = flow_loss_val
    else:
        # still log flow consistency (EPE) even when frozen ---
        with torch.no_grad():
            mag = torch.sum(flow_gt**2, dim=1).sqrt()
            v = (valid[:,0,:,:] >= 0.5) & (mag < max_flow)
            epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)[v.view(-1)]
            if epe.numel() > 0:
                metrics['train/epe_frozen'] = epe.mean().item()
            # compute flow loss value for monitoring only
            flow_loss_val, _ = sequence_flow_loss(
                flow_preds, flow_gt, valid, occl_gt=occl_gt,
                gamma=gamma, max_flow=max_flow, flow_loss_type=flow_loss_type
            )
            metrics['train/loss_flow_frozen'] = flow_loss_val.item()



    if args.occlusion_module is not None:
        #occl_preds = ensure_list(preds_dict.get('occlusion', []))
        occl_loss, occl_metrics = sequence_occl_loss(occl_preds, occl_gt, flow_gt, valid, gamma=gamma, max_flow=max_flow)
        metrics.update(occl_metrics)
        total_loss += (alpha_occl * occl_loss)
        occl_loss_val = occl_loss.detach().item()
        metrics['train/loss_occl'] = occl_loss_val

    if args.occlusion_module is not None and 'uncertainty' in args.occlusion_module:
        #uncertainty_preds = ensure_list(preds_dict.get('uncertainty', []))
        uncertainty_loss, uncertainty_metrics = sequence_uncertainty_loss(flow_preds, uncertainty_preds,
                                                                          flow_gt, valid, gamma=gamma,
                                                                          max_flow=max_flow,
                                                                          uncertainty_loss_type=uncertainty_loss_type,
                                                                          weighting_unc_loss=weighting_unc_loss,
                                                                          occl_gt=occl_gt)
        metrics.update(uncertainty_metrics)
        total_loss += (alpha_uncertainty * uncertainty_loss)
        uncertainty_loss_val = uncertainty_loss.detach().item()
        metrics['train/loss_uncertainty'] = uncertainty_loss_val

    #print(f"Flow preds: {len(flow_preds)}, Occl preds: {len(occl_preds)}, Unc preds: {len(uncertainty_preds)}")
    #print(f"Flow[0]: {flow_preds[0].shape}")
    metrics['train/loss_total'] = total_loss.detach().item()
    return total_loss, metrics

def sequence_occl_loss(occl_preds, occl_gt, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(occl_preds)
    occl_loss = 0.0

    cross_ent_loss = nn.CrossEntropyLoss(reduction='none')

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid[:,0,:,:] >= 0.5) & (mag < max_flow)
    # only 100% occluded and 100% non-occluded are used for training
    occl_valid = torch.logical_or(occl_gt < 0.01, occl_gt > 0.99)
    valid = torch.logical_and(occl_valid[:,0,:,:], valid)

    occl_gt_thresholded = occl_gt > 0.5
    occl_gt_thresholded = occl_gt_thresholded[:,0,:,:].long()

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        #i_loss = cross_ent_loss(occl_preds[i].softmax(dim=1), occl_gt_thresholded)
        i_loss = cross_ent_loss(occl_preds[i], occl_gt_thresholded)
        occl_loss += i_weight * (valid[:, None] * i_loss).mean()
    
    # inside sequence_occl_loss, before cross_ent_loss
    # estimate pos/neg on-the-fly on the masked pixels (cheap)
    with torch.no_grad():
        pos = ((occl_gt_thresholded == 1) & valid).sum().float()
        neg = ((occl_gt_thresholded == 0) & valid).sum().float()
        w_pos = neg / (pos + 1e-9)
        w_neg = pos / (neg + 1e-9)
        ce_w = torch.tensor([w_neg, w_pos], device=occl_preds[0].device)
    cross_ent_loss = nn.CrossEntropyLoss(weight=ce_w, reduction='none')


    metrics = {
        'train/cross_entropy_occl': i_loss.mean().item(),
    }

    return occl_loss, metrics


def sequence_flow_loss(flow_preds, flow_gt, valid, occl_gt=None, gamma=0.8, max_flow=MAX_FLOW, flow_loss_type='L1'):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid[:,0,:,:] >= 0.5) & (mag < max_flow)
    if 'occl' in flow_loss_type:
        assert occl_gt is not None
        hard_occl_mask = torch.squeeze(occl_gt[:,0,:,:], dim=1) > 0.99

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()

        if flow_loss_type == 'L1':
            i_valid = valid
        elif flow_loss_type == 'L1_non_occluded':
            i_valid = torch.logical_and(valid, torch.logical_not(hard_occl_mask))
        elif flow_loss_type == 'L1_occluded_to_epe3':
            flow_epe = torch.sqrt(torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1, keepdim=False)).detach()
            epe_mask = flow_epe < 3.0
            nonoccl_or_epe_mask = torch.logical_or(torch.logical_not(hard_occl_mask), epe_mask)
            i_valid = torch.logical_and(valid, nonoccl_or_epe_mask)
        else:
            raise NotImplementedError(f'Flow loss type {flow_loss_type} not implemented')
        flow_loss += i_weight * (i_valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train/epe': epe.mean().item(),
        'train/1px': (epe < 1).float().mean().item(),
        'train/3px': (epe < 3).float().mean().item(),
        'train/5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def weights_uncertainty_according_epe(epe):
    device = epe.device
    coef = np.array([-7.27864588e-02,  9.00020608e+00, -1.79078330e+01,  8.68281513e+01])
    epe_clamped = torch.clamp(epe, 0, 50).detach()
    epe2 = epe_clamped**2
    epe3 = epe_clamped**3

    weight = epe3 * coef[0] + epe2 * coef[1] + epe_clamped * coef[2] + coef[3]
    weight = weight / 50
    return weight

def sequence_uncertainty_loss(flow_preds, uncertainty_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW,
                              uncertainty_loss_type='huber', weighting_unc_loss=False, occl_gt=None):
    """
    InProceedings (He2019)
    He, Y.; Zhu, C.; Wang, J.; Savvides, M. & Zhang, X.
    Bounding box regression with uncertainty for accurate object detection
    Proceedings of the ieee/cvf conference on computer vision and pattern recognition, 2019, 2888-2897

    Loss from equation 9 and 10
    Loss is weighted (i_weight) in the same way as in the RAFT
    """

    if uncertainty_loss_type in ['huber', 'huber_non_occluded']:
        unc_loss = torch.nn.SmoothL1Loss(reduction='none')
    elif uncertainty_loss_type in ['L2', 'L2_non_occluded']:
        unc_loss = torch.nn.MSELoss(reduction='none')
    elif uncertainty_loss_type == 'huber_epe_direct':
        unc_loss = torch.nn.SmoothL1Loss(reduction='none')
    elif uncertainty_loss_type == 'huber_epe_direct_non_occluded':
        unc_loss = torch.nn.SmoothL1Loss(reduction='none')
    else:
        raise NotImplementedError('This type of loss is not implemented for uncertainty')

    #n_predictions = len(flow_preds)
    n_predictions = min(len(flow_preds), len(uncertainty_preds))

    uncertainty_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid[:,0,:,:] >= 0.5) & (mag < max_flow)

    if uncertainty_loss_type in ['huber', 'L2', 'huber_non_occluded', 'L2_non_occluded']:
        for i in range(n_predictions):
            i_weight = gamma**(n_predictions - i - 1)

            i_alpha = uncertainty_preds[i]
            i_loss_exp_alpha = torch.exp(-i_alpha)
            if uncertainty_loss_type == 'L2':
                i_loss_exp_alpha = 0.5 * i_loss_exp_alpha

            flow_epe = torch.sqrt(torch.sum((flow_preds[i] - flow_gt)**2, dim=1, keepdim=True)).detach()

            unc_loss_comp = unc_loss(flow_epe, torch.zeros_like(flow_epe))
            i_loss = i_loss_exp_alpha * unc_loss_comp + 0.5 * i_alpha

            if 'non_occluded' in uncertainty_loss_type:
                valid = torch.logical_and(valid, torch.logical_not(torch.squeeze(occl_gt, dim=1) > 0.99))
            if weighting_unc_loss:
                i_loss = weights_uncertainty_according_epe(unc_loss_comp) * i_loss
            uncertainty_loss += i_weight * (valid[:, None] * i_loss).mean()
    elif uncertainty_loss_type in ['huber_epe_direct', 'huber_epe_direct_non_occluded']:
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)

            i_alpha = uncertainty_preds[i]
            i_exp_alpha = torch.exp(-i_alpha)
            flow_L2 = torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1, keepdim=True).detach()
            flow_epe = torch.sqrt(flow_L2)

            i_comp_for_alpha = - i_alpha * i_exp_alpha

            i_loss = unc_loss(i_comp_for_alpha, flow_L2)

            if 'non_occluded' in uncertainty_loss_type:
                valid = torch.logical_and(valid, torch.logical_not(torch.squeeze(occl_gt, dim=1) > 0.99))
            if weighting_unc_loss:
                i_loss = weights_uncertainty_according_epe(flow_epe) * i_loss
            uncertainty_loss += i_weight * (valid[:, None] * i_loss).mean()
    else:
        raise NotImplementedError('This type of loss is not implemented for uncertainty')

    metrics = {
        'train/uncert': i_loss.mean().item(),
    }

    return uncertainty_loss, metrics

import torch
import torch.nn.functional as F

@torch.no_grad()
def occlusion_metrics(occ_logits: torch.Tensor, occl_gt: torch.Tensor, valid: torch.Tensor, thr: float = 0.5):
    """
    occ_logits: [B,2,H,W]
    occl_gt:    [B,1,H,W] float in [0,1]
    valid:      [B,1,H,W] float/bool
    Returns dict of scalar metrics.
    """
    # masks
    mask = (valid[:, 0] > 0.5)

    # GT labels (0/1)
    gt = (occl_gt[:, 0] > thr)

    # predicted prob + label
    prob_occ = F.softmax(occ_logits, dim=1)[:, 1]   # [B,H,W]
    pred = (prob_occ > thr)

    # only evaluate where mask is true
    gt_m = gt[mask]
    pred_m = pred[mask]
    prob_m = prob_occ[mask]

    if gt_m.numel() == 0:
        return {"train_diag/occl_mask_frac": 0.0}

    tp = (pred_m & gt_m).sum().item()
    tn = ((~pred_m) & (~gt_m)).sum().item()
    fp = (pred_m & (~gt_m)).sum().item()
    fn = ((~pred_m) & gt_m).sum().item()

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "train_diag/occl_acc": acc,
        "train_diag/occl_prec": prec,
        "train_diag/occl_rec": rec,
        "train_diag/occl_f1": f1,
        "train_diag/occl_pos_rate_gt": gt_m.float().mean().item(),
        "train_diag/occl_pos_rate_pred": pred_m.float().mean().item(),
        "train_diag/occl_prob_mean": prob_m.mean().item(),
        "train_diag/occl_prob_std": prob_m.std(unbiased=False).item(),
        "train_diag/occl_mask_frac": mask.float().mean().item(),
    }


@torch.no_grad()
def uncertainty_metrics(log_sigma2: torch.Tensor, flow_pred: torch.Tensor, flow_gt: torch.Tensor, valid: torch.Tensor):
    """
    log_sigma2: [B,1,H,W] (clamped)
    flow_pred:  [B,2,H,W]
    flow_gt:    [B,2,H,W]
    valid:      [B,1,H,W]
    """
    mask = (valid[:, 0] > 0.5)

    # squared error magnitude
    epe2 = ((flow_pred - flow_gt) ** 2).sum(dim=1)  # [B,H,W]

    # sigma2
    sigma2 = torch.exp(log_sigma2[:, 0])            # [B,H,W]

    epe2_m = epe2[mask]
    sigma2_m = sigma2[mask]
    log_s_m = log_sigma2[:, 0][mask]

    if epe2_m.numel() == 0:
        return {"train_diag/unc_mask_frac": 0.0}

    # quick calibration-ish stats: do they track each other at all?
    # (Pearson correlation)
    e = epe2_m.float()
    s = sigma2_m.float()
    e = e - e.mean()
    s = s - s.mean()
    corr = (e * s).mean() / (e.std(unbiased=False) * s.std(unbiased=False) + 1e-9)

    return {
        "train_diag/epe2_mean": epe2_m.mean().item(),
        "train_diag/sigma2_mean": sigma2_m.mean().item(),
        "train_diag/log_sigma2_mean": log_s_m.mean().item(),
        "train_diag/log_sigma2_min": log_s_m.min().item(),
        "train_diag/log_sigma2_max": log_s_m.max().item(),
        "train_diag/epe2_sigma2_corr": corr.item(),
        "train_diag/unc_mask_frac": mask.float().mean().item(),
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    #optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #    pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, logfile_comment=''):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.logfile_comment = logfile_comment

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(comment=self.logfile_comment)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def create_writer_if_not_set(self):
        if self.writer is None:
            self.writer = SummaryWriter(comment=self.logfile_comment)

    def write_dict(self, results):
        self.create_writer_if_not_set()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def write_image_dict(self, results):
        self.create_writer_if_not_set()

        for key in results:
            self.writer.add_image(key, results[key], self.total_steps)

    def write_images(self, inputs):
        self.create_writer_if_not_set()

        for key in inputs:
            im = inputs[key]

            # grid = torchvision.utils.make_grid(im)
            # self.writer.add_image(key, grid, self.total_steps)
            if key == 'valid':
                data = im.type(torch.uint8) * 255
                # data = torch.unsqueeze(data, 1) * 255
                self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
            elif 'occl' in key:
                data = torch.clamp(im * 255., 0., 255.)
                data = data.type(torch.uint8)
                self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
            elif 'sigma' in key:
                data = torch.clamp(im, 0., 255.)
                data = data.type(torch.uint8)
                self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
            elif 'flow' in key:
                # If this is a real flow field, im is [B,2,H,W]
                if im.dim() == 4 and im.shape[1] == 2:
                    data = im.detach().cpu().numpy().transpose(0,2,3,1)  # [B,H,W,2]
                    color_list = [flow_to_color(data[i]) for i in range(data.shape[0])]
                    color_image = np.stack(color_list, axis=0)           # [B,H,W,3]
                    self.writer.add_images(key, color_image, dataformats='NHWC', global_step=self.total_steps)
                else:
                    # Otherwise treat it as a generic 1-channel map in 0..255
                    data = torch.clamp(im, 0., 255.).to(torch.uint8)
                    self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
        
            else:
                self.writer.add_images(key, im.type(torch.uint8), dataformats='NCHW', global_step=self.total_steps)

    def close(self):
        self.writer.close()

def weight_freezer(model, args):
    if args.freeze_optical_flow_training or args.freeze_features_training:
        model.eval()
        #with torch.no_grad():
        #    out = model(image1, image2, iters=12, test_mode=True)
        #print([k for k in out.keys()])
        #for k, v in out.items():
        #    print(k, v[-1].shape if isinstance(v, list) else v.shape)

        model.requires_grad_(False)
        if not args.freeze_optical_flow_training:
            raise NotImplementedError('Have to be specified')
        if not args.freeze_features_training:
            raise NotImplementedError('Have to be specified')
        #print("DEBUG WAFT_OU attrs:", dir(model.module))

        model.module.occlusion_head.requires_grad_(True)
        model.module.occlusion_head.train()
        model.module.uncertainty_head.requires_grad_(True)
        model.module.uncertainty_head.train()
        
    else:
        model.train()

    if hasattr(model.module, 'freeze_bn') and args.stage != 'chairs':
        model.module.freeze_bn()

    return model



#@iex
def train(args):

    train_timer = Timer()

    os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join([str(gpu_n) for gpu_n in args.gpus])

    args.gpus = range(len(args.gpus))

    import torch
    print("visible:", torch.cuda.device_count())
    print("current:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

    #model = fetch_model(args)
    #if args.restore_ckpt:
    #    load_ckpt(model, args.restore_ckpt)

    #model = nn.DataParallel(model, device_ids=args.gpus)#.cuda()

    #print("Parameter Count: %d" % count_parameters(model))

    #if args.restore_ckpt is not None:
    #    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # after you build/load your pretrained WAFT model:
    model  = fetch_model(args)
    #print("fetch_model returned:", type(model))
    #print("has flow_model:", hasattr(model, "flow_model"))

    load_ckpt(model, args.restore_ckpt)
    sd = torch.load(args.restore_ckpt, map_location="cpu")
    sd = sd["state_dict"] if "state_dict" in sd else sd

    # If the checkpoint was saved from a plain WAFT model,
    # keys probably start with "da_feature...", "flow_head...", etc.
    # If it was saved from a wrapper, keys may include "flow_model." prefixes.

    # Normalize key prefixes to match model.flow_model
    new_sd = {}
    for k,v in sd.items():
        k = k.replace("module.", "")
        # if ckpt has "flow_model.flow_model.", strip one
        if k.startswith("flow_model.flow_model."):
            k = k.replace("flow_model.flow_model.", "", 1)   # <-- note: remove BOTH prefixes for loading into flow_model
        # if ckpt has "flow_model.", strip it for loading into flow_model
        if k.startswith("flow_model."):
            k = k.replace("flow_model.", "", 1)
        new_sd[k] = v

    missing, unexpected = model.flow_model.load_state_dict(new_sd, strict=False)
    

    
    # freeze flow model if you don't want to retrain it
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    #model = WAFT_OU_FromFlow(flow_model, args, use_rgb_residual=False, detach_flow_features=True)

    # keep heads trainable
    model.trunk.requires_grad_(True)
    model.occlusion_head.requires_grad_(True)
    model.uncertainty_head.requires_grad_(True)

    model.train()  # sets modules to train mode; flow_model stays eval as set above

    model.cuda()
    #model = weight_freezer(model, args)

    train_loader = datasets.fetch_dataloader(args)
    #trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    #print("num trainable params:", len(trainable))
    #print("examples:", trainable[:10])
    
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, logfile_comment=args.name)
    logger.create_writer_if_not_set()

    #n_trainable_backbone = sum(p.requires_grad for p in model.flow_model.parameters())
    #print("trainable backbone params:", n_trainable_backbone)


    should_keep_training = True
    print('Training...', train_timer.iter(), train_timer())
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            # print(i_batch, total_steps, train_timer.iter(), train_timer())
            if data_blob is None:
                continue  # all samples in this batch were corrupt
            
            optimizer.zero_grad()
            image1, image2, flow, valid, occl = [x.cuda() for x in data_blob]
            ##DEBUG

            mag_gt = torch.linalg.norm(flow, dim=1)
            print("GT mag mean/max:", mag_gt.mean().item(), mag_gt.max().item())


            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            from torch.cuda.amp import autocast

            with autocast(enabled=args.mixed_precision):
                all_predictions = model(image1, image2, iters=args.iters)
                loss, metrics = sequence_loss(all_predictions, flow, valid, occl_gt=occl, gamma=args.gamma, args=args)


            with torch.no_grad():
                fp = all_predictions["flow"][-1]          # [B,2,H,W]
                fg = flow                                # [B,2,H,W]

                print("shapes pred/gt:", fp.shape, fg.shape)

                mag_fp = torch.sqrt((fp**2).sum(dim=1))   # [B,H,W]
                mag_fg = torch.sqrt((fg**2).sum(dim=1))   # [B,H,W]
                m = (valid[:,0] > 0.5)
                scale = (mag_fp[m].mean() / (mag_fg[m].mean() + 1e-9)).item()
                #print("GT flow mean:", fg.abs().mean().item())
                #print("Pred flow mean:", fp.abs().mean().item())
                print("pred_to_gt_mag ratio:", scale)
                #flow = flow * scale
                #[print("scaled flow mean:", flow.abs().mean().item())]


            #all_predictions = model(image1, image2, iters=args.iters)
            if total_steps < 5:
                def stats(name, x):
                    if isinstance(x, list): x = x[-1]
                    print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
                          f"min={x.min().item():.4g} max={x.max().item():.4g} "
                          f"nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()}")

                stats("flow_est", all_predictions["flow"])
                if "occlusion" in all_predictions:
                    stats("occl_logits", all_predictions["occlusion"])
                if "uncertainty" in all_predictions:
                    stats("uncert", all_predictions["uncertainty"])

                H, W = image1.shape[-2:]
                print("image HW:", (H, W))
                for k in ["flow", "occlusion", "uncertainty"]:
                    if k in all_predictions:
                        t = all_predictions[k][-1] if isinstance(all_predictions[k], list) else all_predictions[k]
                        print(k, "HW:", tuple(t.shape[-2:]))



            #loss, metrics = sequence_loss(all_predictions, flow, valid, occl_gt=occl, gamma=args.gamma, args=args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            if total_steps % 50 == 0:  # cheap, frequent sanity check
                with torch.no_grad():
                    if (args.occlusion_module is not None) and ("occlusion" in all_predictions) and (occl is not None):
                        occ_logits = all_predictions["occlusion"][-1]
                        logger.write_dict(occlusion_metrics(occ_logits, occl, valid))
                with torch.no_grad():
                    if (args.occlusion_module is not None) and ("uncertainty" in all_predictions):
                        log_sigma2 = all_predictions["uncertainty"][-1]
                        flow_pred = all_predictions["flow"][-1]
                        logger.write_dict(uncertainty_metrics(log_sigma2, flow_pred, flow, valid))
                g = 0.0
                for p in model.flow_model.parameters():
                    if p.grad is not None:
                        g = max(g, p.grad.abs().max().item())
                logger.write_dict({"debug/flow_backbone_grad_max": g})

                with torch.no_grad():
                    fp = all_predictions["flow"][-1]
                    mag_gt = torch.sqrt((flow**2).sum(1))
                    mag_fp = torch.sqrt((fp**2).sum(1))
                    m = (valid[:,0] > 0.5) & (mag_gt < MAX_FLOW)
                    if m.any():
                        logger.write_dict({
                            "debug/gt_mag_mean": mag_gt[m].mean().item(),
                            "debug/pred_mag_mean": mag_fp[m].mean().item(),
                            "debug/pred_to_gt_mag": (mag_fp[m].mean() / (mag_gt[m].mean() + 1e-9)).item(),
                        })
                


            if (total_steps % VAL_FREQ == VAL_FREQ - 1):#(total_steps == 7) or 

                del all_predictions, loss
                torch.cuda.empty_cache()

                print('validation ', i_batch, total_steps, train_timer.iter(), train_timer())
                PATH = args.checkpoints + '/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                # print('before val: ', train_timer.iter(), train_timer())
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model))#.module))
                        # print('val: ', train_timer.iter(), train_timer())
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(args, model))#.module))
                    elif val_dataset == 'sintel_val_subsplit':
                        results.update(evaluate.validate_sintel(model, subsplit='validation'))
                        # print('val: ', train_timer.iter(), train_timer())
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model))#.module))
                        # print('val: ', train_timer.iter(), train_timer())
                print('after val: ', total_steps, train_timer.iter(), train_timer())

                logger.write_dict(results)
                
                #train_diag = {}
                #if (args.occlusion_module is not None) and ("occlusion" in all_predictions) and (occl is not None):
                #    m = occlusion_metrics(all_predictions["occlusion"][-1], occl, valid)
                #    train_diag.update({f"train_diag/{k.split('/',1)[1] if '/' in k else k}": v for k, v in m.items()})

                #if (args.occlusion_module is not None) and ("uncertainty" in all_predictions):
                #    m = uncertainty_metrics(all_predictions["uncertainty"][-1], all_predictions["flow"][-1], flow, valid)
                #    train_diag.update({f"train_diag/{k.split('/',1)[1] if '/' in k else k}": v for k, v in m.items()})

                #if train_diag:
                #    logger.write_dict(train_diag)

                #flow_pred = all_predictions["flow"][-1]
                #print("flow slice:", flow_pred[0, :, 100:110, 100:110])


            logger.write_dict({
              "debug/occl_gt_mean": occl.mean().item(),
              "debug/occl_gt_pos_rate": (occl > 0.5).float().mean().item(),
              "debug/loss_total_recon": 5.0*metrics.get("train/loss_occl", 0.0) + metrics.get("train/loss_uncertainty", 0.0),
            })
            

            import torch.nn.functional as F

            H, W = image1.shape[-2:]

            LOG_IMG_FREQ = 200
            if total_steps % LOG_IMG_FREQ == 0:

                # --- Always log inputs and GT ---
                logger.write_images({'image1': image1, 'image2': image2, 'valid': valid})
                logger.write_images({'flow_gt': flow})

                # --- Flow estimate + magnitude for debugging ---
                flow_est = all_predictions['flow'][-1]
                logger.write_images({'flow_est': flow_est})

                flow_mag = torch.sqrt(torch.sum(flow_est**2, dim=1, keepdim=True))  # [B,1,H,W]
                mn = flow_mag.amin(dim=[2,3], keepdim=True)
                mx = flow_mag.amax(dim=[2,3], keepdim=True)
                flow_mag_vis = (flow_mag - mn) / (mx - mn + 1e-9)
                logger.write_images({'flow_mag_minmax': flow_mag_vis * 255})

                # --- Occlusion GT (upsample if needed) ---
                if occl is not None:
                    occl_vis = occl.float()
                    if occl_vis.shape[-2:] != (H, W):
                        occl_vis = F.interpolate(occl_vis, size=(H, W), mode="nearest")
                    # IMPORTANT: occl keys are auto *255 in Logger, so keep 0..1 here
                    logger.write_images({'occl_gt': occl_vis})

                # --- Occlusion prediction (probabilities in 0..1) ---
                if args.occlusion_module is not None and 'occlusion' in all_predictions:
                    occl_logits = all_predictions['occlusion'][-1]          # [B,2,h,w] or [B,2,H,W]
                    if occl_logits.shape[-2:] != (H, W):
                        occl_logits = F.interpolate(occl_logits, size=(H, W), mode="bilinear", align_corners=False)

                    occl_prob = occl_logits.softmax(dim=1)
                    # IMPORTANT: Logger multiplies by 255 for keys containing 'occl'
                    logger.write_images({'occl_est_neg': occl_prob[:, 0:1]})
                    logger.write_images({'occl_est_pos': occl_prob[:, 1:2]})

                    # Debug scalar to confirm it's not constant 0.5 everywhere
                    logger.write_dict({
                        'debug/occl_pos_mean': occl_prob[:,1:2].mean().item(),
                        'debug/occl_pos_std':  occl_prob[:,1:2].std().item(),
                    })

                with torch.no_grad():
                    mag = torch.sum(flow**2, dim=1).sqrt()
                    v = (valid[:,0] >= 0.5) & (mag < MAX_FLOW)
                    occl_valid = (occl < 0.01) | (occl > 0.99)
                    m = v & occl_valid[:,0]
                    logger.write_dict({
                        "debug/occl_train_mask_frac": m.float().mean().item(),
                        "debug/occl_train_pos_rate_gt": ( (occl[:,0] > 0.5) & m ).float().sum().item() / (m.float().sum().item() + 1e-9),
                    })

                with torch.no_grad():
                    a = all_predictions["uncertainty"][-1]
                    logger.write_dict({
                        "debug/unc_frac_at_max": (a >= args.var_max - 1e-4).float().mean().item(),
                        "debug/unc_frac_at_min": (a <= args.var_min + 1e-4).float().mean().item(),
                    })
                



                # --- Uncertainty prediction (visualize min-max) ---
                if (args.occlusion_module is not None) and ('uncertainty' in args.occlusion_module) and ('uncertainty' in all_predictions):
                    unc = all_predictions['uncertainty'][-1]                # [B,1,h,w] or [B,1,H,W]
                    if unc.shape[-2:] != (H, W):
                        unc = F.interpolate(unc, size=(H, W), mode="bilinear", align_corners=False)

                    sigma2 = torch.exp(unc)

                    # Per-image min-max normalization for display
                    mn = sigma2.amin(dim=[2,3], keepdim=True)
                    mx = sigma2.amax(dim=[2,3], keepdim=True)
                    sigma2_minmax = (sigma2 - mn) / (mx - mn + 1e-9)

                    # 'sigma' keys are clamped but NOT scaled inside Logger, so pass 0..255 here
                    logger.write_images({'sigma2_est_minmax': sigma2_minmax * 255})

                    logger.write_dict({
                        'debug/unc_mean': unc.mean().item(),
                        'debug/unc_std':  unc.std().item(),
                    })

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = f'{args.checkpoints}/{args.name}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAFT PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--name', default='waft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--algorithm', type=str, default='waft', help="which flow algorithm to use (e.g. waft, vitwarp, raft)")
    parser.add_argument('--dav2_backbone', default="vits")
    parser.add_argument('--network_backbone', default="vits")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--occlusion_module', type=str, default=None,
                        choices=[None, 'separate', 'with_uncertainty', 'separate_with_uncertainty',
                                 'separate_with_uncertainty_upsample8',
                                 'separate_with_uncertainty_morelayers',
                                 'separate_with_uncertainty_upsample8_morelayers'])
    parser.add_argument('--freeze_optical_flow_training', action='store_true', help='freezes training of optical flow estimation module')
    parser.add_argument('--freeze_features_training', action='store_true', help='freezes training of image features')
    parser.add_argument('--uncertainty_loss', type=str, default='huber',
                        choices=['huber', 'L2', 'huber_epe_direct',
                                 'huber_epe_direct_non_occluded', 'huber_non_occluded', 'L2_non_occluded'])
    parser.add_argument('--optical_flow_loss', type=str, default='L1',
                        choices=['L1', 'L1_non_occluded', 'L1_occluded_to_epe3'])

    parser.add_argument('--weighting_unc_loss', action='store_true', help='reweigting unc loss according epe sintel distribution')
    parser.add_argument('--validation', type=str, nargs='+', default=['sintel'])

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--var_max', type=float, default=12.0)
    parser.add_argument('--var_min', type=float, default=-12.0)

    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dashcam_augmenentation', action='store_true')
    parser.add_argument('--blend_source', default='/datagrid/public_datasets/COCO/train2017', help="path to blending images")
    parser.add_argument('--normalized_features', help='normalize features before costvolume', action='store_true')
    parser.add_argument('--seed', help='', type=int, default=1234)
    parser.add_argument('--checkpoints', help='checkpoint directory', default='checkpoints')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = f'@{sys.argv[1]}'
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.isdir(args.checkpoints):
        os.mkdir(args.checkpoints)

    train(args)
