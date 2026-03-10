import math
import numpy as np
import torch
import torch.utils.data as data
from torch import nn
from torch.autograd import no_grad
from tqdm import tqdm

# Reuse RAFT datasets + padder for consistency
from MFT_WAFT.MFT.RAFT.core import datasets
from MFT_WAFT.MFT.RAFT.core.utils.utils import InputPadder

# --- small running-average helpers (like RAFT) ---
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.n = 0
    def update(self, val, cnt=1): self.sum += float(val) * cnt; self.n += cnt
    @property
    def avg(self): return self.sum / max(self.n, 1)

val_epe  = AverageMeter()
val_px1  = AverageMeter()
val_px3  = AverageMeter()
val_px5  = AverageMeter()
val_fl   = AverageMeter()   # KITTI-style outlier
val_unc  = AverageMeter()   # optional: mean uncertainty
val_occ  = AverageMeter()   # optional: mean occlusion

def _extract_ou_from_output(out_dict):
    """
    Extracts (unc, occ) tensors from WAFT-style outputs.
    Handles:
      - explicit heads: out_dict['uncertainty'] / out_dict['occlusion'] (lists)
      - packed 'info': out_dict['info'] (lists of BxC×HxW, e.g. C=4 [unc, 0, 0, occ])
    """
    unc, occ = None, None

    # --- Prefer explicit heads if available ---
    if 'uncertainty' in out_dict and isinstance(out_dict['uncertainty'], list) and len(out_dict['uncertainty']) > 0:
        unc = out_dict['uncertainty'][-1]  # Bx1xHxW
    if 'occlusion' in out_dict and isinstance(out_dict['occlusion'], list) and len(out_dict['occlusion']) > 0:
        occ = out_dict['occlusion'][-1]    # Bx2xHxW logits

    # --- Fallback to packed info tensor ---
    if (unc is None or occ is None) and ('info' in out_dict) and len(out_dict['info']) > 0:
        info = out_dict['info'][-1]  # BxCxHxW
        C = info.shape[1]
        if unc is None and C >= 1:
            unc = info[:, 0:1, ...]      # channel 0 → uncertainty
        if occ is None:
            if C == 4:
                occ = info[:, 3:4, ...]  # WAFT_OU 4-channel layout
            elif C >= 2:
                occ = info[:, 1:2, ...]  # legacy 2-channel case

    return unc, occ


def _occl_accuracy_from_logits(occl_logits, occl_gt):
    """
    occl_logits: Bx2xHxW (class 1 = occluded)
    occl_gt:     Bx1xHxW (0/1)
    """
    probs = occl_logits.softmax(dim=1)[:, 1:2, ...]  # P(occluded)
    pred  = (probs > 0.5).float()
    acc   = (pred == occl_gt).float().mean()
    return float(acc)

@no_grad()
def validate_sintel(args, model, iters=12, quiet=False):
    """Validation on Sintel (train split), RAFT-style API but WAFT outputs."""
    from MFT_WAFT.MFT.WAFT.train_waft_occlusion_uncertainty import uncertainty_metrics

    model.eval()
    results = {}

    for dstype in ['clean', 'final']:
        # reset meters for this dstype
        for m in [val_epe, val_px1, val_px3, val_px5, val_fl, val_unc, val_occ]:
            m.reset()
        # --- simple accumulators for OU validation metrics ---
        unc_count = 0
        occ_sum = {}   # accumulate occlusion metrics
        unc_sum = {}   # accumulate uncertainty metrics
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, load_occlusion=False)
        val_loader  = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)

        for data_blob in tqdm(val_loader, disable=quiet):
            if len(data_blob) == 5:
                image1, image2, flow_gt, valid, occl_gt = [x.cuda(non_blocking=True) for x in data_blob]
            else:
                image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
                occl_gt = None

            #image1, image2, flow_gt, valid, occl_gt = [x.cuda(non_blocking=True) for x in data_blob]

            padder = InputPadder(image1.shape)
            #image1, image2 = padder.pad(image1, image2)
            image1p, image2p = padder.pad(image1, image2)


            # Forward — WAFT_OU returns dict with lists
            from torch.cuda.amp import autocast

            with autocast(enabled=args.mixed_precision):
                out = model(image1p, image2p, iters=iters, test_mode=True)

            #out = model(image1, image2, iters=iters, test_mode=True)

            #flow_pred = out['flow'][-1]                    # Bx2xHxW
            flow_pred = padder.unpad(out['flow'][-1]) 
            #flow_pred = padder.unpad(flow_pred).cpu()
            #flow_gt   = flow_gt.cpu()
            #valid     = valid.cpu()

            # === Flow metrics ===
            epe = torch.sum((flow_pred - flow_gt) ** 2, dim=1).sqrt()  # [B,H,W]
            mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
            val_mask = (valid[:, 0] >= 0.5) 
            #val = (valid.squeeze(1) >= 0.5)                            # [B,H,W]

            px1 = (epe < 1.0).float()
            px3 = (epe < 3.0).float()
            px5 = (epe < 5.0).float()
            outlier = ((epe > 3.0) & ((epe / (mag + 1e-8)) > 0.05)).float()

            # aggregate per-sample (batch=1 here)
            if val_mask.any():
                val_epe.update(epe[val_mask].mean().item(), 1)
                val_px1.update(px1[val_mask].mean().item(), 1)
                val_px3.update(px3[val_mask].mean().item(), 1)
                val_px5.update(px5[val_mask].mean().item(), 1)
                val_fl.update(100.0 * outlier[val_mask].mean().item(), 1)
            #val_epe.update(epe[val].mean().item(), 1)
            #val_px1.update(px1[val].mean().item(), 1)
            #val_px3.update(px3[val].mean().item(), 1)
            #val_px5.update(px5[val].mean().item(), 1)
            #val_fl.update(100.0 * outlier[val].mean().item(), 1)

            # === Optional: uncertainty / occlusion monitoring ===
            unc, occ = _extract_ou_from_output(out)
            if isinstance(unc, list): unc = unc[-1]
            if isinstance(occ, list): occ = occ[-1]

            #if occ is not None and occl_gt is not None and occ.shape[1] == 2:
            #    occ_logits = padder.unpad(occ)  # [B,2,H,W]
            #    m = occlusion_metrics(occ_logits, occl_gt, valid)  # returns dict of scalars
            #    for k, v in m.items():
            #        occ_sum[k] = occ_sum.get(k, 0.0) + float(v)
            #    ou_count += 1  # count samples where we computed OU metrics
            if unc is not None:
                log_sigma2 = padder.unpad(unc)   # [B,1,H,W]
                m = uncertainty_metrics(log_sigma2, flow_pred, flow_gt, valid)
                for k, v in m.items():
                    unc_sum[k] = unc_sum.get(k, 0.0) + float(v)
                # only increment if we didn't already count this sample above
                if occ is None or occl_gt is None or occ.shape[1] != 2:
                    unc_count += 1

            #if unc is not None:
            #    unc = padder.unpad(unc).cpu()
            #    # log average predicted uncertainty on valid pixels
            #    val_unc.update(unc[val.unsqueeze(1)].mean().item(), 1)

            #if occ is not None:
            #    if occ.shape[1] == 2:
            #        # logits → accuracy vs occl_gt
            #        occl_gt = occl_gt.cpu()
            #        occ_unpad = padder.unpad(occ).cpu()
            #        acc = _occl_accuracy_from_logits(occ_unpad, occl_gt)
            #        val_occ.update(acc, 1)
            #    else:
            #        # single-channel "occlusion score" — just log mean
            #        occ = padder.unpad(occ).cpu()
            #        val_occ.update(occ[val.unsqueeze(1)].mean().item(), 1)

        # print + pack results
        #if not quiet:
        #    print(f"Validation ({dstype}) EPE: {val_epe.avg:.4f}, 1px: {val_px1.avg:.4f}, 3px: {val_px3.avg:.4f}, 5px: {val_px5.avg:.4f}, F1: {val_fl.avg:.2f}")

        results[f'eval/flow {dstype} EPE']  = val_epe.avg
        results[f'eval/flow {dstype} 1px']  = val_px1.avg
        results[f'eval/flow {dstype} 3px']  = val_px3.avg
        results[f'eval/flow {dstype} 5px']  = val_px5.avg
        results[f'eval/flow {dstype} F1']   = val_fl.avg

        if unc_count > 0:
            for k, s in unc_sum.items():
                results[f'eval/uncertainty/{dstype}/{k.split("/",1)[1] if "/" in k else k}'] = s / unc_count
        if not quiet:
            print(f"[Sintel-{dstype}] EPE {val_epe.avg:.4f} | 1px {val_px1.avg:.4f} | 3px {val_px3.avg:.4f} | 5px {val_px5.avg:.4f} | FL {val_fl.avg:.2f}")
            if unc_count > 0:
                if f'eval/uncertainty/{dstype}/epe2_sigma2_corr' in results:
                    print(f"  Unc corr: {results[f'eval/uncertainty/{dstype}/epe2_sigma2_corr']:.3f}")
                if f'eval/occlusion/{dstype}/occl_f1' in results:
                    print(f"  Occ F1: {results[f'eval/occlusion/{dstype}/occl_f1']:.3f}")


        # only add if we actually saw these signals
        #if val_unc.n > 0: results[f'eval/uncertainty {dstype} mean'] = val_unc.avg
        #if val_occ.n > 0: results[f'eval/occlusion {dstype}']        = val_occ.avg

    return results


@no_grad()
def validate_kitti(args, model, iters=24):
    """KITTI-2015 train split (same structure as RAFT)."""
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    epe_list, out_list = [], []
    for idx in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, _ = val_dataset[idx]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        out = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(out['flow'][-1]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = (valid_gt.view(-1) >= 0.5)

        outlier = ((epe > 3.0) & ((epe / (mag + 1e-8)) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(outlier[val].cpu().numpy())

    epe = float(np.mean(epe_list))
    f1  = 100.0 * float(np.mean(np.concatenate(out_list)))
    print(f"Validation KITTI: EPE={epe:.4f}, F1={f1:.2f}")
    return {'kitti-epe': epe, 'kitti-f1': f1}
