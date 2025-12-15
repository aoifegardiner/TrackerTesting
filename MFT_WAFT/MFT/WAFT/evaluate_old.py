import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

#from torchvision import transforms
from MFT_WAFT.MFT.WAFT.config.parser import parse_args

from tqdm import tqdm
from MFT_WAFT.MFT.WAFT.model import fetch_model
from MFT_WAFT.MFT.WAFT.utils.utils import resize_data, load_ckpt

from MFT_WAFT.MFT.WAFT.dataloader.flow.chairs import FlyingChairs
from MFT_WAFT.MFT.WAFT.dataloader.flow.things import FlyingThings3D
from MFT_WAFT.MFT.WAFT.dataloader.flow.sintel import MpiSintel
from MFT_WAFT.MFT.WAFT.dataloader.flow.kitti import KITTI
from MFT_WAFT.MFT.WAFT.dataloader.flow.spring import Spring
from MFT_WAFT.MFT.WAFT.dataloader.flow.hd1k import HD1K
from MFT_WAFT.MFT.WAFT.dataloader.flow.testvid import testvid
from MFT_WAFT.MFT.WAFT.dataloader.stereo.tartanair import TartanAir
from MFT_WAFT.MFT.WAFT.dataloader.stereo.stir import STIR
from MFT_WAFT.MFT.WAFT.utils.flow_viz import flow_to_image
from scipy.ndimage import center_of_mass
from torch.utils.data import DataLoader

from MFT_WAFT.MFT.WAFT.inference_tools import InferenceWrapper, AverageMeter

val_loss = AverageMeter()
val_epe = AverageMeter()
val_fl = AverageMeter()
val_px1 = AverageMeter()

def reset_all_metrics():
    val_loss.reset()
    val_epe.reset()
    val_fl.reset()
    val_px1.reset()

def update_metrics(args, output, flow_gt, valid):
    flow = output['flow'][-1]
    batch_size = flow.shape[0]
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    val = valid >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    px1 = (epe < 1.0).float()
    nf = []
    for i in range(len(output['flow'])):                 
        raw_b = output['info'][i][:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = output['info'][i][:, :2]
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
        term2 = ((flow_gt - output['flow'][i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        term1 = weight - math.log(2) - log_b
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        nf.append(nf_loss.mean(dim=1))

    loss = torch.zeros_like(nf[-1])
    for i in range(len(nf)):
        loss += (args.gamma ** (len(nf) - i - 1)) * nf[i]
    for i in range(batch_size):
        val_epe.update(epe[i][val[i]].mean().item(), 1)
        val_px1.update(px1[i][val[i]].mean().item(), 1)
        val_fl.update(100 * out[i][val[i]].sum().item(), val[i].sum().item())
        val_loss.update(loss[i][val[i]].mean().item(), 1)

@torch.no_grad()
def validate_sintel(args, model):
    """ Peform validation using the Sintel (train) split """
    for dstype in ['clean', 'final']:
        reset_all_metrics()
        val_dataset = MpiSintel(split='training', dstype=dstype)
        val_loader = data.DataLoader(val_dataset, batch_size=1, 
            pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
        pbar = tqdm(total=len(val_loader))
        print(f"load data success {len(val_loader)}")
        for i_batch, data_blob in enumerate(val_loader):
            image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
            output = model.calc_flow(image1, image2)
            update_metrics(args, output, flow_gt, valid)
            pbar.update(1)
        pbar.close()
        print(f"Validation {dstype} EPE: {val_epe.avg}, 1px: {100 * (1 - val_px1.avg)}")
        
        return {
            f"sintel_{dstype}_EPE": val_epe.avg,
            f"sintel_{dstype}_1px": 100 * (1 - val_px1.avg)
        }

@torch.no_grad()
def validate_sintel_new(model, iters=12, n_val=None, subsplit=None, quiet=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, subsplit=subsplit, load_occlusion=True)
        epe_list = []
        uncer_loss_list = []
        occl_loss_list = []
        occl_accuracy_list = []
        uncer_overshoot_list = []
        uncer_sub_1px_list = []
        uncer_sub_5px_list = []
        

        for val_id in range(len(val_dataset)):
            if (n_val is not None) and (val_id >= n_val):
                break
            image1, image2, flow_gt, _, occl_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            prediction_dict = model(image1, image2, iters=iters, test_mode=True)
            flow_low, flow_pr = prediction_dict['coords'], prediction_dict['flow']

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if model.uncertainty_estimation:
                uncertainty_pr = prediction_dict['uncertainty']
                uncertainty = padder.unpad(uncertainty_pr[0]).cpu()
                uncer_loss = uncertainty_loss(uncertainty, flow, flow_gt)
                uncer_loss_list.append(uncer_loss.view(-1).numpy())

                overshoot, sub_1, sub_5 = uncertainty_eval(uncertainty, flow, flow_gt)
                uncer_overshoot_list.append(overshoot)
                uncer_sub_1px_list.append(sub_1)
                uncer_sub_5px_list.append(sub_5)

            if model.occlusion_estimation:
                occl_pr = prediction_dict['occlusion']
                occlusion = padder.unpad(occl_pr[0]).cpu()
                occl_loss = occlusion_loss(occlusion, occl_gt)
                occl_loss_list.append(occl_loss.view(-1).numpy())

                occl_accuracy_list.append(occlusion_accuracy(occlusion, occl_gt))

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        if not quiet:
            print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[f'eval/flow {dstype}'] = np.mean(epe_list)

        if model.uncertainty_estimation:
            # uncer_all = np.concatenate(uncer_loss_list)
            # uncert_mean = np.mean(uncer_all)
            # results[f'eval/uncertainty loss {dstype}'] = uncert_mean

            overshoot = np.mean(uncer_overshoot_list)
            sub_1 = np.mean(uncer_sub_1px_list)
            sub_5 = np.mean(uncer_sub_5px_list)
            results[f'eval/uncertainty overshoot {dstype}'] = overshoot
            results[f'eval/uncertainty sub_1 {dstype}'] = sub_1
            results[f'eval/uncertainty sub_5 {dstype}'] = sub_5
        if model.occlusion_estimation:
            occl_all = np.concatenate(occl_loss_list)
            occl_mean = np.mean(occl_all)
            # print("Validation (%s) OCCL: %f, Uncertainty %f" % (dstype, occl_mean, uncert_mean))
            results[f'eval/occl loss {dstype}'] = occl_mean

            occl_mean = np.mean(occl_accuracy_list)
            if not quiet:
                print("Validation (%s) OCCL_acc: %f, EPE overshoot: %f, sub1: %f, sub5: %f" % (dstype, occl_mean,
                                                                                               overshoot, sub_1, sub_5))
            results[f'eval/occl acc {dstype}'] = occl_mean

    return results

@torch.no_grad()
def validate_kitti(args, model):
    """ Peform validation using the KITTI-2015 (train) split """
    val_dataset = KITTI(split='training')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    print(f"load data success {len(val_loader)}")
    reset_all_metrics()
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda(non_blocking=True) for x in data_blob]
        output = model.calc_flow(image1, image2)
        update_metrics(args, output, flow_gt, valid_gt)
    
    print("Validation KITTI: %f, %f" % (val_epe.avg, val_fl.avg))

@torch.no_grad()
def validate_spring(args, model):
    """ Peform validation using the Spring (val) split """
    val_dataset = Spring(split='val') #+ Spring(split='train')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)
    
    reset_all_metrics()
    print(f"load data success {len(val_loader)}")
    pbar = tqdm(total=len(val_loader))
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        output = model.calc_flow(image1, image2)
        update_metrics(args, output, flow_gt, valid)
        pbar.update(1)

    pbar.close()
    print(f"Validation Spring EPE: {val_epe.avg}, 1px: {100 * (1 - val_px1.avg)}, loss: {val_loss.avg}")


@torch.no_grad()
def validate_chairs(args, model):
    """ Perform validation using the Chairs (val) split and save predicted flow as .pt """

    from pathlib import Path

    val_dataset = FlyingChairs(split='validation') 
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=16, drop_last=False)

    reset_all_metrics()

    predicted_flows = []  # List to collect predicted flows

    pbar = tqdm(total=len(val_loader))
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow_gt, valid = [x.cuda(non_blocking=True) for x in data_blob]
        
        output = model.calc_flow(image1, image2)  # predicted flow (e.g. shape [1, 2, H, W])
        flow_pred = output['flow'][-1].detach().cpu()  # get final predicted flow from sequence

        predicted_flows.append(flow_pred[0])  # remove batch dimension (1, 2, H, W) -> (2, H, W)

        update_metrics(args, output, flow_gt, valid)
        pbar.update(1)

    pbar.close()

    # Save predicted flows as .pt
    save_path = Path(f"predicted_flows_chairs_val.pt")
    torch.save(predicted_flows, save_path)



def extract_keypoints(segmentation):
    labels = torch.unique(segmentation)
    keypoints = []
    for label in labels:
        if label == 0:
            continue  
        mask = (segmentation == label).cpu().numpy()
        if np.sum(mask) == 0:
            continue
        cy, cx = center_of_mass(mask)
        keypoints.append((int(cx), int(cy), int(label.item())))
    return keypoints 

def compute_sparse_flow(seg_start, seg_end):
    kp_start = extract_keypoints(seg_start)
    kp_end = extract_keypoints(seg_end)

    end_dict = {label: (x, y) for x, y, label in kp_end}

    flow_points = []
    for x0, y0, label in kp_start:
        if label in end_dict:
            x1, y1 = end_dict[label]
            flow_points.append((x0, y0, x1 - x0, y1 - y0)) 
    return flow_points

def evaluate_sparse_flow(flow_pred, sparse_flow_points):
    epe_sum = 0
    count = 0
    H, W = flow_pred.shape[1:3]
    for (x, y, dx, dy) in sparse_flow_points:
        if 0 <= x < W and 0 <= y < H:
            if flow_pred.ndim == 3:  # [C,H,W]
                pred_flow = flow_pred[:, y, x].cpu().numpy()  # shape (2,)
            else:  # [B,C,H,W]
                pred_flow = flow_pred[0, :, y, x].cpu().numpy()
            #pred_flow = flow_pred[0, :, y, x].cpu().numpy() 
            gt_flow = np.array([dx, dy])
            epe = np.linalg.norm(pred_flow - gt_flow)
            epe_sum += epe
            count += 1
    if count == 0:
        return None  
    return epe_sum / count


def warp_flow(flow, ref_flow):
    # flow: [2, H, W] — the flow you want to warp
    # ref_flow: [2, H, W] — the flow to warp with

    B, H, W = 1, flow.shape[1], flow.shape[2]

    # Build mesh grid
    y, x = torch.meshgrid(
        torch.arange(0, H, device=flow.device),
        torch.arange(0, W, device=flow.device),
        indexing="ij"
    )
    grid = torch.stack((x, y), dim=0).float()  # [2, H, W]

    # Add ref_flow to grid => where to sample from
    vgrid = grid + ref_flow  # [2, H, W]

    # Normalize to [-1, 1] for grid_sample
    vgrid[0] = 2.0 * vgrid[0] / (W - 1) - 1.0
    vgrid[1] = 2.0 * vgrid[1] / (H - 1) - 1.0

    vgrid = vgrid.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 2]

    # flow to warp must be [B, C, H, W]
    flow = flow.unsqueeze(0)  # [1, 2, H, W]

    warped = F.grid_sample(flow, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0)  # [2, H, W]

@torch.no_grad()
def validate_stir(args, model):
    import cv2



    val_dataset = STIR(root=args.dataset_root)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    val_epe_meter = AverageMeter()

    #train_H, train_W = 432, 960
    train_H, train_W = args.image_size

    for i, (frames, seg_start, seg_end) in enumerate(val_loader):

        frames = frames[0]  # Remove batch dim => [T, 3, H, W]

        #T = frames.shape[0]
        T, _, orig_H, orig_W = frames.shape

        total_flow = None
        all_flows = []

        for t in range(T - 1):
            frame1 = frames[t].unsqueeze(0).cuda(non_blocking=True) #frames[t].unsqueeze(0)
            frame2 = frames[t + 1].unsqueeze(0).cuda(non_blocking=True) #frames[t + 1].unsqueeze(0)
            
            with torch.cuda.amp.autocast():
                output = model.calc_flow(frame1, frame2)

            flow_pred = output['flow'][-1].detach()

            if 'var' in output:
                var_pred = output['var'][-1].detach().cpu() 
                var_pred = var_pred.squeeze().numpy()      
            else:
                var_pred = None

            flow_pred_cpu = flow_pred.squeeze(0).cpu() 
            all_flows.append(flow_pred_cpu)
            total_flow = flow_pred_cpu

            
            if var_pred is not None:
                var_norm = (var_pred - var_pred.min()) / (var_pred.max() - var_pred.min() + 1e-8)
                var_heatmap = (255 * var_norm).astype(np.uint8)
                var_heatmap = cv2.applyColorMap(var_heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(f"var_vis_batch{i}_t{t}.png", var_heatmap)
            else:
                print(f"[DEBUG] No variance output for batch {i}, t={t}")   

            del frame1, frame2, output, flow_pred,  
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        save_path = f"pred_flow_1908_{i}_new.pt"
        torch.save(all_flows, save_path)
        sparse_flow = compute_sparse_flow(seg_start[0], seg_end[0])
        predicted_end_points = []
        gt_end_points = []

        H, W = all_flows[0].shape[1], all_flows[0].shape[2]

        for (x, y, dx_gt, dy_gt) in sparse_flow:
            x_pos = x
            y_pos = y

            # Follow flow through all frames to get predicted final position
            for flow in all_flows:
                u_map = flow[0]  # [H, W]
                v_map = flow[1]
                x_clamp = int(np.clip(x_pos, 0, W - 1))
                y_clamp = int(np.clip(y_pos, 0, H - 1))
                u = u_map[y_clamp, x_clamp].item()
                v = v_map[y_clamp, x_clamp].item()
                x_pos += u
                y_pos += v

            predicted_end_points.append([x_pos, y_pos])
            # Ground truth end point:
            gt_end_points.append([x + dx_gt, y + dy_gt])

        predicted_end_points = np.array(predicted_end_points)
        gt_end_points = np.array(gt_end_points)

        # Compute End Point Error (EPE) per keypoint
        errors = np.linalg.norm(predicted_end_points - gt_end_points, axis=1)
        avg_epe = errors.mean()

        pred_sparse_flow = []
        for (x, y, dx, dy) in sparse_flow:
            u = total_flow[0, y, x].item()
            v = total_flow[1, y, x].item()
            pred_sparse_flow.append([x, y, u, v])

        pred_sparse_flow = torch.tensor(pred_sparse_flow, dtype=torch.float32)

        epe = evaluate_sparse_flow(total_flow, sparse_flow)
        if epe is not None:
            val_epe_meter.update(epe, 1)

    print(f"STIR Test EPE on sparse landmarks: {val_epe_meter.avg:.4f}")




@torch.no_grad()
def validate_test(args, model):
    val_dataset = testvid(root=args.dataset_root)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('val loader: ',val_loader)
    val_epe_meter = AverageMeter()
    print(f"Loaded test test data: {len(val_loader)} samples")
    for i, (frames) in enumerate(val_loader):
        if i >= 4:
            break

        frames = frames[0] 
        T = frames.shape[0]
        total_flow = None
        all_flows = []

        for t in range(T - 1):
            if t>= 100:
                break
            
            frame1 = frames[t].unsqueeze(0).cuda(non_blocking=True) 
            frame2 = frames[t + 1].unsqueeze(0).cuda(non_blocking=True) 

            with torch.cuda.amp.autocast():
                output = model.calc_flow(frame1, frame2)

            flow_pred = output['flow'][-1].detach()

            flow_pred_cpu = flow_pred.cpu()
            all_flows.append(flow_pred_cpu)

            if total_flow is None:
                total_flow = flow_pred_cpu
            else:
                total_flow += flow_pred_cpu
            del frame1, frame2, output, flow_pred, flow_pred_cpu

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        #save_path = f"pred_flow_test_{i}.pt"
        #torch.save(all_flows, save_path)


        
def eval(args):
    args.gpus = [0]


    model = fetch_model(args)
    print(f"[DEBUG] fetch_model sees image_size={args.image_size}")

    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)
    print(f"[DEBUG] InferenceWrapper train_size={wrapped_model.train_size}")


    with torch.no_grad():
        if args.dataset == 'spring':
            validate_spring(args, wrapped_model)
        elif args.dataset == 'sintel':
            validate_sintel(args, wrapped_model)
        elif args.dataset == 'kitti':
            validate_kitti(args, wrapped_model)
        elif args.dataset == 'stir':
            validate_stir(args, wrapped_model)
        elif args.dataset == 'testvid':
            validate_test(args, wrapped_model)
        elif args.dataset == 'chairs':
            validate_chairs(args, wrapped_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--dataset', help='dataset to evaluate on', choices=['sintel', 'kitti', 'spring', 'stir', 'testvid','chairs'], required=True, type=str)
    parser.add_argument('--scale', help='scale factor for input images', default=0.0, type=float)
    parser.add_argument('--dataset_root', help='root directory of the dataset', required=False, default=None, type=str)
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()
