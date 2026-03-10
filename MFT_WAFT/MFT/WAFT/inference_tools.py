import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

def generate_gaussian(size, sigma):
    """
    Generate a 2D Gaussian pattern based on the distance to the center.

    Args:
        size (tuple): (height, width) of the output pattern.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: 2D Gaussian pattern.
    """
    height, width = size
    # Create a coordinate grid
    y = torch.arange(0, height).view(-1, 1) / height
    x = torch.arange(0, width).view(1, -1) / width
    
    # Compute the center coordinates
    center_y, center_x = 0.5, 0.5
    
    # Compute the squared distance from each point to the center
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    
    # Apply the Gaussian function
    gaussian = torch.exp(-distance_squared / (2 * sigma ** 2))
    return gaussian

def _unwrap_tensor(x):
    # Peel off list/tuple wrappers until we hit a tensor (or give up)
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]
    return x if torch.is_tensor(x) else None

def _storage_ptr(x) -> int:
    t = _unwrap_tensor(x)
    if t is None:
        return -1  # indicates "not a tensor"
    try:
        return t.untyped_storage().data_ptr()  # newer torch
    except Exception:
        try:
            return t.storage().data_ptr()       # older torch
        except Exception:
            return t.data_ptr()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val += val
        self.count += n
        self.avg = self.val / self.count

class InferenceWrapper(object):
    def __init__(self, model, scale=0, train_size=None, pad_to_train_size=False, tiling=False):
        self.model = model
        self.train_size = train_size
        self.scale = scale
        #self.pad_to_train_size = pad_to_train_size
        #self.tiling = tiling
        self.pad_to_train_size = False
        self.tiling = False
        
        

    def inference_padding(self, image):
        h, w = self.train_size
        H, W = image.shape[2:]
        pad_h = max(h - H, 0)
        pad_w = max(w - W, 0)
        padded_image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0)
        return padded_image, pad_h, pad_w

    def patch_inference(self, image1, image2, patches, tile_h, tile_w):
        output = None
        n, _, h, w = image1.shape
        valid = torch.zeros((n, h, w), device=image1.device)
        for h_ij, w_ij in patches:
            hl, hr = h_ij, h_ij + tile_h
            wl, wr = w_ij, w_ij + tile_w
            weight = generate_gaussian((hr - hl, wr - wl), 1).to(image1.device).unsqueeze(0)
            image1_ij = image1[:, :, hl: hr, wl: wr]
            image2_ij = image2[:, :, hl: hr, wl: wr]
            output_ij = self.model(image1_ij, image2_ij)

            def _get0(key):
                v = output_ij.get(key, None)
                if v is None:
                    return None
                if isinstance(v, list):
                    v = v[0]
                return v.detach().clone()

            occ_ij = _get0("occlusion")      # (n,2,tile_h,tile_w) logits typically
            unc_ij = _get0("uncertainty")    # (n,1,tile_h,tile_w) log_sigma2

            valid[:, hl:hr, wl:wr] += weight

#            if output is None:
#                output = {}
#                for key, val in output_ij.items():
#                    if isinstance(val, list):
#                        L = len(val)
#
#                        if key == "flow":
#                            output[key] = [torch.zeros((n, 2, h, w), device=image1.device) for _ in range(L)]
#                        elif key == "info":
#                            output[key] = [torch.zeros((n, val[0].shape[1], h, w), device=image1.device) for _ in range(L)]
#                        elif key == "occlusion":
#                            # NOTE: logits are usually (n,2,*,*); keep channels consistent with val[0]
#                            C = val[0].shape[1]
#                            output[key] = [torch.zeros((n, C, h, w), device=image1.device) for _ in range(L)]
#                        elif key == "uncertainty":
#                            C = val[0].shape[1]  # usually 1
#                            output[key] = [torch.zeros((n, C, h, w), device=image1.device) for _ in range(L)]
#                        else:
#                            # generic list output
#                            C = val[0].shape[1]
#                            output[key] = [torch.zeros((n, C, h, w), device=image1.device) for _ in range(L)]
#                    else:
#                        # Non-list keys: just copy through (rare)
#                        output[key] = val

            if output is None:
                output = {}

                # only keep these keys
                keep_keys = {"flow", "occlusion", "uncertainty"}

                for key, val in output_ij.items():
                    if key not in keep_keys:
                        continue
                    if not isinstance(val, list):
                        continue
                    
                    L = len(val)
                    C = val[0].shape[1]
                    output[key] = [torch.zeros((n, C, h, w), device=image1.device) for _ in range(L)]

            keep_keys = {"flow", "occlusion", "uncertainty"}

            for key, val in output_ij.items():
                if key not in keep_keys:
                    continue
                if not isinstance(val, list) or len(val) == 0:
                    continue
                    
                if key == "occlusion":
                    if occ_ij is not None:
                        output[key][0][:, :, hl:hr, wl:wr] += weight * occ_ij
                    continue
                
                if key == "uncertainty":
                    if unc_ij is not None:
                        output[key][0][:, :, hl:hr, wl:wr] += weight * unc_ij
                    continue
                
                # flow: multi-step list
                for i in range(len(val)):
                    output[key][i][:, :, hl:hr, wl:wr] += weight * val[i].detach()




#            for key, val in output_ij.items():
#                if not isinstance(val, list):
#                    continue
#
#                if key == "occlusion":
#                    # use our cloned tensor occ_ij (avoids aliasing issues)
#                    if occ_ij is not None:
#                        output[key][0][:, :, hl:hr, wl:wr] += weight * occ_ij
#                    continue
#
#                if key == "uncertainty":
#                    if unc_ij is not None:
#                        output[key][0][:, :, hl:hr, wl:wr] += weight * unc_ij
#                    continue
#
#                for i in range(len(val)):
#                    output[key][i][:, :, hl:hr, wl:wr] += weight * val[i].detach()


        # Normalize (OUT-OF-PLACE to avoid weird aliasing surprises)
        denom = valid.unsqueeze(1).clamp_min(1e-9)
        for key, val in output.items():
            if not isinstance(val, list):
                continue
            for i in range(len(val)):
                output[key][i] = output[key][i] / denom

        return output                    
#            for key in output_ij:
#                if key in ("occlusion", "uncertainty"):
#                    # Only one output
#                    output[key][0][:, :, hl:hr, wl:wr] += weight * output_ij[key][0]
#                    continue
#
#                # Normal multi-step outputs (flow, info)
#                for i in range(len(output_ij[key])):
#                    output[key][i][:, :, hl:hr, wl:wr] += weight * output_ij[key][i]


        # Normalize
#        for key in output:
#            for i in range(len(output[key])):
#                output[key][i] /= valid.unsqueeze(1)
#                if key == "uncertainty":
#                    print("[ALLOC] unc buf max right after zeros:",
#                          output[key][0].max().item(),
#                          "storage:", _storage_ptr(output[key][0]))
#
#
#        if "uncertainty" in output:
#            post = output["uncertainty"][0].detach()
#            print(f"[STITCH POST-NORM] unc max={post.max().item():.6f} min={post.min().item():.6f} mean={post.mean().item():.6f}")
#        
#        return output

    def forward_flow(self, image1, image2):
        H, W = image1.shape[2:]
        if self.pad_to_train_size:
            image1, inf_pad_h, inf_pad_w = self.inference_padding(image1)
            image2, inf_pad_h, inf_pad_w = self.inference_padding(image2)
        else:
            inf_pad_h, inf_pad_w = 0, 0

        if self.tiling and self.pad_to_train_size:
            h, w = image1.shape[2:]
            tile_h, tile_w = self.train_size
            step_h, step_w = tile_h // 4 * 3, tile_w // 4 * 3
            patches = []
            for i in range(0, h, step_h):
                for j in range(0, w, step_w):
                    h_ij = max(min(i, h - tile_h), 0)
                    w_ij = max(min(j, w - tile_w), 0)
                    patches.append((h_ij, w_ij))

            # remove duplicates
            patches = list(set(patches))
        else:
            h, w = image1.shape[2:]
            tile_h, tile_w = h, w
            patches = [(0, 0)]

        output = self.patch_inference(image1, image2, patches, tile_h, tile_w)

        #for i in range(len(output['flow'])):
        #    for key in output.keys():
        #        output[key][i] = output[key][i][:, :, inf_pad_h // 2: inf_pad_h // 2 + H, inf_pad_w // 2: inf_pad_w // 2 + W]
        for key in output.keys():
            if not isinstance(output[key], list):
                continue
            #if key in ("occlusion", "uncertainty"):
            #    continue  
            
            for i in range(len(output[key])):
                output[key][i] = output[key][i][
                    :,
                    :,
                    inf_pad_h // 2 : inf_pad_h // 2 + H,
                    inf_pad_w // 2 : inf_pad_w // 2 + W,
                ]


        return output
    
    def calc_flow(self, image1, image2):
        img1 = F.interpolate(image1, scale_factor=2 ** self.scale, mode='bilinear', align_corners=True)
        img2 = F.interpolate(image2, scale_factor=2 ** self.scale, mode='bilinear', align_corners=True)
        H, W = img1.shape[2:]
        output = self.forward_flow(img1, img2)
        for i in range(len(output['flow'])):
            for key in output.keys():
                #if key in ("occlusion", "uncertainty"):
                #    # Do NOT tile, do NOT rescale using flow count
                #    continue
                if not isinstance(output[key], list):
                    continue

                ii = min(i, len(output[key]) - 1)
                if 'flow' in key:
                    #output[key][i] = F.interpolate(output[key][i], scale_factor=0.5 ** self.scale, mode='bilinear', align_corners=True) * (0.5 ** self.scale)
                    output[key][ii] = F.interpolate(output[key][ii], scale_factor=0.5 ** self.scale,mode="bilinear", align_corners=True) * (0.5 ** self.scale)
                else:
                    #output[key][i] = F.interpolate(output[key][i], scale_factor=0.5 ** self.scale, mode='bilinear', align_corners=True)
                    output[key][ii] = F.interpolate(output[key][ii], scale_factor=0.5 ** self.scale, mode="bilinear", align_corners=True)
        return output




