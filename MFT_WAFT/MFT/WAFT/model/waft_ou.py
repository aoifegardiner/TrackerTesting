import torch
import torch.nn as nn
import torch.nn.functional as F
from MFT_WAFT.MFT.WAFT.inference_tools import InferenceWrapper


def flow_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp img (B,C,H,W) using flow (B,2,H,W) in pixel units (dx, dy).
    """
    B, C, H, W = img.shape
    device = img.device

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=0).float()          # (2,H,W)
    base = base[None].repeat(B, 1, 1, 1)                 # (B,2,H,W)

    vgrid = base + flow                                  # pixel coords
    vgrid_x = 2.0 * (vgrid[:, 0] / (W - 1)) - 1.0
    vgrid_y = 2.0 * (vgrid[:, 1] / (H - 1)) - 1.0
    grid = torch.stack((vgrid_x, vgrid_y), dim=-1)        # (B,H,W,2)

    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

def _last_tensor(x):
    # unwrap list/tuple nesting until we hit a tensor
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[-1]
    return x


class WAFT_OU_FromFlow(nn.Module):
    """
    Wrap a pretrained flow model (WAFT) and add occlusion + uncertainty heads.

    Uses ONLY out["flow"][-1] plus image warp residual features.
    Does NOT require flow-model internals (net/inp/corr/...).

    Outputs:
      out["occlusion"]   = [logits]     logits shape (B,2,H,W)
      out["uncertainty"] = [log_sigma2] log variance shape (B,1,H,W)
    """
    def __init__(self, flow_model: nn.Module, args,
                 use_rgb_residual: bool = False,
                 detach_flow_features: bool = True):
        super().__init__()
        self.args = args
        self.use_rgb_residual = use_rgb_residual
        self.detach_flow_features = detach_flow_features

        self.flow_model = flow_model  # the ViTWarpV8 nn.Module
        self.flow_infer = InferenceWrapper(
            self.flow_model,
            scale=getattr(args, "scale", 0),           # or just 0
            train_size=getattr(args, "train_size", None),
            pad_to_train_size=getattr(args, "pad_to_train_size", False),
            tiling=getattr(args, "tiling", False),
        )        

        # Features: flow(2) + residual(1 or 3) + flow_mag(1) => 4 or 6 channels
        in_ch = 2 + (3 if use_rgb_residual else 1) + 1
        hidden = 64

        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.occlusion_head = nn.Conv2d(hidden, 2, 1)   # logits for {non-occ, occ}
        self.uncertainty_head = nn.Conv2d(hidden, 1, 1) # log(sigma^2)

    def convert_to_pixel_flow(self, flow, hw):
        H, W = hw
        f = flow.clone()
        f[:, 0] *= (W - 1) / 2.0
        f[:, 1] *= (H - 1) / 2.0
        return f

    def forward(self, image1, image2, iters=None, flow_gt=None, **kwargs):
        # keep compatibility with callers that pass test_mode
        kwargs.pop("test_mode", None)

        # Run flow model as-is
        #out = self.flow_model(image1, image2, iters=iters, flow_gt=flow_gt, **kwargs)
        #out = self.flow_model.calc_flow(image1, image2)
        #with torch.no_grad():
        #    out = self.flow_infer.calc_flow(image1, image2)
        with torch.no_grad():
            out = self.flow_model(image1, image2, iters=iters)   # or correct signature

        
        
        flows = out["flow"] if isinstance(out["flow"], list) else [out["flow"]]
        flow_pred = flows[-1]

        #flows = out["flow"]

        #for i, f in enumerate(flows):
        #    mag = torch.linalg.norm(f, dim=1)
        #    print(i, f.shape, f.min().item(), f.max().item(), mag.mean().item(), mag.max().item())

        # Use final flow prediction (B,2,H,W)
        #flow_pred = out["flow"][-1]

        #flow_pred = out["flow"][-1]
        #flow_pred = self.convert_to_pixel_flow(flow_pred, image1.shape[-2:])

        #out["flow"][-1] = flow_pred
        #flow_scale = getattr(self.args, "flow_unit_scale", 32.0)  # default 32
        #flow_pred = flow_pred * flow_scale
        #out["flow"][-1] = flow_pred

        # Build features
        # Optionally detach flow so heads don't backprop into flow model
        flow_feat = flow_pred.detach() if self.detach_flow_features else flow_pred



        # Warp image2 to image1 using predicted flow
        img2_warp = flow_warp(image2, flow_feat)

        g1x = image1[:,:,:,1:] - image1[:,:,:,:-1]
        g2x = img2_warp[:,:,:,1:] - img2_warp[:,:,:,:-1]
        residual = (g1x - g2x).abs()
        residual = F.pad(residual, (0,1,0,0))  # match size
        residual = residual.mean(dim=1, keepdim=True)

        # Photometric residual
        #residual = (image1 - img2_warp).abs()
        #if not self.use_rgb_residual:
        #    residual = residual.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # Flow magnitude
        flow_mag = torch.sqrt(torch.sum(flow_feat ** 2, dim=1, keepdim=True) + 1e-6)

        feat = torch.cat([flow_feat, residual, flow_mag], dim=1)
        h = self.trunk(feat)

        occ_logits = self.occlusion_head(h)            # (B,2,H,W)
        log_sigma2 = self.uncertainty_head(h)          # (B,1,H,W)

        # Clamp uncertainty to avoid runaway
        var_min = getattr(self.args, "var_min", -12.0)
        var_max = getattr(self.args, "var_max",  12.0)

 
        log_sigma2 = log_sigma2.clamp(var_min, var_max)

        
       

        # Return single-element lists (do NOT duplicate)
        #out["occlusion"] = [occ_logits]
        #out["uncertainty"] = [log_sigma2]

        out = {
            "flow": flows,
            "occlusion": [occ_logits.contiguous()],
            "uncertainty": [log_sigma2],
        }
        return out


        
