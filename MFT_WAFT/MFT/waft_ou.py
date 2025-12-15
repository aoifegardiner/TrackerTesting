import torch
import torch.nn as nn
import einops
import cv2
import numpy as np

class WAFTFlowOUForMFT(nn.Module):
    """
    Drop-in WAFT wrapper that matches MFT's FLOW-OU interface.

    Exposes:
        compute_flow(im1, im2, mode='flow', init_flow=None, numpy_out=False)

    where im1, im2 are BGR uint8 numpy arrays of shape (H, W, 3),
    and returns:
        flow: (2, H, W) torch tensor (or numpy if numpy_out=True)
        extra: dict with keys:
            'occlusion': (1, H, W)
            'sigma':     (1, H, W)   # uncertainty (std dev)
    """

    def __init__(self, core_waft_model, device="cuda"):
        super().__init__()
        self.model = core_waft_model   # your trained WAFT network
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # MFT uses it in inference mode

    @torch.no_grad()
    def compute_flow(self, im1, im2, mode='flow', init_flow=None,
                     numpy_out=False, **kwargs):
        # --- 1. Inputs: BGR uint8 H×W×3 numpy (MFT convention) ---
        assert im1.ndim == 3 and im1.shape[2] == 3
        assert im2.ndim == 3 and im2.shape[2] == 3
        H, W = im1.shape[:2]

        # --- 2. Convert to CHW RGB float32 torch on self.device ---
        image1 = einops.rearrange(
            torch.from_numpy(im1[:, :, ::-1].copy()),
            "H W C -> 1 C H W"
        ).float().to(self.device)

        image2 = einops.rearrange(
            torch.from_numpy(im2[:, :, ::-1].copy()),
            "H W C -> 1 C H W"
        ).float().to(self.device)

        # --- 3. Call WAFT model: it should expose calc_flow(image1, image2) ---
        # Expected output format:
        #   out["flow"] : list of (B, 2, H, W)
        #   out["info"] : list of (B, 4, H, W)   (optional)
        out = self.model.calc_flow(image1, image2)

        # --- 4. Take final flow prediction: (2, H, W) ---
        flow = out["flow"][-1][0]    # (2, H, W)
        assert flow.shape[1:] == (H, W), \
            f"Flow spatial size mismatch: {flow.shape[1:]} vs ({H},{W})"

        # --- 5. Derive occlusion & uncertainty from WAFT "info" channels, if present ---
        if "info" in out:
            # info_last: (4, H, W) — you already saw this in your debug prints
            info_last = out["info"][-1][0]  # (4, H, W)

            # One reasonable mapping (the same spirit as you used before):
            #   channel 0: inverse confidence (alpha) → sigma = exp(-clamp(alpha))
            #   channel 3: occlusion logit         → p_occ = sigmoid(channel 3)
            sigma_raw      = info_last[0:1]  # (1, H, W)
            occlusion_raw  = info_last[3:4]  # (1, H, W)

            sigma = torch.exp(-torch.clamp(sigma_raw, min=-12, max=12))
            occlusion = torch.sigmoid(occlusion_raw)
        else:
            # Fallback if you haven't trained an info head yet
            occlusion = torch.zeros((1, H, W), device=flow.device, dtype=flow.dtype)
            sigma     = torch.full((1, H, W), 1e-3, device=flow.device, dtype=flow.dtype)

        if numpy_out:
            flow_np = flow.detach().cpu().numpy()
            occ_np  = occlusion.detach().cpu().numpy()
            sig_np  = sigma.detach().cpu().numpy()
            return flow_np, {"occlusion": occ_np, "sigma": sig_np}

        return flow, {"occlusion": occlusion, "sigma": sigma}
