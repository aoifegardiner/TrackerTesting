# MFT/waft.py
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import einops
import numpy as np
import torch.nn.functional as F


from MFT_WAFT.MFT.utils.geom_utils import torch_get_featuremap_coords, get_featuremap_coords
from MFT_WAFT.MFT.utils.misc import ensure_numpy
from MFT_WAFT.MFT.utils import interpolation

from MFT_WAFT.MFT.WAFT.model import fetch_model
from MFT_WAFT.MFT.WAFT.utils.utils import load_ckpt
from MFT_WAFT.MFT.WAFT.inference_tools import InferenceWrapper

logger = logging.getLogger(__name__)


def _json_to_args(json_path: Path, overrides: dict = None) -> SimpleNamespace:
    """Load WAFT JSON config and turn it into an argparse-like object."""
    d = json.loads(Path(json_path).read_text())
    overrides = overrides or {}
    d.update(overrides)
    # Ensure image_size is a tuple
    if "image_size" in d and isinstance(d["image_size"], list):
        d["image_size"] = tuple(d["image_size"])
    return SimpleNamespace(**d)


import torch
import torch.nn.functional as F
import numpy as np

class FlowOUTrackingResult:
    def __init__(self, flow, occlusion, sigma):
        """
        Container for flow outputs from RAFT or WAFT.

        Args:
            flow: torch.Tensor (2, H, W)
            occlusion: torch.Tensor (1, H, W)
            sigma: torch.Tensor (1, H, W) or None — optional uncertainty
        """
        self.flow = flow
        self.occlusion = occlusion
        self.sigma = sigma #if sigma is not None else torch.zeros_like(occlusion)

    def to(self, device):
        """Move all tensors to the given device."""
        self.flow = self.flow.to(device)
        self.occlusion = self.occlusion.to(device)
        self.sigma = self.sigma.to(device)
        return self
    
    def detach(self):
        """Detach all tensors from the computation graph."""
        self.flow = self.flow.detach()
        self.occlusion = self.occlusion.detach()
        self.sigma = self.sigma.detach()
        return self
    

    @staticmethod
    def identity(hw, device="cuda", dtype=torch.float32):
        """Create an identity (zero-motion) flow field for initialization."""
        H, W = hw
        flow = torch.zeros((2, H, W), device=device, dtype=dtype)
        occlusion = torch.zeros((1, H, W), device=device, dtype=dtype)
        sigma = torch.zeros((1, H, W), device=device, dtype=dtype)
        return FlowOUTrackingResult(flow, occlusion, sigma)

    def clone(self):
        """Clone all tensors."""
        return FlowOUTrackingResult(
            self.flow.clone(),
            self.occlusion.clone(),
            self.sigma.clone()
        )

    def cpu(self):
        """Move all tensors to CPU."""
        return FlowOUTrackingResult(
            self.flow.cpu(),
            self.occlusion.cpu(),
            self.sigma.cpu()
        )
    

    def warp_forward_points(self, queries):
        """Warp query points forward using bilinear sampling of flow."""
        _, H, W = self.flow.shape
        x_norm = (queries[:, 0] / (W - 1)) * 2 - 1
        y_norm = (queries[:, 1] / (H - 1)) * 2 - 1
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        flow_batch = self.flow.unsqueeze(0)  # (1,2,H,W)
        sampled_flow = F.grid_sample(flow_batch, grid, mode='bilinear', align_corners=True)
        sampled_flow = sampled_flow.squeeze(0).squeeze(-1).T  # (N,2)
        return queries + sampled_flow

    def warp_forward(self, img, mask=None):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 3 and img.shape[0] != 2:  # assume HWC
            img = img.permute(2, 0, 1)  # CHW
        img = img.unsqueeze(0)  # B,C,H,W

        flow = self.flow.unsqueeze(0)  # 1,2,H,W
        device = flow.device
        img = img.to(device)

        B, C, H, W = img.shape
        norm_flow = torch.zeros_like(flow)
        norm_flow[:, 0, :, :] = 2.0 * flow[:, 0, :, :] / max(W - 1, 1)
        norm_flow[:, 1, :, :] = 2.0 * flow[:, 1, :, :] / max(H - 1, 1)
        grid = torch.stack([norm_flow[:, 0, :, :], norm_flow[:, 1, :, :]], dim=-1)  # B,H,W,2

        warped = F.grid_sample(img.float(), grid, mode='bilinear', padding_mode='border', align_corners=True)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(device)
            warped = warped * mask.float().unsqueeze(0).unsqueeze(0)

        return warped.squeeze(0)
    
    #def chain(self, flow_next):
    #    """
    #    Compose this flow (A→B) with another flow (B→C) to get A→C.
    #    flow_next: torch.Tensor of shape (2,H,W)
    #    """
    #    # Warp flow_next back using current flow
    #    warped_next = self.warp_backward(flow_next)
#
    #    # Compose: total flow = current_flow + warped_next
    #    new_flow = self.flow + warped_next
#
    #    # Occlusions: OR combination
    #    new_occlusion = torch.clamp(self.occlusion + self.warp_backward(self.occlusion), 0, 1)
#
    #    # Sigmas: simple sum of uncertainties if available
    #    if hasattr(self, "sigma"):
    #        new_sigma = torch.sqrt(self.sigma ** 2 + self.warp_backward(self.sigma) ** 2)
    #    else:
    #        new_sigma = torch.zeros_like(new_occlusion)
#
    #    return FlowOUTrackingResult(new_flow, new_occlusion, new_sigma)
    #
    def chain(self, flow_next):
        """
        Compose this flow (A→B) with another flow (B→C) to get total flow A→C.
        Returns the composed flow tensor (2,H,W), NOT a FlowOUTrackingResult.
        """
        # Warp next flow into current coordinate frame
        warped_next = self.warp_backward(flow_next)
        # Combine: total flow = current_flow + warped_next
        new_flow = self.flow + warped_next
        return new_flow


    def warp_backward(self, tensor):
        """
        Warp a (2,H,W) or (1,H,W) tensor backward using this flow (like inverse warp).
        Used internally by MFT to align next-frame flow into the current frame's space.
        """
        _, H, W = self.flow.shape
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.flow.device),
            torch.arange(W, device=self.flow.device),
            indexing="ij"
        )
        # Convert to normalized coordinates [-1,1]
        x = grid_x + self.flow[0]
        y = grid_y + self.flow[1]
        x_norm = 2.0 * x / max(W - 1, 1) - 1.0
        y_norm = 2.0 * y / max(H - 1, 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1,H,W,2)

        # Ensure tensor shape (1,C,H,W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        warped = F.grid_sample(tensor, grid, mode="bilinear", align_corners=True)
        return warped.squeeze(0)
    
    def invalid_mask(self):
        """Compute a mask of invalid flows (those pointing outside the image)."""
        device = self.flow.device
        _, H, W = self.flow.shape

        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        # Add flow to get target positions
        x2 = x + self.flow[0]
        y2 = y + self.flow[1]

        # Mark positions outside the valid range
        invalid = (
            (x2 < 0) | (x2 >= W) |
            (y2 < 0) | (y2 >= H)
        )

        return invalid


class WAFTWrapper:
    """RAFTWrapper-compatible tracker for WAFT."""

    def __init__(self, config):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.C = config

        # Load JSON args
        waft_json = getattr(self.C, "waft_json", None) or getattr(self.C, "cfg", None)
        assert waft_json is not None, "WAFTWrapper: config.waft_json must be set"
        waft_json = Path(waft_json)

        overrides = {}
        if hasattr(self.C, "image_size"):
            overrides["image_size"] = self.C.image_size
        if hasattr(self.C, "scale"):
            overrides["scale"] = self.C.scale
        if hasattr(self.C, "iters"):
            overrides["iters"] = self.C.iters

        args = _json_to_args(waft_json, overrides=overrides)

        # Build WAFT model and load checkpoint
        model = fetch_model(args)
        ckpt_path = getattr(self.C, "ckpt", None) or getattr(self.C, "model", None)
        assert ckpt_path is not None, "WAFTWrapper: config.ckpt or config.model must be set"
        load_ckpt(model, ckpt_path)
        model = model.to(device).eval()

        # Wrap for inference
        self.model = InferenceWrapper(
            model,
            scale=args.scale,
            train_size=args.image_size,
            pad_to_train_size=False,
            tiling=False
        )

        # Internal state
        self.prev_frame = None
        self.initialized = False

    def init(self, frame):
        """Initialize tracker with first frame."""
        self.prev_frame = frame
        self.initialized = True
        # Return dummy result for first frame
        H, W = frame.shape[:2]
        flow = torch.zeros((2, H, W), device=self.device)
        occlusion = torch.zeros((1, H, W), device=self.device)
        return FlowOUTrackingResult(flow, occlusion)
        

    def track(self, frame, queries=None):
        """Compute flow from previous frame to current frame."""
        if not self.initialized:
            raise RuntimeError("WAFTWrapper must be initialized first.")

        # Compute flow and extra outputs (occlusion + sigma)
        flow, extra = self.compute_flow(self.prev_frame, frame)
        

        occlusion = extra.get('occlusion', None)
        sigma = extra.get('sigma', None)

        # Optional debug info
        # print(f"[WAFT TRACK] Flow mean={flow.abs().mean():.3f}, σ_mean={sigma.mean().item():.5f}, occ_mean={occlusion.mean().item():.5f}")

        # Update internal state
        self.prev_frame = frame

        # Wrap results for point-tracking compatibility
        result = FlowOUTrackingResult(flow, occlusion)
        result.sigma = sigma  # attach uncertainty map if consumers use it later
        return result
    

    #def compute_flow(self, src_img, dst_img, mode="flow", numpy_out=False):
    #def compute_flow(self, im1, im2, mode='flow', **kwargs):
#
    #    """Compute WAFT flow between two frames."""
    #    H, W = src_img.shape[:2]
#
    #    # Convert to Torch CHW RGB float
    #    image1 = einops.rearrange(torch.from_numpy(src_img[:, :, ::-1].copy()),
    #                              "H W C -> 1 C H W", C=3).float().to(self.device)
    #    image2 = einops.rearrange(torch.from_numpy(dst_img[:, :, ::-1].copy()),
    #                              "H W C -> 1 C H W", C=3).float().to(self.device)
#
    #    with torch.no_grad():
    #        output = self.model.calc_flow(image1, image2)
#
    #    # Last flow prediction, first batch
    #    flow = output["flow"][-1][0]  # (2,H,W)
    #    assert flow.shape[1:] == (H, W)
#
    #    # WAFT does not output occlusion → zeros
    #    occlusion = torch.zeros((1, H, W), device=flow.device)
#
    #    if numpy_out:
    #        flow = ensure_numpy(flow)
    #        occlusion = ensure_numpy(occlusion)
#
    #    return flow, occlusion

# in your WAFT wrapper class
    def compute_flow(self, im1, im2, mode='flow', init_flow=None, numpy_out=False, **kwargs):
        """
        Compute WAFT flow between two frames (MFT-compatible).
        MFT may pass init_flow; WAFT ignores it.
        Returns: (flow(2,H,W), {'occlusion': (1,H,W), 'sigma': (1,H,W)})
        """
        import torch, einops, cv2, numpy as np

        # --- inputs are BGR uint8 (H,W,3) per MFT contract ---
        H, W = im1.shape[:2]
        # to CHW RGB float32 torch
        image1 = einops.rearrange(
            torch.from_numpy(im1[:, :, ::-1].copy()),  # BGR→RGB
            "H W C -> 1 C H W", C=3).float().to(self.device)
        image2 = einops.rearrange(
            torch.from_numpy(im2[:, :, ::-1].copy()),
            "H W C -> 1 C H W", C=3).float().to(self.device)
        
        with torch.no_grad():
            out = self.model.calc_flow(image1, image2)

            # === DEBUG: Inspect WAFT outputs ===
            if isinstance(out, dict):
                if "info" in out:
                    info_last = out["info"][-1][0]  # (4, H, W)
                    for i in range(info_last.shape[0]):
                        ch = info_last[i]
                        print(f"  Channel {i}: min={ch.min().item():.3f}, max={ch.max().item():.3f}, mean={ch.mean().item():.3f}")

                    # Optional: save normalized info channels for inspection
                    debug_dir = "./debug_waft_info"
                    import os
                    os.makedirs(debug_dir, exist_ok=True)
                    for i in range(info_last.shape[0]):
                        ch_np = info_last[i].detach().cpu().numpy()
                        ch_norm = (ch_np - ch_np.min()) / (ch_np.max() - ch_np.min() + 1e-6)
                        cv2.imwrite(os.path.join(debug_dir, f"waft_info_ch{i}.png"), (ch_norm * 255).astype(np.uint8))

            # === Extract flow ===
            flow = out["flow"][-1][0]  # (2, H, W)
            assert flow.shape[1:] == (H, W)

            # === Interpret WAFT info channels if present ===
            if "info" in out:
                info = out["info"][-1][0]  # (4, H, W)

                # Channel mapping: based on your stats
                sigma_raw = info[0:1]      # Channel 0: inverse confidence
                occlusion_raw = info[3:4]  # Channel 3: occlusion-like mask

                # Transform into RAFT-style uncertainty & occlusion
                sigma = torch.exp(-torch.clamp(sigma_raw, min=-12, max=12))  # lower → more confident
                occlusion = torch.sigmoid(occlusion_raw)                     # 0–1 probability mask
            else:
                # Fallback if no info provided
                occlusion = torch.zeros((1, H, W), device=flow.device, dtype=flow.dtype)
                sigma = torch.full((1, H, W), 1e-3, device=flow.device, dtype=flow.dtype)

        # === Optional: numpy outputs (for debug only) ===
        if numpy_out:
            flow_np = flow.detach().cpu().numpy()
            occ_np  = occlusion.detach().cpu().numpy()
            sig_np  = sigma.detach().cpu().numpy()
            return flow_np, {'occlusion': occ_np, 'sigma': sig_np}

        return flow, {'occlusion': occlusion, 'sigma': sigma}
    