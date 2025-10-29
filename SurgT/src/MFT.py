# SurgT/src/MFT_classic.py
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F

from MFT_WAFT.MFT.config_RAFT import load_config

CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_RAFT_cfg.py")

class RAFTFlowAdapter:
    def __init__(self, flow_tensor, device):
        # Accept (H,W,2), (2,H,W), (1,2,H,W) or numpy
        if isinstance(flow_tensor, np.ndarray):
            flow_tensor = torch.from_numpy(flow_tensor)

        if flow_tensor.ndim == 4:
            # (B,2,H,W) -> (H,W,2)
            flow_tensor = flow_tensor[0].permute(1, 2, 0).contiguous()

        elif flow_tensor.ndim == 3:
            if flow_tensor.shape[0] == 2:         # (2,H,W)
                flow_tensor = flow_tensor.permute(1, 2, 0).contiguous()  # -> (H,W,2)
            elif flow_tensor.shape[2] == 2:       # (H,W,2)
                pass
            else:
                raise ValueError(f"Unexpected flow shape: {tuple(flow_tensor.shape)}")

        else:
            raise ValueError(f"Unexpected flow shape: {tuple(flow_tensor.shape)}")

        self.flow = flow_tensor.to(device)  # (H, W, 2)


    @torch.no_grad()
    def warp_forward_points(self, points_xy: torch.Tensor) -> torch.Tensor:
        H, W, _ = self.flow.shape
        device = points_xy.device
        # normalize to [-1,1]
        gx = (points_xy[:, 0] / (W - 1)) * 2 - 1
        gy = (points_xy[:, 1] / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2).to(device)  # (1,1,N,2)
        flow_4d = self.flow.permute(2, 0, 1).unsqueeze(0)  # (1,2,H,W)
        sampled = F.grid_sample(flow_4d, grid, align_corners=True)  # (1,2,1,N)
        sampled = sampled.squeeze(0).squeeze(1).permute(1, 0).contiguous()  # (N,2)
        return points_xy + sampled


class MFTRAFTTrackerSurgT:
    def __init__(self, im1, im2, bbox1_gt, bbox2_gt, steps=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = load_config(CONFIG)

        # Two independent classic-MFT trackers (left/right)
        self.left = self.cfg.tracker_class(self.cfg)
        self.right = self.cfg.tracker_class(self.cfg)

        # IMPORTANT: initialize temporal state with the anchor frames
        self.left.init(im1)    # prev_left = im1@t
        self.right.init(im2)   # prev_right = im2@t

        # Query grids inside GT boxes
        self.pts_left  = self._bbox_to_queries(bbox1_gt, steps=steps) if bbox1_gt is not None else None
        self.pts_right = self._bbox_to_queries(bbox2_gt, steps=steps) if bbox2_gt is not None else None

        self.last_bbox_left = bbox1_gt
        self.last_bbox_right = bbox2_gt

    def _bbox_to_queries(self, bbox, steps=5):
        x, y, w, h = bbox
        xs = torch.linspace(x, x + w - 1, steps=steps)
        ys = torch.linspace(y, y + h - 1, steps=steps)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([xv.flatten(), yv.flatten()], dim=-1).float().to(self.device)

    def _points_to_bbox_xywh(self, pts, H, W):
        x_min = pts[:, 0].min().item(); x_max = pts[:, 0].max().item()
        y_min = pts[:, 1].min().item(); y_max = pts[:, 1].max().item()
        x = max(0, min(W - 1, x_min));  y = max(0, min(H - 1, y_min))
        w = max(1, min(W - x, x_max - x_min)); h = max(1, min(H - y, y_max - y_min))
        return [x, y, w, h]

    def _to_adapter(self, flow_obj):
        """Normalize different MFT-RAFT return types to RAFTFlowAdapter."""
        # WAFT-like object
        if hasattr(flow_obj, "warp_forward_points"):
            return flow_obj  # already usable

        # Namespaced result (common in classic MFT RAFT)
        if hasattr(flow_obj, "result") and hasattr(flow_obj.result, "flow"):
            return RAFTFlowAdapter(flow_obj.result.flow, self.device)

        # Direct attributes
        if hasattr(flow_obj, "flow"):
            return RAFTFlowAdapter(flow_obj.flow, self.device)

        # Raw tensor
        if torch.is_tensor(flow_obj) or isinstance(flow_obj, np.ndarray):
            return RAFTFlowAdapter(flow_obj, self.device)

        raise TypeError(f"Unrecognized flow object: {type(flow_obj)} (keys={getattr(flow_obj, '__dict__', {}).keys()})")

    def tracker_update(self, im1, im2):
        bbox_pred_left, bbox_pred_right = None, None

        # LEFT (temporal: prev_left → im1)
        if self.pts_left is not None:
            flow_left = self.left.track(im1, self.pts_left)           # MFT computes flow(prev_left, im1) internally
            adapter_l = self._to_adapter(flow_left)
            new_left = adapter_l.warp_forward_points(self.pts_left)
            # quick sanity print
            #print(f"[DEBUG:left Δ] mean|Δ|={(new_left - self.pts_left).abs().mean().item():.4f}")
            self.pts_left = new_left
            H1, W1 = im1.shape[:2]
            bbox_pred_left = self._points_to_bbox_xywh(self.pts_left, H1, W1)
            self.last_bbox_left = bbox_pred_left

        # RIGHT (temporal: prev_right → im2)
        if self.pts_right is not None:
            flow_right = self.right.track(im2, self.pts_right)
            adapter_r = self._to_adapter(flow_right)
            new_right = adapter_r.warp_forward_points(self.pts_right)
            #print(f"[DEBUG:right Δ] mean|Δ|={(new_right - self.pts_right).abs().mean().item():.4f}")
            self.pts_right = new_right
            H2, W2 = im2.shape[:2]
            bbox_pred_right = self._points_to_bbox_xywh(self.pts_right, H2, W2)
            self.last_bbox_right = bbox_pred_right

        #print(f"[DEBUG:update] Pred bbox left={bbox_pred_left}, right={bbox_pred_right}")
        return bbox_pred_left, bbox_pred_right
