# SurgT/src/MFT_classic_leftonly.py (patched)
from pathlib import Path
import torch
import numpy as np
from MFT_WAFT.MFT.config_RAFT import load_config
from MFT.point_tracking import convert_to_point_tracking  # <<< use this

CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_RAFT_cfg.py")

class MFTRAFTTrackerSurgT:
    """
    Monocular (left-only) temporal MFT tracker.
    IMPORTANT: keep queries in INIT frame coords; do NOT update them frame-to-frame.
    """
    def __init__(self, im1, bbox1_gt, steps=5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = load_config(CONFIG)

        # Single MFT tracker
        self.tracker = self.cfg.tracker_class(self.cfg)

        # MFT expects np.uint8 BGR HxWxC (OpenCV). im1 from Video is already BGR uint8.
        self.tracker.init(im1)

        # Define queries inside bbox (INIT FRAME COORDS) — keep these FIXED
        self.init_queries = self._bbox_to_queries(bbox1_gt, steps=steps) if bbox1_gt is not None else None
        if self.init_queries is not None:
            # MFT utils expect torch float on same device; do NOT divide by any scale unless you resized images
            self.init_queries = self.init_queries.float().to(self.device)

    def _bbox_to_queries(self, bbox, steps=5):
        x, y, w, h = bbox
        xs = torch.linspace(x, x + w - 1, steps=steps)
        ys = torch.linspace(y, y + h - 1, steps=steps)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([xv.flatten(), yv.flatten()], dim=-1)

    def _points_to_bbox_xywh(self, pts_np, H, W):
        # pts_np: (N,2) numpy
        x_min = max(0, min(W - 1, float(pts_np[:, 0].min())))
        x_max = max(0, min(W - 1, float(pts_np[:, 0].max())))
        y_min = max(0, min(H - 1, float(pts_np[:, 1].min())))
        y_max = max(0, min(H - 1, float(pts_np[:, 1].max())))
        w = max(1.0, x_max - x_min)
        h = max(1.0, y_max - y_min)
        return [x_min, y_min, w, h]

    def tracker_update(self, im1):
        """
        Track to the next frame and return a bbox built from the current coords
        obtained by warping the FIXED init queries with the ACCUMULATED flow.
        """
        if self.init_queries is None:
            return None

        meta = self.tracker.track(im1)  # accumulated flow init->current

        # Convert flow to current per-point locations; DO NOT update stored queries
        coords_np, occ_np = convert_to_point_tracking(meta.result, self.init_queries)

        # Optionally filter occluded points (keep visible ones if available)
        vis_mask = (occ_np < 0.5)
        if vis_mask.any():
            coords_use = coords_np[vis_mask]
        else:
            coords_use = coords_np  # if all occluded, fall back to all points

        H, W = im1.shape[:2]
        bbox_pred = self._points_to_bbox_xywh(coords_use, H, W)
        return bbox_pred
