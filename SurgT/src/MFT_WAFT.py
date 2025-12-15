# SurgT/src/MFT_WAFT.py
from pathlib import Path
import torch
import numpy as np

from MFT_WAFT.MFT.config import load_config

CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")

class MFTWAFTTrackerSurgT:
    """
    SurgT expects:
      __init__(im1, im2, bbox1_gt, bbox2_gt)  # called at an anchor frame (time t)
      tracker_update(im1, im2)                 # called at later frames (t+1, t+2, ...)
    We will:
      - run a separate temporal WAFT tracker on the LEFT stream (im1_t -> im1_{t+1})
      - run a separate temporal WAFT tracker on the RIGHT stream (im2_t -> im2_{t+1})
      - track query points sampled from each GT bbox, then re-box the warped points
    """

    def __init__(self, im1, im2, bbox1_gt=None, bbox2_gt=None, steps=5):
        print("Calling load_config with:", CONFIG)
        cfg = load_config(CONFIG)
        print("Loaded config:", cfg)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Two **independent** WAFT trackers so their prev_frame states don’t collide
        self.left = cfg.tracker_class(cfg)   # temporal tracker for LEFT stream
        self.right = cfg.tracker_class(cfg)  # temporal tracker for RIGHT stream

        # Initialize temporal state with the current stereo pair at the anchor frame (time t)
        self.left.init(im1)   # prev_frame(left) = im1_t
        self.right.init(im2)  # prev_frame(right) = im2_t

        # Sample query grids from each GT bbox (x,y,w,h) at the anchor
        self.pts_left  = self._bbox_to_queries(bbox1_gt, steps=steps) if bbox1_gt is not None else None
        self.pts_right = self._bbox_to_queries(bbox2_gt, steps=steps) if bbox2_gt is not None else None

        # Keep last predicted boxes around (optional, for debugging)
        self.last_bbox_left = bbox1_gt
        self.last_bbox_right = bbox2_gt


        # Debug
        #print("[DEBUG:init_bbox_gt]", bbox1_gt, bbox2_gt)
        #if self.pts_left is not None:
        #    print("[DEBUG:init1] Initial query points (first 5):",
        #          self.pts_left[:5].detach().cpu().numpy())
        #if self.pts_right is not None:
        #    print("[DEBUG:init2] Initial query points (first 5):",
        #          self.pts_right[:5].detach().cpu().numpy())

    def _bbox_to_queries(self, bbox, steps=5):
        """ bbox is (x, y, w, h) with top-left origin; sample a steps×steps grid. """
        if bbox is None:
            return None
        x, y, w, h = bbox
        # stay inside the box: go to x+w-1, y+h-1
        xs = torch.linspace(x, x + w - 1, steps=steps)
        ys = torch.linspace(y, y + h - 1, steps=steps)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
        points = torch.stack([xv.flatten(), yv.flatten()], dim=-1).float().to(self.device)
        #print("[DEBUG:bbox_to_queries] bbox:", bbox)
        #print("[DEBUG:bbox_to_queries] first 5 queries:", points[:5].detach().cpu().numpy())
        return points

    def _points_to_bbox_xywh(self, pts, H, W):
        """ Tight [x,y,w,h] box around a set of (N,2) points, clamped to image. """
        x_min = pts[:, 0].min().item()
        x_max = pts[:, 0].max().item()
        y_min = pts[:, 1].min().item()
        y_max = pts[:, 1].max().item()
        x = max(0, min(W - 1, x_min))
        y = max(0, min(H - 1, y_min))
        w = max(1, min(W - x, x_max - x_min))
        h = max(1, min(H - y, y_max - y_min))
        return [x, y, w, h]

    def tracker_update(self, im1, im2):
        """
        Called each subsequent frame.
        IMPORTANT: im1 is LEFT_t, im2 is RIGHT_t at the SAME time index.
        We use **temporal** updates per stream:
          left:  prev_left -> im1  (queries = left points)
          right: prev_right -> im2 (queries = right points)
        WAFT wrappers keep their own prev_frame internally.
        """
        # LEFT stream temporal update
        bbox_pred_left = None
        if self.pts_left is not None:
            flow_left = self.left.track(im1, self.pts_left)   # temporal: prev_left -> im1
            self.pts_left = flow_left.warp_forward_points(self.pts_left)
            H1, W1 = im1.shape[:2]
            bbox_pred_left = self._points_to_bbox_xywh(self.pts_left, H1, W1)
            self.last_bbox_left = bbox_pred_left
            #print(f"[DEBUG:update] Predicted LEFT bbox: {bbox_pred_left}")

        # RIGHT stream temporal update
        bbox_pred_right = None
        if self.pts_right is not None:
            flow_right = self.right.track(im2, self.pts_right)  # temporal: prev_right -> im2
            self.pts_right = flow_right.warp_forward_points(self.pts_right)
            H2, W2 = im2.shape[:2]
            bbox_pred_right = self._points_to_bbox_xywh(self.pts_right, H2, W2)
            self.last_bbox_right = bbox_pred_right
            #print(f"[DEBUG:update] Predicted RIGHT bbox: {bbox_pred_right}")

        # Return both bboxes (left for bbox1, right for bbox2) → matches SurgT’s expectations
        return bbox_pred_left, bbox_pred_right
