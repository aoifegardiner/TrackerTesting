from pathlib import Path
import torch
import numpy as np
from MFT_WAFT.MFT.config import load_config

CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")

import sys, importlib, os

# Ensure our local WAFT implementation is used instead of the installed one
local_waft_path = "/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT"
if local_waft_path not in sys.path:
    sys.path.insert(0, local_waft_path)
print("[DEBUG] sys.path[0] set to:", sys.path[0])

# Force-load your local WAFT module now, before MFT tries to import it
waft_wrapper_path = os.path.join(local_waft_path, "waft.py")
spec = importlib.util.spec_from_file_location("MFT.waft", waft_wrapper_path)
waft_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(waft_module)
sys.modules["MFT.waft"] = waft_module

import inspect
print("[DEBUG] Forced WAFTWrapper loaded from:", inspect.getfile(waft_module))

class MFTWAFTTrackerSurgT:
    def __init__(self, im1, bbox1_gt, steps=5):
        cfg = load_config(CONFIG)
        import MFT_WAFT
        import importlib

        waft_mod = importlib.import_module("MFT_WAFT.MFT.WAFT.config.WAFT_cfg")
        #print("WAFTConfig loaded from:", waft_mod.__file__)
        #print("Has of_class? ", hasattr(waft_mod.WAFTConfig, "of_class"))
        #print(f"[DEBUG] cfg type: {type(cfg).__name__}")
        #print(f"[DEBUG] tracker_class: {cfg.tracker_class.__name__}")
        #print(f"[DEBUG] flow_config type: {type(getattr(cfg, 'flow_config', None)).__name__}")
        #print(f"[DEBUG] has of_class? {'of_class' in dir(getattr(cfg, 'flow_config', object))}")



        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize tracker ONCE
        self.tracker = cfg.tracker_class(cfg)
        self.tracker.init(im1)
        self.initialized = True

        # Fixed query points from GT bbox
        self.init_queries = self._bbox_to_queries(bbox1_gt, steps).to(self.device)
        self.last_bbox = bbox1_gt

    def _bbox_to_queries(self, bbox, steps=5):
        x, y, w, h = bbox
        xs = torch.linspace(x, x + w - 1, steps=steps)
        ys = torch.linspace(y, y + h - 1, steps=steps)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([xv.flatten(), yv.flatten()], dim=-1).float()

    def _points_to_bbox_xywh(self, pts, H, W):
        x_min = pts[:, 0].min().item(); x_max = pts[:, 0].max().item()
        y_min = pts[:, 1].min().item(); y_max = pts[:, 1].max().item()
        x = max(0, min(W - 1, x_min)); y = max(0, min(H - 1, y_min))
        w = max(1, min(W - x, x_max - x_min)); h = max(1, min(H - y, y_max - y_min))
        return [x, y, w, h]

    def tracker_update(self, im1):
        """
        Temporal update — accumulates flow from initial frame → current frame.
        """

        print(f"[DEBUG] tracker type: {type(self.tracker).__name__}")

        meta = self.tracker.track(im1)

        # Some configs return a namespace (with .result), some return FlowOUTrackingResult directly
        flow_result = meta.result if hasattr(meta, "result") else meta

        # Warp the initial queries through current cumulative flow
        coords_np = flow_result.warp_forward_points(self.init_queries).detach().cpu().numpy()

        # === DEBUG diagnostic ===
        if not hasattr(self, "_prev_coords"):
            self._prev_coords = coords_np.copy()
        else:
            delta_from_prev = np.mean(coords_np - self._prev_coords, axis=0)
            delta_from_init = np.mean(coords_np - self.init_queries.cpu().numpy(), axis=0)
            #print(f"[DEBUG] Mean Δ(prev): {delta_from_prev},  Mean Δ(init): {delta_from_init}")
            self._prev_coords = coords_np.copy()
        # === end DEBUG ===
        


        # Warp initial query points through cumulative flow
        H, W = im1.shape[:2]
        bbox_pred = self._points_to_bbox_xywh(coords_np, H, W)
        self.last_bbox = bbox_pred
        return bbox_pred
