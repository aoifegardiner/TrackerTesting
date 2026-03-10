from pathlib import Path
import torch
import numpy as np
from agardiner_waft.MFT_WAFT.MFT.config import load_config
#from agardiner_waft.MFT_WAFT.MFT.point_tracking import convert_to_point_tracking

#CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")
CONFIG = Path("/Workspace/agardiner_waft/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")

import sys, importlib, os

# Ensure our local WAFT implementation is used instead of the installed one
#local_waft_path = "/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT"
local_waft_path = "/Workspace/agardiner_waft/MFT_WAFT/MFT"
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
        print("[MFTWAFT] Loading config:", CONFIG)
        cfg = load_config(CONFIG)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize tracker ONCE
        self.tracker = cfg.tracker_class(cfg)
        print(f"[MFTWAFT] Initialized tracker: {type(self.tracker).__name__}")

        self.tracker.init(im1)
        self.initialized = True

        # Fixed query points from GT bbox
        self.pts0 = self._bbox_to_queries(bbox1_gt, steps).to(self.device)
        self.pts = self.pts0.clone()
        self.last_bbox = bbox1_gt

#    def __init__(self, im1, bbox1_gt, steps=5):
#        print("Calling load_config with:", CONFIG)
#        cfg = load_config(CONFIG)
#        print("Loaded config:", cfg)
#        import MFT_WAFT
#        import importlib
#
#        waft_mod = importlib.import_module("MFT_WAFT.MFT.WAFT.config.WAFT_cfg")
#        #print("WAFTConfig loaded from:", waft_mod.__file__)
#        #print("Has of_class? ", hasattr(waft_mod.WAFTConfig, "of_class"))
#        #print(f"[DEBUG] cfg type: {type(cfg).__name__}")
#        #print(f"[DEBUG] tracker_class: {cfg.tracker_class.__name__}")
#        #print(f"[DEBUG] flow_config type: {type(getattr(cfg, 'flow_config', None)).__name__}")
#        #print(f"[DEBUG] has of_class? {'of_class' in dir(getattr(cfg, 'flow_config', object))}")
#
#
#
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#
#        # Initialize tracker ONCE
#        #raw = cfg.tracker_class(cfg)
#        self.tracker = cfg.tracker_class(cfg)
#        print(f"[DEBUG] Initialized tracker of type: {type(self.tracker).__name__}")
#        self.tracker.init(im1)
#        self.initialized = True
#
#        # Fixed query points from GT bbox
#        self.init_queries = self._bbox_to_queries(bbox1_gt, steps).to(self.device)
#        self.last_bbox = bbox1_gt

#    def _bbox_to_queries(self, bbox, steps=5):
#        x, y, w, h = bbox
#        xs = torch.linspace(x, x + w - 1, steps=steps)
#        ys = torch.linspace(y, y + h - 1, steps=steps)
#        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
#        return torch.stack([xv.flatten(), yv.flatten()], dim=-1).float()

#    def _points_to_bbox_xywh(self, pts, H, W):
#        x_min = pts[:, 0].min().item(); x_max = pts[:, 0].max().item()
#        y_min = pts[:, 1].min().item(); y_max = pts[:, 1].max().item()
#        x = max(0, min(W - 1, x_min)); y = max(0, min(H - 1, y_min))
#        w = max(1, min(W - x, x_max - x_min)); h = max(1, min(H - y, y_max - y_min))
#        return [x, y, w, h]
    
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




    @staticmethod
    def _sample_flow_bilinear(flow_2hw: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
        import torch
        import torch.nn.functional as F
        import numpy as np
        """
        flow_2hw: (2,H,W) in pixels
        pts_xy:   (N,2) in pixel coords (x,y)
        returns:  (N,2) flow vectors (dx,dy) at those points
        """
        assert flow_2hw.dim() == 3 and flow_2hw.shape[0] == 2
        H, W = flow_2hw.shape[1], flow_2hw.shape[2]

        # grid_sample expects normalized coords in [-1,1] with (x,y) order
        x = pts_xy[:, 0]
        y = pts_xy[:, 1]
        gx = (x / (W - 1)) * 2 - 1
        gy = (y / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2)  # (1,N,1,2)

        flow = flow_2hw.unsqueeze(0)  # (1,2,H,W)
        samp = F.grid_sample(flow, grid, mode="bilinear", padding_mode="border", align_corners=True)
        samp = samp.view(2, -1).transpose(0, 1)  # (N,2)
        return samp

    @staticmethod
    def _bbox_xywh_to_corners_xy(bbox_xywh: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        import numpy as np
        x, y, w, h = bbox_xywh.tolist()
        return np.array([
            [x,     y],
            [x + w, y],
            [x,     y + h],
            [x + w, y + h],
        ], dtype=np.float32)

    @staticmethod
    def _corners_xy_to_bbox_xywh(corners_xy: np.ndarray, H: int, W: int) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        import numpy as np
        xs = corners_xy[:, 0]
        ys = corners_xy[:, 1]
        x0 = float(np.clip(xs.min(), 0, W - 1))
        y0 = float(np.clip(ys.min(), 0, H - 1))
        x1 = float(np.clip(xs.max(), 0, W - 1))
        y1 = float(np.clip(ys.max(), 0, H - 1))
        return np.array([x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)], dtype=np.float32)

    def _bbox_to_queries(self, bbox, steps=5):
        """
        Uniform grid of query points inside bbox.
        """
        x, y, w, h = bbox
        xs = torch.linspace(x, x + w - 1, steps=steps)
        ys = torch.linspace(y, y + h - 1, steps=steps)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([xv.flatten(), yv.flatten()], dim=-1).float()

    def _points_to_bbox_xywh(self, pts, H, W):
        """
        Convert warped points → bounding box.
        """
        x_min = pts[:, 0].min().item()
        x_max = pts[:, 0].max().item()
        y_min = pts[:, 1].min().item()
        y_max = pts[:, 1].max().item()

        x = max(0, min(W - 1, x_min))
        y = max(0, min(H - 1, y_min))
        w = max(1, min(W - x, x_max - x_min))
        h = max(1, min(H - y, y_max - y_min))

        return [x, y, w, h]

    @torch.no_grad()
    def tracker_update(self, im1):
        meta = self.tracker.track(im1)
        flow_result = meta.result if hasattr(meta, "result") else meta

        # initialize current points once
        if not hasattr(self, "pts"):
            self.pts = self.init_queries.clone().to(flow_result.flow.device)

        old_pts = self.pts
        new_pts = flow_result.warp_forward_points(old_pts)

        # sanity check: should NOT be 0 if things are moving
        disp = (new_pts - old_pts).norm(dim=1)
        print("[WARP CHECK] mean|Δ|=", disp.mean().item(), " max|Δ|=", disp.max().item())

        # persist update (this is what your “working” code wasn’t doing)
        self.pts = new_pts.detach()

        H, W = im1.shape[:2]
        bbox_pred = self._points_to_bbox_xywh(new_pts.detach().cpu().numpy(), H, W)
        self.last_bbox = bbox_pred

        flow = flow_result.flow  # (2,H,W)

        print("[PTS] mean xy =", self.pts.mean(dim=0).tolist(),
              "std xy =", self.pts.std(dim=0).tolist())
        
        
       
        # global mean flow
        g = flow.view(2, -1).mean(dim=1)

        # local flow at bbox center
        cx = float(bbox_pred[0] + bbox_pred[2]/2)
        cy = float(bbox_pred[1] + bbox_pred[3]/2)
        p = torch.tensor([[cx, cy]], device=flow.device)
        dxy_c = self._sample_flow_bilinear(flow, p)[0]

        print(f"[FLOW CHECK] global={g.tolist()} center={dxy_c.tolist()} diff={(dxy_c-g).tolist()}")

        # spread of flow over your tracked points
        dxy_pts = self._sample_flow_bilinear(flow, self.pts)
        print(f"[PTS FLOW] std(dx,dy)=({dxy_pts[:,0].std().item():.4f},{dxy_pts[:,1].std().item():.4f}) "
              f"mean={dxy_pts.mean(dim=0).tolist()}")

        return {
            "bbox": bbox_pred,
            "occlusion": getattr(flow_result, "occlusion", None),
            "sigma": getattr(flow_result, "sigma", None),   # <— use sigma, not "uncertainty"
        }



#    def tracker_update(self, im1):
#        """
#        Temporal update — accumulates flow from initial frame → current frame.
#        """
#
#        print(f"[DEBUG] tracker type: {type(self.tracker).__name__}")
#
#        meta = self.tracker.track(im1)
#        print(f"[DEBUG] track() returned type: {type(meta).__name__}")
#        # Some configs return a namespace (with .result), some return FlowOUTrackingResult directly
#        flow_result = meta.result if hasattr(meta, "result") else meta
#        print(f"[DEBUG] flow_result type: {type(flow_result).__name__}")
#        print(f"[DEBUG] flow_result has attributes: {dir(flow_result)}")
#        print(f"[DEBUG] flow_result.flow shape: {flow_result.flow.shape}")
#        # Warp the initial queries through current cumulative flow
#        coords_np = flow_result.warp_forward_points(self.init_queries).detach().cpu().numpy()
#
#        # === DEBUG diagnostic ===
#        if not hasattr(self, "_prev_coords"):
#            self._prev_coords = coords_np.copy()
#        else:
#            delta_from_prev = np.mean(coords_np - self._prev_coords, axis=0)
#            delta_from_init = np.mean(coords_np - self.init_queries.cpu().numpy(), axis=0)
#            #print(f"[DEBUG] Mean Δ(prev): {delta_from_prev},  Mean Δ(init): {delta_from_init}")
#            self._prev_coords = coords_np.copy()
#        # === end DEBUG ===
#        
#
#        # Warp initial query points through cumulative flow
#        H, W = im1.shape[:2]
#        bbox_pred = self._points_to_bbox_xywh(coords_np, H, W)
#        self.last_bbox = bbox_pred
#        return bbox_pred
    
#    def tracker_update(self, im1):
#        """
#        Temporal update — accumulate flow incrementally.
#        """
#
#        print(f"[DEBUG] tracker type: {type(self.tracker).__name__}")
#
#        meta = self.tracker.track(im1)
#        flow_result = meta.result if hasattr(meta, "result") else meta
#        
#
#        print(f"[DEBUG] flow_result.flow shape: {flow_result.flow.shape}")
#        # Before warping
#        #center = self.init_queries.mean(dim=0, keepdim=True)  # (1,2)
#        #flow_at_center = flow_result.warp_forward_points(center)
#        #print(f"[DEBUG] init center: {center[0].tolist()} -> {flow_at_center[0].tolist()}  (Δ = {(flow_at_center - center)[0].tolist()})")
#        
#        # --------------------------------------------------------
#        #  FIX: accumulate flow INCREMENTALLY, not from init frame
#        # --------------------------------------------------------
#        #if not hasattr(self, "prev_queries"):
#        #    # First frame after init
#        #    self.prev_queries = self.init_queries.clone().to(self.device)
#        
#        init_queries = self.init_queries.to(flow_result.flow.device)
#
#        new_coords = flow_result.warp_forward_points(init_queries)
#        new_coords_np = new_coords.detach().cpu().numpy()
#
#
#        center = init_queries.mean(dim=0, keepdim=True)
#        center_new = new_coords.mean(dim=0, keepdim=True)
#        print(f"[DEBUG] init center: {center[0].tolist()} -> {center_new[0].tolist()} "
#              f" (Δ = {(center_new - center)[0].tolist()})")
#
#        H, W = im1.shape[:2]
#        bbox_pred = self._points_to_bbox_xywh(new_coords_np, H, W)
#        self.last_bbox = bbox_pred
#
#
#        # Warp previous points using F(t→t+1)
#        #new_coords = flow_result.warp_forward_points(self.prev_queries)
#        #new_coords_np = new_coords.detach().cpu().numpy()
#
#        # Save for next iteration
#        #self.prev_queries = new_coords.detach()
#
#        #H, W = im1.shape[:2]
#        #bbox_pred = self._points_to_bbox_xywh(new_coords_np, H, W)
#        #self.last_bbox = bbox_pred
#
#        occl_map = getattr(flow_result, "occlusion", None)
#        uncert_map = getattr(flow_result, "uncertainty", None)
#
#        return {
#           "bbox": bbox_pred,
#           "occlusion": occl_map,
#           "uncertainty": uncert_map
#        }       
#    
#    def tracker_update(self, im1):
#        meta = self.tracker.track(im1)
#        flow_result = meta.result if hasattr(meta, "result") else meta
#
#        flow = flow_result.flow  # (2,H,W) tensor
#
#        showing_H, showing_W = im1.shape[:2]
#        assert flow.shape[1] == showing_H and flow.shape[2] == showing_W, \
#            f"flow shape {tuple(flow.shape)} != image {showing_H}x{showing_W}"
#
#        # --- warp bbox corners ---
#        corners = self._bbox_xywh_to_corners_xy(np.array(self.last_bbox, dtype=np.float32))
#        corners_t = torch.from_numpy(corners).to(flow.device)
#        global_flow = torch.median(
#            flow.view(2, -1), dim=1
#        ).values  # (2,)
#
#        flow_corrected = flow - global_flow.view(2, 1, 1)
#
#
#        dxy = self._sample_flow_bilinear(flow_corrected, corners_t)          # (4,2)
#        max_step = 8.0  # pixels per frame
#        dxy = torch.clamp(dxy, -max_step, max_step)
#        corners_new = corners_t + dxy
#        corners_new_np = corners_new.detach().cpu().numpy()
#
#        bbox_pred = self._corners_xy_to_bbox_xywh(corners_new_np, showing_H, showing_W)
#        self.last_bbox = bbox_pred.tolist()
#
#        # Debug: compare bbox center motion to GT-flow sample (you already print)
#        bx, by, bw, bh = bbox_pred
#
#        print(
#            "[FLOW CHECK]",
#            "global =", global_flow.tolist(),
#            "local =", dxy.mean(dim=0).tolist()
#        )
#        
#        print(f"[SANITY] bbox center ({bx+bw/2:.1f},{by+bh/2:.1f}) vs FLOW@GT pred center you printed earlier")
#
#        print(f"[DEBUG] bbox center = {(bx + bw/2):.2f}, {(by + bh/2):.2f} | "
#              f"mean dxy corners = {dxy.mean(dim=0).tolist()}")
#
#        return {
#            "bbox": self.last_bbox,
#            "flow": flow,  # optional, but helpful for your debug pipeline
#            "occlusion": getattr(flow_result, "occlusion", None),
#            "uncertainty": getattr(flow_result, "sigma", None),  # <-- in your code sigma is the uncertainty
#        }

