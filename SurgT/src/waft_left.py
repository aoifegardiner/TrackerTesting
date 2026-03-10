from curses import meta
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

@torch.no_grad()
def sample_map_at_points(map_1hw: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    if map_1hw is None:
        return None
    if map_1hw.ndim == 2:
        map_1hw = map_1hw.unsqueeze(0)  # (1,H,W)
    _, H, W = map_1hw.shape
    gx = (points_xy[:, 0] / (W - 1)) * 2 - 1
    gy = (points_xy[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)  # (1,1,N,2)
    m4 = map_1hw.unsqueeze(0)  # (1,1,H,W)
    samp = F.grid_sample(m4, grid, align_corners=True)      # (1,1,1,N)
    return samp.view(-1)                                    # (N,)

def sample_delta_indices_at_points(selected_delta_i_11hw: torch.Tensor,
                                   points_xy: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbour sampling of selected_delta_i at given points.

    selected_delta_i_11hw: (1,1,H,W) int64 (often stored on CPU)
    points_xy: (N,2) float tensor in pixel coords (can be on GPU)
    returns: (N,) int64 indices into meta.used_deltas
    """
    if selected_delta_i_11hw is None or points_xy is None:
        return None

    sdi = selected_delta_i_11hw
    if torch.is_tensor(sdi) and sdi.is_cuda:
        sdi = sdi.cpu()

    pts = points_xy.detach().cpu()
    H, W = sdi.shape[-2], sdi.shape[-1]

    xi = torch.round(pts[:, 0]).long().clamp(0, W - 1)
    yi = torch.round(pts[:, 1]).long().clamp(0, H - 1)

    return sdi[0, 0, yi, xi].long()  # (N,)

def mode_delta_from_indices(delta_idx: torch.Tensor, used_deltas) -> tuple:
    """
    delta_idx: (N,) int64 indices
    used_deltas: list of delta values (may include np.inf)

    returns: (mode_delta_value, mode_fraction, counts_tensor)
    """
    if delta_idx is None or delta_idx.numel() == 0:
        return None, None, None

    D = len(used_deltas)
    counts = torch.bincount(delta_idx, minlength=D)
    mode_i = int(torch.argmax(counts).item())
    mode_frac = float(counts[mode_i].item() / max(1, delta_idx.numel()))
    mode_val = used_deltas[mode_i]
    return mode_val, mode_frac, counts

class MFTWAFTTrackerSurgT:
    def __init__(self, im1, bbox1_gt, steps=5):
        print("Calling load_config with:", CONFIG)
        cfg = load_config(CONFIG)
        print("Loaded config:", cfg)
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
        #raw = cfg.tracker_class(cfg)
        self.tracker = cfg.tracker_class(cfg)
        print(f"[DEBUG] Initialized tracker of type: {type(self.tracker).__name__}")
        self.tracker.init(im1)
        self.initialized = True
        self.frame_i = 0

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

#    @staticmethod
#    def _sample_map_center_nn(map_1hw, cx, cy):
#        if map_1hw is None:
#            return None
#        if map_1hw.dim() == 4:   # (1,1,H,W)
#            map_1hw = map_1hw[0]
#        if map_1hw.dim() == 3:   # (1,H,W)
#            map_1hw = map_1hw[0]
#
#        H, W = map_1hw.shape
#        ix = int(round(cx))
#        iy = int(round(cy))
#        ix = max(0, min(W - 1, ix))
#        iy = max(0, min(H - 1, iy))
#        return float(map_1hw[iy, ix].item())


    @torch.inference_mode()
    def tracker_update(self, im1):
        """
        Temporal update — accumulate flow incrementally.
        """
        import os, csv
        # at top of tracker_update
        #print(f"[DEBUG] tracker type: {type(self.tracker).__name__}")

        meta = self.tracker.track(im1)

        flow_result = meta.result if hasattr(meta, "result") else meta

        print(f"[DEBUG] flow_result.flow shape: {flow_result.flow.shape}")
        # Before warping
        #center = self.init_queries.mean(dim=0, keepdim=True)  # (1,2)
        #flow_at_center = flow_result.warp_forward_points(center)
        #print(f"[DEBUG] init center: {center[0].tolist()} -> {flow_at_center[0].tolist()}  (Δ = {(flow_at_center - center)[0].tolist()})")
        
        # --------------------------------------------------------
        #  FIX: accumulate flow INCREMENTALLY, not from init frame
        # --------------------------------------------------------
        #if not hasattr(self, "prev_queries"):
        #    # First frame after init
        #    self.prev_queries = self.init_queries.clone().to(self.device)
        
        init_queries = self.init_queries.to(flow_result.flow.device)

#        occ_pts_iq = sample_map_at_points(flow_result.occlusion, init_queries)
#        sig_pts_iq = sample_map_at_points(flow_result.sigma, init_queries)
#        
#        if occ_pts_iq is not None:
#            occ_mean = occ_pts_iq.mean().item()
#            occ_med  = occ_pts_iq.median().item()
#            occ_frac = (occ_pts_iq > self.tracker.C.occlusion_threshold).float().mean().item()
#            print(f"[OCC@PTS INIT QUERIES] mean={occ_mean:.3f} med={occ_med:.3f} frac>thr={occ_frac:.3f}")
#
#        if sig_pts_iq is not None:
#            sig_mean = sig_pts_iq.mean().item()
#            sig_med  = sig_pts_iq.median().item()
#            sig_p90  = sig_pts_iq.kthvalue(int(0.9 * (sig_pts_iq.numel()-1)) + 1).values.item()
#            print(f"[SIG@PTS INIT QUERIES] mean={sig_mean:.3f} med={sig_med:.3f} p90={sig_p90:.3f}")


        new_coords = flow_result.warp_forward_points(init_queries)
        new_coords_np = new_coords.detach().cpu().numpy()

        delta_mode_val = None
        
        if hasattr(meta, "selected_delta_i") and hasattr(meta, "used_deltas"):
            delta_idx_pts = sample_delta_indices_at_points(meta.selected_delta_i, new_coords)
            if delta_idx_pts is not None:
                delta_mode_val = mode_delta_from_indices(delta_idx_pts, meta.used_deltas)
        
                print(f"[DELTA@PTS] mode={delta_mode_val} ")
        else:
            print("[DELTA@PTS] missing meta.selected_delta_i/meta.used_deltas (not stored in MFT.track)")



        occ_pts = sample_map_at_points(flow_result.occlusion, new_coords)
        sig_pts = sample_map_at_points(flow_result.sigma, new_coords)
        
        if occ_pts is not None:
            occ_mean = occ_pts.mean().item()
            occ_med  = occ_pts.median().item()
            occ_frac = (occ_pts > self.tracker.C.occlusion_threshold).float().mean().item()
            print(f"[OCC@PTS] mean={occ_mean:.3f} med={occ_med:.3f} frac>thr={occ_frac:.3f}")

        if sig_pts is not None:
            sig_mean = sig_pts.mean().item()
            sig_med  = sig_pts.median().item()
            sig_max  = sig_pts.max().item()
            sig_p90  = sig_pts.kthvalue(int(0.9 * (sig_pts.numel()-1)) + 1).values.item()
            print(f"[SIG@PTS] mean={sig_mean:.3f} med={sig_med:.3f} max={sig_max:.3f} p90={sig_p90:.3f}")

        


        center = init_queries.mean(dim=0, keepdim=True)
        center_new = new_coords.mean(dim=0, keepdim=True)
        print(f"[DEBUG] init center: {center[0].tolist()} -> {center_new[0].tolist()} "
              f" (Δ = {(center_new - center)[0].tolist()})")

        H, W = im1.shape[:2]
        bbox_pred = self._points_to_bbox_xywh(new_coords_np, H, W)
        self.last_bbox = bbox_pred

        occl_map = getattr(flow_result, "occlusion", None)
        sigma_map = getattr(flow_result, "sigma", None)
        
        if torch.is_tensor(occl_map):
            occl_map = occl_map.detach().cpu()
        if torch.is_tensor(sigma_map):
            sigma_map = sigma_map.detach().cpu()

        x, y, w, h = bbox_pred
        cx = float(x + 0.5 * w)
        cy = float(y + 0.5 * h)
        pt = torch.tensor([[cx, cy]], device=flow_result.flow.device, dtype=torch.float32)

        if not hasattr(self, "frame_i"):
            self.frame_i = 0
        self.frame_i += 1


        return {
           "bbox": bbox_pred,
           "occlusion": occl_map,
           "sigma": sigma_map,
           "flow_result": flow_result
        }  
    
        # Warp previous points using F(t→t+1)
        #new_coords = flow_result.warp_forward_points(self.prev_queries)
        #new_coords_np = new_coords.detach().cpu().numpy()

        # Save for next iteration
        #self.prev_queries = new_coords.detach()

        #H, W = im1.shape[:2]
        #bbox_pred = self._points_to_bbox_xywh(new_coords_np, H, W)
        #self.last_bbox = bbox_pred


        #selected_delta = None
        #if hasattr(meta, "selected_delta_i") and hasattr(meta, "used_deltas"):
        #    delta_map = meta.selected_delta_i.to(flow_result.flow.device).float()  # (1,1,H,W)
        #    idx = self._sample_map_bilinear(delta_map, pt)  # returns (1,)
        #    idx = int(torch.round(idx).item())
        #    idx = max(0, min(idx, len(meta.used_deltas) - 1))
        #    selected_delta = meta.used_deltas[idx]
        #    print(f"[CENTER SAMPLE] selected_delta={selected_delta} (idx={idx})")
        
#        occl_center = None
#        sigma_center = None

#        if occl_map is not None:
#            with torch.no_grad():
#                occl_center = float((occl_map, pt).item()) if occl_map is not None else None
#
#        if sigma_map is not None:            
#            with torch.no_grad():
#                sigma_center = float((sigma_map, pt).item()) if sigma_map is not None else None
#



#        log_dir = "./results/case1_video1_left_0802_1"
#        os.makedirs(log_dir, exist_ok=True)
#        csv_path = os.path.join(log_dir, "center_samples.csv")
#
#        write_header = not os.path.exists(csv_path)
#        
#        with open(csv_path, "a", newline="") as f:
#            w = csv.writer(f)
#            if write_header:
#                w.writerow([
#                    "frame", "cx", "cy",
#                    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
#                    "occ_center", "sigma_center"
#                ])
#            w.writerow([
#                getattr(self.tracker, "current_frame_i", self.frame_i),
#                cx, cy,
#                x, y, w, h,
#                occl_center, sigma_center
#            ])
#
#        #print(f"[CENTER SAMPLE] (cx,cy)=({cx:.1f},{cy:.1f})  occ={occl_center}  sigma={sigma_center}")
#        #occl_out = occl_map.detach().cpu() if occl_map is not None else None
        #sigma_out = sigma_map.detach().cpu() if sigma_map is not None else None

     
    
