# SurgT/src/MFT_classic_leftonly.py (patched)
from csv import writer
import cv2
from curses import meta
from pathlib import Path
import torch
import numpy as np
from MFT_WAFT.MFT.config_RAFT import load_config
from agardiner_STIR_submission.MFT_WAFT.MFT.point_tracking import convert_to_point_tracking  # <<< use this


CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_RAFT_cfg.py")

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



def save_occlusion_png(occl_map_1hw: torch.Tensor, out_dir: Path, frame_i: int,
                       also_color: bool = False):
    """
    occl_map_1hw: torch (1,H,W) or (H,W), expected in [0,1]
    Saves:
      occ_XXXXXX.png (uint8 grayscale)
      occ_XXXXXX_color.png (optional colormap)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    occ = occl_map_1hw
    if occ is None:
        return
    if torch.is_tensor(occ):
        occ = occ.detach().float().cpu()
        if occ.ndim == 3:
            occ = occ[0]
        occ = occ.clamp(0, 1).numpy()
    else:
        # if already numpy
        if occ.ndim == 3:
            occ = occ[0]
        occ = np.clip(occ, 0, 1)

    occ_u8 = (occ * 255.0 + 0.5).astype(np.uint8)  # (H,W)

    p_gray = out_dir / f"occ_{frame_i:06d}.png"
    cv2.imwrite(str(p_gray), occ_u8)

    if also_color:
        occ_color = cv2.applyColorMap(occ_u8, cv2.COLORMAP_TURBO)
        p_col = out_dir / f"occ_{frame_i:06d}_color.png"
        cv2.imwrite(str(p_col), occ_color)

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


        meta = self.tracker.track(im1)

        flow_result = meta.result if hasattr(meta, "result") else meta


        coords_np, occ_np = convert_to_point_tracking(meta.result, self.init_queries)
        print("tracker:",self.tracker.track)
        # Optionally filter occluded points (keep visible ones if available)
        vis_mask = (occ_np < 0.5)
        if vis_mask.any():
            coords_use = coords_np[vis_mask]
        else:
            coords_use = coords_np  # if all occluded, fall back to all points

        H, W = im1.shape[:2]
        bbox_pred = self._points_to_bbox_xywh(coords_use, H, W)
        #return bbox_pred

        print(f"[DEBUG] flow_result.flow shape: {flow_result.flow.shape}")

        
        init_queries = self.init_queries.to(flow_result.flow.device)

        new_coords = flow_result.warp_forward_points(init_queries)
        new_coords_np = new_coords.detach().cpu().numpy()

        # ---- DELTA@PTS (mode delta over tracked points) ----
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

        if not hasattr(self, "frame_i"):
            self.frame_i = 0
        self.frame_i += 1        

        center = init_queries.mean(dim=0, keepdim=True)
        center_new = new_coords.mean(dim=0, keepdim=True)
        print(f"[DEBUG] init center: {center[0].tolist()} -> {center_new[0].tolist()} "
              f" (Δ = {(center_new - center)[0].tolist()})")

        H, W = im1.shape[:2]
        bbox_pred = self._points_to_bbox_xywh(new_coords_np, H, W)
        self.last_bbox = bbox_pred

        occl_map = getattr(flow_result, "occlusion", None)
        sigma_map = getattr(flow_result, "sigma", None)
        
        if not hasattr(self, "occ_out_dir"):
            self.occ_out_dir = Path("./occ_pngs_1")  # change path as desired

        save_occlusion_png(flow_result.occlusion, self.occ_out_dir, self.frame_i, also_color=True)

        if torch.is_tensor(occl_map):
            occl_map = occl_map.detach().cpu()
        if torch.is_tensor(sigma_map):
            sigma_map = sigma_map.detach().cpu()

        x, y, w, h = bbox_pred
        cx = float(x + 0.5 * w)
        cy = float(y + 0.5 * h)
        pt = torch.tensor([[cx, cy]], device=flow_result.flow.device, dtype=torch.float32)



        return {
           "bbox": bbox_pred,
           "occlusion": occl_map,
           "sigma": sigma_map,
           "flow_result": flow_result
        }  