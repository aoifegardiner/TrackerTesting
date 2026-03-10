import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

#from agardiner_waft.MFT_WAFT.MFT.config import load_config
from agardiner_STIR_submission.MFT_WAFT.MFT.config_RAFT import load_config




# ----------------------------
# User inputs
# ----------------------------
#CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")
#CONFIG = Path("/Workspace/agardiner_waft/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")
CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_RAFT_cfg.py")
VIDEO_PATH = Path("/public_datasets/STIRDataset/22/left/seq13/frames/00108280ms-00117800ms-visible.mp4")

POINT_XY = (1024, 595)   # center in init frame
BOX_WH   = (44, 44)      # initial box size in init frame
STEPS    = 5             # grid resolution inside bbox (steps x steps points)


# ----------------------------
# Helpers (matching your tracker)
# ----------------------------
@torch.no_grad()
def sample_map_at_points(map_1hw: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    if map_1hw is None or points_xy is None:
        return None

    if map_1hw.ndim == 2:
        map_1hw = map_1hw.unsqueeze(0)  # (1,H,W)

    # --- make sure points live on same device as map ---
    points_xy = points_xy.to(map_1hw.device)

    _, H, W = map_1hw.shape
    gx = (points_xy[:, 0] / (W - 1)) * 2 - 1
    gy = (points_xy[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)  # (1,1,N,2)

    m4 = map_1hw.unsqueeze(0)  # (1,1,H,W)
    samp = F.grid_sample(m4, grid, align_corners=True)      # (1,1,1,N)
    return samp.view(-1)

def bbox_to_queries(center_xy, box_wh, steps: int) -> torch.Tensor:
    """
    Create (steps*steps, 2) query points inside init bbox in init-frame coords.
    """
    cx, cy = center_xy
    bw, bh = box_wh
    x0 = cx - bw / 2.0
    y0 = cy - bh / 2.0
    x1 = x0 + bw
    y1 = y0 + bh

    xs = torch.linspace(x0, x1 - 1, steps=steps)
    ys = torch.linspace(y0, y1 - 1, steps=steps)
    xv, yv = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack([xv.flatten(), yv.flatten()], dim=-1)  # (N,2)


def points_to_bbox_xywh(pts_np: np.ndarray, H: int, W: int):
    """
    pts_np: (N,2) float numpy
    returns bbox [x,y,w,h] clamped to frame
    """
    if pts_np.size == 0:
        return None

    x_min = max(0, min(W - 1, float(pts_np[:, 0].min())))
    x_max = max(0, min(W - 1, float(pts_np[:, 0].max())))
    y_min = max(0, min(H - 1, float(pts_np[:, 1].min())))
    y_max = max(0, min(H - 1, float(pts_np[:, 1].max())))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    return [x_min, y_min, w, h]


def sample_delta_indices_at_points(selected_delta_i_11hw: torch.Tensor,
                                   points_xy: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbour sample selected_delta_i at point locations.

    selected_delta_i_11hw: (1,1,H,W) int64 (often on CPU)
    points_xy: (N,2) float tensor in pixel coords
    returns: (N,) int64 indices into used_deltas
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

    return sdi[0, 0, yi, xi].long()


def mode_delta_from_indices(delta_idx: torch.Tensor, used_deltas):
    """
    returns (mode_val, mode_frac, counts_tensor)
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


def as_float(x):
    return float(x.item()) if torch.is_tensor(x) else float(x)


# ----------------------------
# Main
# ----------------------------
def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame.")

    cfg = load_config(CONFIG)
    tracker = cfg.tracker_class(cfg)
    tracker.init(frame0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_queries = bbox_to_queries(POINT_XY, BOX_WH, steps=STEPS).float().to(device)

    # counts over frames
    delta_counts = Counter()

    # CSV header
    print("frame,"
          "bbox_x,bbox_y,bbox_w,bbox_h,"
          "sig_pts_mean,sig_pts_med,sig_pts_max,"
          "occ_pts_mean,occ_pts_med,"
          "delta_mode,delta_mode_frac,"
          "sig_full_max")

    frame_i = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        meta = tracker.track(frame)
        flow_result = meta.result if hasattr(meta, "result") else meta

        init_queries = init_queries.to(flow_result.flow.device)
        new_coords = flow_result.warp_forward_points(init_queries)
        
        # Warp fixed init queries with accumulated flow init->t
        new_coords = flow_result.warp_forward_points(init_queries)
        new_coords_np = new_coords.detach().cpu().numpy()

        H, W = frame.shape[:2]

        # Bbox from warped points (this is what makes w/h vary)
        bbox = points_to_bbox_xywh(new_coords_np, H, W)
        if bbox is None:
            bx = by = bw = bh = np.nan
        else:
            bx, by, bw, bh = bbox

        # Sample sigma/occ at points
        sigma_map = flow_result.sigma
        occ_map = flow_result.occlusion

        sig_pts = sample_map_at_points(sigma_map, new_coords)
        occ_pts = sample_map_at_points(occ_map, new_coords)

        if sig_pts is not None and sig_pts.numel() > 0:
            sig_pts_mean = as_float(sig_pts.mean())
            sig_pts_med  = as_float(sig_pts.median())
            sig_pts_max  = as_float(sig_pts.max())
        else:
            sig_pts_mean = sig_pts_med = sig_pts_max = np.nan

        if occ_pts is not None and occ_pts.numel() > 0:
            occ_pts_mean = as_float(occ_pts.mean())
            occ_pts_med  = as_float(occ_pts.median())
        else:
            occ_pts_mean = occ_pts_med = np.nan

        # Full-frame max sigma
        sig_full_max = np.nan
        if sigma_map is not None:
            sm = sigma_map.detach()
            if sm.is_cuda:
                sm = sm.cpu()
            sig_full_max = float(sm.max().item())

        #occl_map = getattr(flow_result, "occlusion", None)

        occ_out_dir = Path("./occ_pngs_4")  # change path as desired

        save_occlusion_png(flow_result.occlusion, occ_out_dir, frame_i, also_color=True)

        # Mode delta at points
        delta_mode_val = ""
        delta_mode_frac = ""
        if hasattr(meta, "selected_delta_i") and hasattr(meta, "used_deltas"):
            delta_idx_pts = sample_delta_indices_at_points(meta.selected_delta_i, new_coords)
            mode_val, mode_frac, _ = mode_delta_from_indices(delta_idx_pts, meta.used_deltas)

            if mode_val is not None:
                if isinstance(mode_val, float) and not np.isfinite(mode_val):
                    delta_mode_val = "inf"
                    delta_counts["inf"] += 1
                else:
                    # mode_val sometimes is numpy scalar / float
                    delta_mode_val = str(int(round(float(mode_val))))
                    delta_counts[int(delta_mode_val)] += 1

                delta_mode_frac = f"{mode_frac:.3f}"

        print(f"{frame_i},"
              f"{bx:.3f},{by:.3f},{bw:.3f},{bh:.3f},"
              f"{sig_pts_mean},{sig_pts_med},{sig_pts_max},"
              f"{occ_pts_mean},{occ_pts_med},"
              f"{delta_mode_val},{delta_mode_frac},"
              f"{sig_full_max}")

        frame_i += 1

    cap.release()

    print("\nDelta mode occurrences (per frame):")
    for k, v in delta_counts.most_common():
        print(f"  {k}: {v}")

    print("Done.")


if __name__ == "__main__":
    main()