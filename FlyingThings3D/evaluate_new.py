#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from agardiner_STIR_submission.MFT_WAFT.MFT.config_RAFT import load_config
#from agardiner_waft.MFT_WAFT.MFT.config import load_config

# ----------------------------
# User inputs
# ----------------------------
SEQ_DIR = Path("/public_datasets/FlyingThings3D/frames_cleanpass/TEST/A/0001/left")  # folder of frames
CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_RAFT_cfg.py")
#CONFIG = Path("/Workspace/agardiner_waft/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")

# Init region definition (in frame0 coords)
POINT_XY = (500, 200)   # center of init region in frame 0
BOX_WH   = (44, 44)      # init box size
STEPS    = 5             # query grid resolution inside init box (steps x steps)

# Outputs
OUT_DIR = Path("/Workspace/results/MFT_FT")
OUT_VIDEO_NAME = "overlay_pred_bbox.mp4"   # set None to disable video
OUT_FPS = 24

# If your frames are not 0-indexed or have gaps, we just iterate sorted filenames.
# We'll print "frame" as 1..N-1 (matching your STIR logs).
# ----------------------------


# ----------------------------
# Helpers
# ----------------------------
@torch.no_grad()
def sample_map_at_points(map_1hw: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """map_1hw: (1,H,W) or (H,W); points_xy: (N,2) in pixel coords"""
    if map_1hw is None or points_xy is None:
        return None
    if map_1hw.ndim == 2:
        map_1hw = map_1hw.unsqueeze(0)  # (1,H,W)

    # ensure device match
    points_xy = points_xy.to(map_1hw.device)

    _, H, W = map_1hw.shape
    gx = (points_xy[:, 0] / (W - 1)) * 2 - 1
    gy = (points_xy[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)  # (1,1,N,2)

    m4 = map_1hw.unsqueeze(0)  # (1,1,H,W)
    samp = F.grid_sample(m4, grid, align_corners=True)      # (1,1,1,N)
    return samp.view(-1)

def bbox_to_queries(center_xy, box_wh, steps: int) -> torch.Tensor:
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

def mode_delta_from_indices(delta_idx: torch.Tensor, used_deltas):
    if delta_idx is None or delta_idx.numel() == 0:
        return None, None, None
    D = len(used_deltas)
    counts = torch.bincount(delta_idx, minlength=D)
    mode_i = int(torch.argmax(counts).item())
    mode_frac = float(counts[mode_i].item() / max(1, delta_idx.numel()))
    mode_val = used_deltas[mode_i]
    return mode_val, mode_frac, counts

def as_float(x):
    return float(x.item()) if torch.is_tensor(x) else float(x)

def draw_xywh(img_bgr, bbox, color=(0, 0, 255), thickness=2):
    x, y, w, h = bbox
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)


# ----------------------------
# Main
# ----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect frames
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    frames = []
    for e in exts:
        frames += list(SEQ_DIR.glob(e))
    frames = sorted(frames)
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 image frames in {SEQ_DIR}, found {len(frames)}")

    # Load first frame
    frame0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if frame0 is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    H, W = frame0.shape[:2]

    # Init tracker
    cfg = load_config(CONFIG)
    tracker = cfg.tracker_class(cfg)
    tracker.init(frame0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_queries = bbox_to_queries(POINT_XY, BOX_WH, steps=STEPS).float().to(device)

    # Optional video
    writer = None
    if OUT_VIDEO_NAME is not None:
        out_video_path = OUT_DIR / OUT_VIDEO_NAME
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, OUT_FPS, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {out_video_path}")
    else:
        out_video_path = None

    # Print CSV header (STIR-like)
    print("frame,"
          "bbox_x,bbox_y,bbox_w,bbox_h,"
          "sig_pts_mean,sig_pts_med,sig_pts_max,"
          "occ_pts_mean,occ_pts_med,"
          "delta_mode,delta_mode_frac,"
          "sig_full_max")

    delta_counts = Counter()

    # Write frame0 overlay if desired
    if writer is not None:
        vis0 = frame0.copy()
        # show init box for context
        x0 = POINT_XY[0] - BOX_WH[0] / 2.0
        y0 = POINT_XY[1] - BOX_WH[1] / 2.0
        draw_xywh(vis0, (x0, y0, BOX_WH[0], BOX_WH[1]), color=(0, 255, 0), thickness=2)
        cv2.putText(vis0, "INIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis0)

    # Iterate frames 1..end (STIR prints starting at 1)
    for frame_i in range(1, len(frames)):
        frame = cv2.imread(str(frames[frame_i]), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] Failed to read {frames[frame_i]}, skipping")
            continue
        if frame.shape[:2] != (H, W):
            raise RuntimeError(f"Frame size changed at {frames[frame_i]}: got {frame.shape[:2]}, expected {(H,W)}")

        meta = tracker.track(frame)
        flow_result = meta.result if hasattr(meta, "result") else meta

        # Warp fixed init queries with accumulated flow init->t
        init_queries = init_queries.to(flow_result.flow.device)
        new_coords = flow_result.warp_forward_points(init_queries)
        new_coords_np = new_coords.detach().cpu().numpy()

        # Pred bbox from warped points
        bbox = points_to_bbox_xywh(new_coords_np, H, W)
        if bbox is None:
            bx = by = bw = bh = np.nan
        else:
            bx, by, bw, bh = bbox

        # Sigma/occ sampled at tracked points
        sigma_map = flow_result.sigma
        occ_map   = flow_result.occlusion

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

        # Mode delta at points (if stored by MFT.track)
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
                    delta_mode_val = str(int(round(float(mode_val))))
                    delta_counts[int(delta_mode_val)] += 1
                delta_mode_frac = f"{mode_frac:.3f}"

        # Print CSV row
        print(f"{frame_i},"
              f"{bx:.3f},{by:.3f},{bw:.3f},{bh:.3f},"
              f"{sig_pts_mean},{sig_pts_med},{sig_pts_max},"
              f"{occ_pts_mean},{occ_pts_med},"
              f"{delta_mode_val},{delta_mode_frac},"
              f"{sig_full_max}")

        # Overlay video
        if writer is not None:
            vis = frame.copy()
            if bbox is not None and np.isfinite(bx) and np.isfinite(by):
                draw_xywh(vis, bbox, color=(0, 0, 255), thickness=2)

            txt1 = f"frame {frame_i:04d}"
            txt2 = f"sig mean/med/max: {sig_pts_mean:.3f}/{sig_pts_med:.3f}/{sig_pts_max:.3f}" if np.isfinite(sig_pts_mean) else "sig: n/a"
            txt3 = f"occ mean/med: {occ_pts_mean:.3f}/{occ_pts_med:.3f}" if np.isfinite(occ_pts_mean) else "occ: n/a"
            txt4 = f"delta mode: {delta_mode_val} ({delta_mode_frac})" if delta_mode_val != "" else "delta: n/a"
            txt5 = f"sig_full_max: {sig_full_max:.3f}" if np.isfinite(sig_full_max) else "sig_full_max: n/a"

            y = 30
            for t in [txt1, txt2, txt3, txt4, txt5]:
                cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                y += 25

            writer.write(vis)

    if writer is not None:
        writer.release()
        print(f"\n[VIDEO] wrote {out_video_path}")

    # Delta summary like STIR
    print("\nDelta mode occurrences (per frame):")
    for k, v in delta_counts.most_common():
        print(f"  {k}: {v}")

    print("Done.")


if __name__ == "__main__":
    main()