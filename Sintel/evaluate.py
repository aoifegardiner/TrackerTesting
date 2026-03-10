#!/usr/bin/env python3
"""
Sintel (or Sintel-like) pairwise flow evaluation on ONE sequence folder,
with:
  - fixed initial ROI bbox (x,y,w,h) in frame 0 coordinates
  - incremental ROI tracking using predicted flow (frame i -> i+1)
  - mean sigma computed inside the tracked ROI each step
  - debug MP4 video with ROI + sigma text overlay

Assumptions:
  - frames are PNGs named like frame_0001.png ... (or any sortable names)
  - GT flow .flo files exist with matching names (optional; if absent, EPE is skipped)
  - your flower wrapper returns:
        flow_pred: torch Tensor (2,H,W) or (1,2,H,W) or numpy (H,W,2)/(2,H,W)
        extra: dict with "sigma" possibly (torch (1,H,W) or (H,W) or list-wrapped)

Edit the PATHS and ROI_INIT below, then run.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

#from agardiner_waft.MFT_WAFT.MFT.config import load_config
from agardiner_STIR_submission.MFT_WAFT.MFT.config_RAFT import load_config


# ----------------------------
# User settings (EDIT THESE)
# ----------------------------

# One sequence folder with frames (a single directory containing frame_*.png)
SEQ_DIR = Path("/public_datasets/Sintel-complete/training/clean/market_2")  # <-- change to your folder

# Optional GT flow folder for that sequence (can be None / not exist)
GT_FLOW_DIR = Path("/public_datasets/Sintel-complete/training/flow/market_2")  # <-- change or set to None

# Your MFT config (for flower wrapper)
#CONFIG = Path("/Workspace/agardiner_waft/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")
CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_RAFT_cfg.py")

# Fixed ROI bbox in frame 0 coordinates: (x, y, w, h)
ROI_INIT = [600, 200, 50, 50]  # <-- change

# Output
OUT_DIR = Path("./sintel_one_seq_outputs_mft")
OUT_VIDEO_NAME = "roi_debug_2.mp4"
OUT_CSV_NAME = "roi_sigma_2.csv"

# Debug video FPS
OUT_FPS = 24


# ----------------------------
# IO helpers
# ----------------------------

def read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Bad .flo magic: {magic} in {path}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape(h, w, 2)


def unwrap_list(x):
    if isinstance(x, list):
        return x[0]
    return x


# ----------------------------
# Math helpers
# ----------------------------

def clamp_bbox_xywh(b, H, W):
    x, y, w, h = b
    x = max(0.0, min(float(W - 1), float(x)))
    y = max(0.0, min(float(H - 1), float(y)))
    w = max(1.0, min(float(W) - x, float(w)))
    h = max(1.0, min(float(H) - y, float(h)))
    return [x, y, w, h]


@torch.no_grad()
def warp_points_xy(flow_2hw: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly samples flow at pts and adds displacement.
    flow_2hw: (2,H,W) dx,dy in pixels
    pts_xy: (N,2) pixel coords in source frame
    returns: (N,2) pixel coords in target frame
    """
    device = flow_2hw.device
    H, W = flow_2hw.shape[1:]
    pts_xy = pts_xy.to(device).float()

    gx = (pts_xy[:, 0] / (W - 1)) * 2 - 1
    gy = (pts_xy[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)  # (1,1,N,2)

    flow_b = flow_2hw.unsqueeze(0)  # (1,2,H,W)
    samp = F.grid_sample(flow_b, grid, align_corners=True)  # (1,2,1,N)
    dxy = samp.view(2, -1).T  # (N,2)

    return pts_xy + dxy


@torch.no_grad()
def warp_bbox_xywh(flow_2hw: torch.Tensor, bbox_xywh, H, W):
    """
    Warps the 4 corners of bbox by flow, returns axis-aligned bbox covering warped corners.
    """
    x, y, w, h = bbox_xywh
    corners = torch.tensor(
        [
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h],
        ],
        dtype=torch.float32,
        device=flow_2hw.device,
    )
    corners2 = warp_points_xy(flow_2hw, corners)
    x2_min = corners2[:, 0].min().item()
    x2_max = corners2[:, 0].max().item()
    y2_min = corners2[:, 1].min().item()
    y2_max = corners2[:, 1].max().item()
    return clamp_bbox_xywh([x2_min, y2_min, x2_max - x2_min, y2_max - y2_min], H, W)


def mean_in_bbox(map_hw: torch.Tensor, bbox_xywh):
    """
    map_hw: (H,W) torch
    bbox_xywh: floats
    mean over integer crop region
    """
    H, W = map_hw.shape
    x, y, w, h = bbox_xywh
    x0 = int(max(0, min(W - 1, round(x))))
    y0 = int(max(0, min(H - 1, round(y))))
    x1 = int(max(0, min(W, round(x + w))))
    y1 = int(max(0, min(H, round(y + h))))
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    crop = map_hw[y0:y1, x0:x1]
    return crop.mean().item()


@torch.no_grad()
def epe_map(flow_pred_2hw: torch.Tensor, flow_gt_2hw: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(flow_pred_2hw - flow_gt_2hw, dim=0)  # (H,W)


def normalize_flow_pred(flow_pred, device):
    """
    Accepts flow_pred as torch or numpy in common layouts and returns torch (2,H,W) on device.
    """
    if isinstance(flow_pred, np.ndarray):
        # could be (H,W,2) or (2,H,W)
        if flow_pred.ndim == 3 and flow_pred.shape[2] == 2:
            flow_pred = np.transpose(flow_pred, (2, 0, 1))  # (2,H,W)
        flow_t = torch.from_numpy(flow_pred).to(device)
    else:
        flow_t = flow_pred.to(device)

    if flow_t.ndim == 4:
        # (1,2,H,W)
        flow_t = flow_t[0]
    if flow_t.shape[0] != 2:
        raise ValueError(f"Expected flow with 2 channels, got shape {tuple(flow_t.shape)}")
    return flow_t


def normalize_sigma(extra, device):
    """
    Returns sigma_hw torch (H,W) on device, or None.
    """
    if extra is None or "sigma" not in extra:
        return None
    sigma = unwrap_list(extra["sigma"])
    if sigma is None:
        return None
    if torch.is_tensor(sigma):
        sigma_t = sigma.to(device)
        if sigma_t.ndim == 3:
            # (1,H,W)
            sigma_t = sigma_t[0]
        elif sigma_t.ndim != 2:
            raise ValueError(f"Unexpected sigma shape {tuple(sigma_t.shape)}")
        return sigma_t
    # if numpy
    if isinstance(sigma, np.ndarray):
        sig = torch.from_numpy(sigma).to(device)
        if sig.ndim == 3 and sig.shape[0] == 1:
            sig = sig[0]
        return sig
    return None


# ----------------------------
# Main
# ----------------------------

@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init flow model/wrapper ("flower")
    cfg = load_config(CONFIG)
    flower = cfg.flow_config.of_class(cfg.flow_config)

    # Collect frames
    frames = sorted([p for p in SEQ_DIR.glob("*.png")])
    if len(frames) < 2:
        raise RuntimeError(f"Need at least 2 PNG frames in {SEQ_DIR}, found {len(frames)}")

    # Read first frame to size + init ROI
    im0_bgr = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if im0_bgr is None:
        raise RuntimeError(f"Failed to read {frames[0]}")
    H0, W0 = im0_bgr.shape[:2]
    roi_bbox = clamp_bbox_xywh(ROI_INIT, H0, W0)

    # Video writer
    out_video_path = OUT_DIR / OUT_VIDEO_NAME
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, OUT_FPS, (W0, H0))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {out_video_path}")

    # CSV output
    out_csv_path = OUT_DIR / OUT_CSV_NAME
    f_csv = open(out_csv_path, "w", newline="")
    f_csv.write("i,im1,im2,roi_x,roi_y,roi_w,roi_h,roi_sigma_mean,epe_mean\n")

    epe_means = []
    roi_sigma_means = []

    # Write first frame visualization too (ROI at init)
    vis0 = im0_bgr.copy()
    x, y, w, h = roi_bbox
    cv2.rectangle(vis0, (int(round(x)), int(round(y))), (int(round(x + w)), int(round(y + h))), (0, 255, 0), 2)
    cv2.putText(vis0, f"INIT ROI", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(vis0)

    # Iterate pairs
    for i in range(len(frames) - 1):
        im1_path = frames[i]
        im2_path = frames[i + 1]

        im1_bgr = cv2.imread(str(im1_path), cv2.IMREAD_COLOR)
        im2_bgr = cv2.imread(str(im2_path), cv2.IMREAD_COLOR)
        if im1_bgr is None or im2_bgr is None:
            print(f"[WARN] Failed to read {im1_path} or {im2_path}, skipping")
            continue

        # Predict flow (expects RGB arrays per your previous usage)
        im1_rgb = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2RGB)
        im2_rgb = cv2.cvtColor(im2_bgr, cv2.COLOR_BGR2RGB)

        flow_pred, extra = flower.compute_flow(im1_rgb, im2_rgb, mode="flow", init_flow=None)
        flow_pred_t = normalize_flow_pred(flow_pred, device)
        sigma_hw = normalize_sigma(extra, device)

        H, W = flow_pred_t.shape[1:]
        if (H, W) != (H0, W0):
            # If your model resizes internally, ROI tracking + video need consistent shapes.
            # For now, we hard error to avoid silent misalignment.
            raise RuntimeError(
                f"Flow size {(H,W)} != image size {(H0,W0)}. "
                "If your wrapper resizes, you must scale ROI + visualization accordingly."
            )

        # Track ROI forward by predicted flow
        roi_bbox = warp_bbox_xywh(flow_pred_t, roi_bbox, H, W)

        # Mean sigma in ROI
        roi_sigma_mean = float("nan")
        if sigma_hw is not None:
            roi_sigma_mean = mean_in_bbox(sigma_hw, roi_bbox)
            roi_sigma_means.append(roi_sigma_mean)

        # Optional EPE if GT available
        epe_roi_mean = float("nan")
        if GT_FLOW_DIR is not None:
            flo_path = GT_FLOW_DIR / im1_path.name.replace(".png", ".flo")
            if flo_path.exists():
                flow_gt = read_flo(str(flo_path))  # (H,W,2)
                flow_gt_t = torch.from_numpy(flow_gt).permute(2, 0, 1).to(device)  # (2,H,W)
                #epe = epe_map(flow_pred_t, flow_gt_t)
                #epe_mean = epe.mean().item()
                epe = epe_map(flow_pred_t, flow_gt_t)
                epe_roi_mean = mean_in_bbox(epe, roi_bbox)
                epe_means.append(epe_roi_mean)

        # Render visualization on im2
        vis = im2_bgr.copy()
        x, y, w, h = roi_bbox
        cv2.rectangle(
            vis,
            (int(round(x)), int(round(y))),
            (int(round(x + w)), int(round(y + h))),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"{SEQ_DIR.name} {im2_path.name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            vis,
            f"ROI sigma mean: {roi_sigma_mean:.3f}" if np.isfinite(roi_sigma_mean) else "ROI sigma mean: n/a",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if np.isfinite(epe_roi_mean):
            cv2.putText(
                vis,
                f"EPE ROI mean: {epe_roi_mean:.3f}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        writer.write(vis)

        # Log CSV
        f_csv.write(
            f"{i},{im1_path.name},{im2_path.name},"
            f"{x:.3f},{y:.3f},{w:.3f},{h:.3f},"
            f"{roi_sigma_mean:.6f},{epe_roi_mean:.6f}\n"
        )

        # Console
        msg = f"[{i:04d}] {im1_path.name}->{im2_path.name} ROI sigma={roi_sigma_mean:.3f}"
        if np.isfinite(epe_roi_mean):
            msg += f" EPE={epe_roi_mean:.3f}"
        print(msg)

    # Cleanup
    writer.release()
    f_csv.close()

    print("\nDone.")
    print(f"Video written: {out_video_path}")
    print(f"CSV written  : {out_csv_path}")

    if epe_means:
        print(f"Overall EPE mean over {len(epe_means)} pairs: {float(np.mean(epe_means)):.4f}")
    if roi_sigma_means:
        print(f"Overall ROI sigma mean over {len(roi_sigma_means)} pairs: {float(np.mean(roi_sigma_means)):.4f}")


if __name__ == "__main__":
    main()