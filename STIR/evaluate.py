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
#CONFIG = Path("/Workspace/agardiner_waft/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")
CONFIG = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_RAFT_cfg.py")
VIDEO_PATH = Path("/public_datasets/STIRDataset/22/left/seq13/frames/00108280ms-00117800ms-visible.mp4")

POINT_XY = (1024, 595)
BOX_WH = (44, 44)


# ----------------------------
# Helpers
# ----------------------------
def clamp_box_xywh(x, y, w, h, W, H):
    x = float(x); y = float(y); w = float(w); h = float(h)
    x0 = max(0, min(W - 1, x))
    y0 = max(0, min(H - 1, y))
    x1 = max(0, min(W, x + w))
    y1 = max(0, min(H, y + h))
    w2 = max(0.0, x1 - x0)
    h2 = max(0.0, y1 - y0)
    if w2 < 1 or h2 < 1:
        return None
    return (x0, y0, w2, h2)

def crop_map(map_1hw: torch.Tensor, box_xywh):
    """
    map_1hw: (1,H,W) torch tensor
    returns: (N,) flattened tensor within box
    """
    if map_1hw is None or box_xywh is None:
        return None
    if map_1hw.ndim == 2:
        map_1hw = map_1hw.unsqueeze(0)
    _, H, W = map_1hw.shape
    x, y, w, h = box_xywh
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(1, min(W, x1))
    y1 = max(1, min(H, y1))
    patch = map_1hw[:, y0:y1, x0:x1]  # (1,hh,ww)
    return patch.reshape(-1)

def stats_1d(t: torch.Tensor):
    """
    Return mean, median, max as python floats for a 1D tensor.
    """
    if t is None or t.numel() == 0:
        return (None, None, None)
    t = t.float()
    mean = float(t.mean().item())
    med  = float(t.median().item())
    mx   = float(t.max().item())
    return (mean, med, mx)

def occ_stats_1d(t: torch.Tensor):
    """
    Return mean, median as python floats for occlusion tensor.
    """
    if t is None or t.numel() == 0:
        return (None, None)
    t = t.float()
    mean = float(t.mean().item())
    med  = float(t.median().item())
    return (mean, med)

def mode_delta_in_box(selected_delta_i_11hw: torch.Tensor, used_deltas, box_xywh, W, H):
    """
    Mode delta over all pixels in the bbox, using nearest pixels of selected_delta_i.

    selected_delta_i_11hw: (1,1,H,W) int64 (usually on CPU)
    used_deltas: list of delta values (may include np.inf)
    """
    if selected_delta_i_11hw is None or used_deltas is None or box_xywh is None:
        return None

    sdi = selected_delta_i_11hw
    if torch.is_tensor(sdi) and sdi.is_cuda:
        sdi = sdi.cpu()

    x, y, w, h = box_xywh
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))

    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(1, min(W, x1))
    y1 = max(1, min(H, y1))

    patch = sdi[0, 0, y0:y1, x0:x1].reshape(-1).to(torch.int64)
    if patch.numel() == 0:
        return None

    counts = torch.bincount(patch, minlength=len(used_deltas))
    mode_i = int(torch.argmax(counts).item())
    return used_deltas[mode_i]


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

    # init point (kept fixed in init frame coords, warped each frame)
    p0 = torch.tensor([[POINT_XY[0], POINT_XY[1]]], dtype=torch.float32)

    print("frame,"
          "bbox_x,bbox_y,bbox_w,bbox_h,"
          "sig_mean,sig_med,sig_max,"
          "occ_mean,occ_med,"
          "delta_mode,"
          "sig_full_max")

    frame_i = 1
    delta_counts = Counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        meta = tracker.track(frame)
        flow_result = meta.result if hasattr(meta, "result") else meta

        # Warp init point to current frame
        p = flow_result.warp_forward_points(p0)
        cx, cy = p[0].tolist()

        # Build fixed-size bbox centered at tracked point
        bw, bh = BOX_WH
        x = cx - bw / 2.0
        y = cy - bh / 2.0

        H, W = frame.shape[:2]
        bbox = clamp_box_xywh(x, y, bw, bh, W, H)

        # --- Sigma stats in bbox ---
        sigma_map = flow_result.sigma
        if torch.is_tensor(sigma_map):
            sigma_map = sigma_map.detach().cpu()
        sig_patch = crop_map(sigma_map, bbox)
        sig_mean, sig_med, sig_max = stats_1d(sig_patch)

        # --- Occlusion stats in bbox ---
        occ_map = flow_result.occlusion
        if torch.is_tensor(occ_map):
            occ_map = occ_map.detach().cpu()
        occ_patch = crop_map(occ_map, bbox)
        occ_mean, occ_med = occ_stats_1d(occ_patch)

        # --- Mode delta in bbox ---
        delta_mode = None
        if hasattr(meta, "selected_delta_i") and hasattr(meta, "used_deltas"):
            delta_mode = mode_delta_in_box(meta.selected_delta_i, meta.used_deltas, bbox, W, H)
            # count occurrences per-frame
            if delta_mode is not None:
                delta_key = "inf" if (isinstance(delta_mode, float) and not np.isfinite(delta_mode)) else int(delta_mode)
                delta_counts[delta_key] += 1

        # --- Full-frame max sigma ---
        sig_full_max = None
        if sigma_map is not None:
            # sigma_map is (1,H,W) or (H,W)
            if sigma_map.ndim == 3:
                sig_full_max = float(sigma_map.max().item())
            else:
                sig_full_max = float(sigma_map.max().item())

        # Print row
        if bbox is None:
            bx = by = bw2 = bh2 = np.nan
        else:
            bx, by, bw2, bh2 = bbox

        # delta display
        if delta_mode is None:
            delta_disp = ""
        else:
            delta_disp = "inf" if (isinstance(delta_mode, float) and not np.isfinite(delta_mode)) else str(int(delta_mode))

        print(f"{frame_i},"
              f"{bx:.3f},{by:.3f},{bw2:.3f},{bh2:.3f},"
              f"{sig_mean},{sig_med},{sig_max},"
              f"{occ_mean},{occ_med},"
              f"{delta_disp},"
              f"{sig_full_max}")

        frame_i += 1

    cap.release()
    print("\nDelta mode occurrences (per frame):")
    for k, v in delta_counts.most_common():
        print(f"  {k}: {v}")
    print("Done.")


if __name__ == "__main__":
    main()


## ----------------------------
## Utility: sample map at subpixel location
## ----------------------------
#@torch.no_grad()
#def sample_map_at_point(map_1hw: torch.Tensor, point_xy: torch.Tensor):
#    """
#    map_1hw: (1,H,W) or (H,W)
#    point_xy: (1,2) tensor in pixel coords
#    """
#    if map_1hw is None:
#        return None
#
#    if map_1hw.ndim == 2:
#        map_1hw = map_1hw.unsqueeze(0)
#
#    _, H, W = map_1hw.shape
#
#    gx = (point_xy[:, 0] / (W - 1)) * 2 - 1
#    gy = (point_xy[:, 1] / (H - 1)) * 2 - 1
#    grid = torch.stack([gx, gy], dim=-1).view(1, 1, 1, 2)
#
#    m4 = map_1hw.unsqueeze(0)  # (1,1,H,W)
#    val = F.grid_sample(m4, grid, align_corners=True)
#    return val.view(-1)[0]
#
#
## ----------------------------
## Main
## ----------------------------
#def main():
#    cap = cv2.VideoCapture(str(VIDEO_PATH))
#    if not cap.isOpened():
#        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
#
#    ok, frame0 = cap.read()
#    if not ok:
#        raise RuntimeError("Could not read first frame.")
#    cfg = load_config(CONFIG)
#    tracker = cfg.tracker_class(cfg)
#    tracker.init(frame0)
#
#    p0 = torch.tensor([[POINT_XY[0], POINT_XY[1]]], dtype=torch.float32)
#
#    print("frame,x,y,sigma,occlusion,delta")
#
#    frame_i = 1
#
#    while True:
#        ok, frame = cap.read()
#        if not ok:
#            break
#
#        meta = tracker.track(frame)
#        flow_result = meta.result if hasattr(meta, "result") else meta
#
#        # Warp initial point
#        p = flow_result.warp_forward_points(p0)
#        x, y = p[0].tolist()
#
#        # Sample maps at this location
#        sigma_val = sample_map_at_point(flow_result.sigma, p)
#        occ_val = sample_map_at_point(flow_result.occlusion, p)
#
#        delta_val = None
#
#        if hasattr(meta, "selected_delta_i") and hasattr(meta, "used_deltas"):
#            sdi = meta.selected_delta_i  # expected (1,1,H,W) on CPU or GPU
#
#            # Use the tracked point position (x,y) you already computed
#            xi = int(round(x))
#            yi = int(round(y))
#
#            H, W = sdi.shape[-2], sdi.shape[-1]
#            xi = max(0, min(W - 1, xi))
#            yi = max(0, min(H - 1, yi))
#
#            delta_idx = int(sdi[0, 0, yi, xi].item())
#            delta_idx = max(0, min(len(meta.used_deltas) - 1, delta_idx))
#
#            delta_val = meta.used_deltas[delta_idx]
#
#
#        # Convert to python floats
#        sigma_val = float(sigma_val.item()) if sigma_val is not None else None
#        occ_val = float(occ_val.item()) if occ_val is not None else None
#        #delta_val = float(delta_val.item()) if delta_val is not None else None
#
#        print(f"{frame_i},{x:.3f},{y:.3f},{sigma_val},{occ_val},{delta_val}")
#
#        frame_i += 1
#
#    cap.release()
#    print("Done.")
#
#
#if __name__ == "__main__":
#    main()
#