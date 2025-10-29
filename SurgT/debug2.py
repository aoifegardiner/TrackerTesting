import cv2
import torch
import numpy as np
import yaml
from pathlib import Path

from MFT_WAFT.MFT.config import load_config
from MFT_WAFT.MFT.waft import WAFTWrapper

# -------------------------------
# CONFIG / PATHS
# -------------------------------
VIDEO_PATH = Path("/Workspace/agardiner_STIR_submission/data/validation/case_1/1/video.mp4")
GT_PATH = Path("/Workspace/agardiner_STIR_submission/data/validation/case_1/1/gt_rectified_0.yaml")
CONFIG_PATH = Path("/Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/MFT_cfg.py")
OUTPUT_DIR = Path("./debug_output")
OUTPUT_DIR.mkdir(exist_ok=True)

#with open(GT_PATH) as f:
#    gt_data = yaml.load(f, Loader=yaml.FullLoader)
#def tuple_to_list(t):
#    if isinstance(t, tuple):
#        return [tuple_to_list(x) for x in t]
#    elif isinstance(t, list):
#        return [tuple_to_list(x) for x in t]
#    else:
#        return t
#
#gt_data = tuple_to_list(gt_data)



# -------------------------------
# LOAD CONFIG & TRACKER
# -------------------------------
config = load_config(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
tracker = config.tracker_class(config)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def gt_to_bbox(gt_tuple):
    """Convert GT [x_center, y_center, width, height] -> [x1, y1, x2, y2]"""
    x_c, y_c, w, h = gt_tuple
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return [x1, y1, x2, y2]

def bbox_to_queries(bbox, steps=5):
    """Convert bbox [x1,y1,x2,y2] into grid of points for tracking"""
    x1, y1, x2, y2 = bbox
    xs = torch.linspace(x1, x2, steps=steps)
    ys = torch.linspace(y1, y2, steps=steps)
    xv, yv = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack([xv.flatten(), yv.flatten()], dim=-1).float().to(device)
    return points

def draw_bbox(frame, bbox, color=(0,255,0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

# -------------------------------
# LOAD VIDEO AND GT
# -------------------------------
cap = cv2.VideoCapture(str(VIDEO_PATH))
with open(GT_PATH) as f:
    gt_data = yaml.load(f, Loader=yaml.FullLoader)

# -------------------------------
# INITIALIZE TRACKER
# -------------------------------
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Failed to read first frame from video")

frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
gt_bbox0 = gt_to_bbox(gt_data[0][2][0])  # Use first GT tuple for first bbox
queries = bbox_to_queries(gt_bbox0)
tracker.init(frame0)
last_points = queries

# -------------------------------
# TRACK AND DEBUG
# -------------------------------
frame_idx = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # TRACK
    result = tracker.track(frame_rgb, last_points)
    coords = result.warp_forward_points(last_points)
    last_points = coords

    # Compute predicted bbox
    x_min = coords[:,0].min().item()
    x_max = coords[:,0].max().item()
    y_min = coords[:,1].min().item()
    y_max = coords[:,1].max().item()
    bbox_pred = [x_min, y_min, x_max, y_max]

    # Get GT for this frame
    gt_bbox = gt_to_bbox(gt_data[frame_idx][2][0])

    # DEBUG PRINT
    print(f"Frame {frame_idx}: Predicted bbox: {bbox_pred}, GT: {gt_bbox}")

    # OPTIONAL: Save annotated frame
    frame_debug = frame.copy()
    draw_bbox(frame_debug, bbox_pred, color=(0,255,0))
    draw_bbox(frame_debug, gt_bbox, color=(0,0,255))
    cv2.imwrite(str(OUTPUT_DIR/f"frame_{frame_idx:04d}.png"), frame_debug)

    frame_idx += 1

cap.release()
