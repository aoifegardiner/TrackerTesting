import os
import cv2
import pickle
import numpy as np
from pathlib import Path

# === CONFIG ===
flow_dir = Path("./debug_flows_raw_left")      # folder where flow_####.png are saved
pkl_path = Path("./results/case1_video1_left/case1_video1_left_predictions_waft.pkl")
output_path = Path("./results/flow_overlay_400_944.mp4")
frame_start, frame_end = 400, 944

# === Load predictions from .pkl ===
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

frame_indices = np.array(data["frame_indices"])
pred_bboxes = np.array(data["predicted_bboxes"])

# Filter for the 500–600 range
mask = (frame_indices >= frame_start) & (frame_indices <= frame_end)
frame_indices = frame_indices[mask]
pred_bboxes = pred_bboxes[mask]

if len(frame_indices) == 0:
    raise ValueError("No frames found in range 500–600 in the .pkl file!")

# === Helper: draw bbox on frame ===
def draw_bbox(frame, bbox, color=(255, 0, 0), thickness=2):
    """bbox = [x, y, w, h]"""
    x, y, w, h = map(int, bbox)
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    return cv2.rectangle(frame.copy(), top_left, bottom_right, color, thickness)

# === Load first flow image to determine size ===
first_frame_path = flow_dir / f"flow_left_{frame_indices[0]:04d}.png"
if not first_frame_path.exists():
    raise FileNotFoundError(f"Could not find {first_frame_path}")

sample = cv2.imread(str(first_frame_path))
h, w, _ = sample.shape

# === Prepare video writer ===
os.makedirs(output_path.parent, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (w, h))  # 10 fps

# === Iterate through frames ===
for idx, bbox in zip(frame_indices, pred_bboxes):
    flow_path = flow_dir / f"flow_left_{idx:04d}.png"
    if not flow_path.exists():
        print(f"[WARN] Missing flow file for frame {idx}, skipping...")
        continue

    frame = cv2.imread(str(flow_path))
    frame = draw_bbox(frame, bbox, color=(255, 0, 0), thickness=2)  # predicted bbox = blue
    cv2.putText(frame, f"Frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out.write(frame)

out.release()
print(f"\n✅ Saved flow overlay video: {output_path}")
