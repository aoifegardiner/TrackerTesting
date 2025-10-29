import cv2
import os
from pathlib import Path

# === Configuration ===
flow_dir = Path("/Workspace/agardiner_STIR_submission/debug_flows_raw_left")  # folder with your flow images
output_path = flow_dir / "flow_video.mp4"  # output video file
fps = 10  # adjust as needed (frames per second)

# === Gather and sort frames ===
frames = sorted(flow_dir.glob("flow_left_*.png"))  # match your naming scheme
if not frames:
    raise FileNotFoundError(f"No flow_*.png images found in {flow_dir}")

# === Read first frame to get dimensions ===
first_frame = cv2.imread(str(frames[0]))
height, width, _ = first_frame.shape

# === Initialize video writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# === Write frames to video ===
for i, frame_path in enumerate(frames):
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"[WARN] Skipping unreadable frame: {frame_path}")
        continue
    out.write(frame)
    if i % 50 == 0:
        print(f"[INFO] Added frame {i}/{len(frames)}")

out.release()
print(f"\n✅ Saved video to: {output_path}")
