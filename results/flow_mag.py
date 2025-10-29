import cv2, numpy as np
import glob, os

flow_files = sorted(glob.glob("debug_flows_raw_left/flow_left_*.png"))
for i in range(450, 550):
    f1 = f"debug_flows_raw_left/flow_left_{i:04d}.png"
    f2 = f"debug_flows_raw_left/flow_left_{i+1:04d}.png"
    if os.path.exists(f1) and os.path.exists(f2):
        im1 = cv2.imread(f1).astype(np.float32)
        im2 = cv2.imread(f2).astype(np.float32)
        diff = np.abs(im2 - im1).mean()
        print(f"[DEBUG FLOW DIFF] {i}->{i+1}: mean pixel diff={diff:.3f}")
