import numpy as np
import cv2, glob

flows = sorted(glob.glob("/Workspace/agardiner_STIR_submission/debug_flows/flow_*.png"))
magnitudes = []
for f in flows:
    img = cv2.imread(f).astype(np.float32) / 255.
    magnitudes.append(img.std())  # or np.mean(img) depending on encoding
print("Mean std per frame:", np.mean(magnitudes))
print("Variance:", np.var(magnitudes))
print("Max std:", np.max(magnitudes))
print("Min std:", np.min(magnitudes))
