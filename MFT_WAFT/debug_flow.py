import torch
import cv2
import numpy as np
import einops

from MFT_WAFT.MFT.waft import FlowOUTrackingResult as WAFT  # your WAFT wrapper
from MFT_WAFT.MFT.raft import RAFTWrapper as RAFT                  # vanilla RAFT
from MFT_WAFT.MFT.WAFT.utils.utils import InputPadder

DEVICE = "cuda"

def to_tensor(img):
    # img must be HWC uint8
    return einops.rearrange(
        torch.from_numpy(img[:, :, ::-1].copy()),  # <-- .copy fixes negative stride
        "H W C -> 1 C H W"
    ).float().to(DEVICE)


def load_img(path):
    return cv2.imread(path)

# ---- load test images ----
im1 = cv2.imread("MFT_WAFT/frame_0001.png")
im2 = cv2.imread("MFT_WAFT/frame_0002.png")

t1 = to_tensor(im1)
t2 = to_tensor(im2)

# ---- initialize models ----
raft = RAFT().to(DEVICE).eval()
waft = WAFT().to(DEVICE).eval()

# ---- run RAFT ----
with torch.no_grad():
    raft_flow = raft(t1, t2)[-1][0]    # (2,H,W)

# ---- run WAFT ----
with torch.no_grad():
    waft_flow, _ = waft.calc_flow(t1, t2)
    waft_flow = waft_flow[0]           # (2,H,W)

# ---- compute differences ----
diff = (waft_flow - raft_flow).abs()
print("Mean abs diff:", diff.mean().item())
print("Max abs diff:", diff.max().item())

# ---- visualize ----
def viz(flow):
    mag = np.sqrt(flow[0]**2 + flow[1]**2)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return mag

cv2.imwrite("raft_mag.png", viz(raft_flow.cpu().numpy()))
cv2.imwrite("waft_mag.png", viz(waft_flow.cpu().numpy()))
cv2.imwrite("diff_mag.png", viz(diff.cpu().numpy()))
