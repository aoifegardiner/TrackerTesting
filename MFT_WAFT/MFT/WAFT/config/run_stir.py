import torch
from utils import flow_viz
from dataloader import STIR
from core.utils.utils import InputPadder
from core.waft import build_network
import cv2
import os

# Load WAFT model
model = build_network("configs/waft_stir_test.json")
model.eval()

# Load mini STIR dataset
dataset = STIR.STIR(root='datasets/STIR_sample', extract_if_needed=True)

for i in range(len(dataset)):
    img1, img2, _, _ = dataset[i]
    padder = InputPadder(img1.shape[-2:])
    image1, image2 = padder.pad(img1[None], img2[None])

    with torch.no_grad():
        flow_predictions = model(image1.cuda(), image2.cuda(), iters=5)
    
    flow = flow_predictions[-1][0].cpu().numpy().transpose(1, 2, 0)
    flow_img = flow_viz.flow_to_image(flow)
    
    os.makedirs("output/demo/", exist_ok=True)
    cv2.imwrite(f"output/demo/flow_{i}.png", flow_img[:, :, ::-1])
