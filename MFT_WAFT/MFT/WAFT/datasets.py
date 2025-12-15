import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
from glob import glob
import einops


# ============================================================
# Utility: read .flo or .pfm flow
# ============================================================

def read_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise Exception('Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        return data.reshape(h, w, 2)


def read_pfm(path):
    file = open(path, "rb")
    header = file.readline().rstrip().decode("utf-8")
    dims = file.readline().decode("utf-8")
    width, height = map(int, dims.split())
    scale = float(file.readline().rstrip().decode("utf-8"))
    data = np.fromfile(file, "<f") if scale < 0 else np.fromfile(file, ">f")
    data = np.reshape(data, (height, width, 1 if header == "Pf" else 3))
    return data[:, :, :2]    # return xy flow only


# ============================================================
# Base WAFT Dataset Class
# ============================================================

class WAFTDataset(data.Dataset):
    """
    A simple WAFT-friendly dataset:
      returns:
        img1:  (3,H,W) float32, 0–255
        img2:  (3,H,W) float32, 0–255
        flow:  (2,H,W) float32, optical flow in pixels
        occl:  (1,H,W) float32, 0/1 (optional)
        valid: (1,H,W) float32 mask (1=valid flow)
    """

    def __init__(self, augmentor=None):
        self.augmentor = augmentor
        self.image_pairs = []
        self.flow_list = []
        self.occl_list = []

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):

        im1_path, im2_path = self.image_pairs[idx]
        flow_path = self.flow_list[idx] if self.flow_list else None
        occl_path = self.occl_list[idx] if self.occl_list else None

        # images
        im1 = cv2.imread(im1_path)[:, :, ::-1]   # BGR→RGB
        im2 = cv2.imread(im2_path)[:, :, ::-1]

        # convert to float
        im1 = im1.astype(np.float32)
        im2 = im2.astype(np.float32)

        # flow
        if flow_path is not None:
            if flow_path.endswith('.flo'):
                flow = read_flo(flow_path)
            elif flow_path.endswith('.pfm'):
                flow = read_pfm(flow_path)
            else:
                raise ValueError(f"Unknown flow format: {flow_path}")
        else:
            H, W, _ = im1.shape
            flow = np.zeros((H, W, 2), np.float32)

        # occlusion mask
        if occl_path is not None:
            occl = cv2.imread(occl_path, cv2.IMREAD_UNCHANGED)
            occl = (occl.astype(np.float32) / 255.0)
            if occl.ndim == 2:
                occl = occl[..., None]
        else:
            occl = np.zeros((flow.shape[0], flow.shape[1], 1), np.float32)

        valid = (np.abs(flow).sum(axis=2, keepdims=True) < 1e9).astype(np.float32)

        # augmentation
        if self.augmentor is not None:
            im1, im2, flow, valid, occl = self.augmentor(im1, im2, flow, valid, occl)

        # convert to torch CHW
        im1 = einops.rearrange(torch.from_numpy(im1), "H W C -> C H W").float()
        im2 = einops.rearrange(torch.from_numpy(im2), "H W C -> C H W").float()
        flow = einops.rearrange(torch.from_numpy(flow), "H W C -> C H W").float()
        occl = einops.rearrange(torch.from_numpy(occl), "H W C -> C H W").float()
        valid = einops.rearrange(torch.from_numpy(valid), "H W C -> C H W").float()

        return im1, im2, flow, valid, occl


# ============================================================
# FlyingThings3D — WAFT version
# ============================================================

class FlyingThings3D_WAFT(WAFTDataset):
    def __init__(self, root, split='TRAIN', subset='clean', augmentor=None):
        super().__init__(augmentor)

        img_root = os.path.join(root, f"frames_{subset}pass", split)
        flow_root = os.path.join(root, "optical_flow", split)

        dirs = sorted(glob(os.path.join(img_root, "*/*")))
        for d in dirs:
            left = os.path.join(d, "left")
            imgs = sorted(glob(os.path.join(left, "*.png")))
            for i in range(len(imgs)-1):
                self.image_pairs.append([imgs[i], imgs[i+1]])

        flow_dirs = sorted(glob(os.path.join(flow_root, "*/*")))
        for d in flow_dirs:
            left = os.path.join(d, "into_future/left")
            flows = sorted(glob(os.path.join(left, "*.pfm")))
            self.flow_list.extend(flows[0:len(self.image_pairs)])

        # occlusions optional
        self.occl_list = [None] * len(self.image_pairs)


# ============================================================
# Sintel WAFT dataset
# ============================================================

class Sintel_WAFT(WAFTDataset):
    def __init__(self, root, dstype="clean", augmentor=None):
        super().__init__(augmentor)

        img_root = os.path.join(root, "training", dstype)
        flow_root = os.path.join(root, "training", "flow")
        occl_root = os.path.join(root, "training", "occlusions")

        scenes = sorted(os.listdir(img_root))
        for s in scenes:
            imgs = sorted(glob(os.path.join(img_root, s, "*.png")))
            flows = sorted(glob(os.path.join(flow_root, s, "*.flo")))
            occls = sorted(glob(os.path.join(occl_root, s, "*.png")))

            for i in range(len(imgs)-1):
                self.image_pairs.append([imgs[i], imgs[i+1]])
                self.flow_list.append(flows[i])
                self.occl_list.append(occls[i])


# ============================================================
# Kubric WAFT dataset (flowou format)
# ============================================================

def read_flowou(path):
    """ Load Kubric .flowou.png : (flow_x,flow_y,occl,uncertainty). """
    data = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    data = einops.rearrange(data, "H W C -> C H W")
    flow = data[:2].astype(np.float32)
    occl = data[2:3].astype(np.float32) / 65535.0
    sigma = data[3:4].astype(np.float32) / 65535.0
    return flow, occl, sigma


class Kubric_WAFT(WAFTDataset):
    def __init__(self, root, split="train", augmentor=None):
        super().__init__(augmentor)

        scenes = sorted(os.listdir(os.path.join(root, split)))
        for s in scenes:
            img_dir = os.path.join(root, split, s, "images")
            flow_dir = os.path.join(root, split, s, "flowou")

            imgs = sorted(glob(os.path.join(img_dir, "*.png")))
            flows = sorted(glob(os.path.join(flow_dir, "*.flowou.png")))

            for i in range(len(imgs)-1):
                self.image_pairs.append([imgs[i], imgs[i+1]])
                self.flow_list.append(flows[i+1])  # flow_0001.png is flow from 0→1
                self.occl_list.append(flows[i+1])  # we decode occl from same file

    def __getitem__(self, idx):
        im1_path, im2_path = self.image_pairs[idx]
        flow_path = self.flow_list[idx]

        # images
        im1 = cv2.imread(im1_path)[:, :, ::-1].astype(np.float32)
        im2 = cv2.imread(im2_path)[:, :, ::-1].astype(np.float32)

        flow, occl, sigma = read_flowou(flow_path)  # shapes: (2,H,W), (1,H,W), (1,H,W)
        flow = einops.rearrange(flow, "C H W -> H W C")
        occl = einops.rearrange(occl, "C H W -> H W C")
        sigma = einops.rearrange(sigma, "C H W -> H W C")

        valid = (np.abs(flow).sum(axis=2, keepdims=True) < 1e9).astype(np.float32)

        if self.augmentor:
            im1, im2, flow, valid, occl = self.augmentor(im1, im2, flow, valid, occl)

        # convert to torch
        im1 = einops.rearrange(torch.from_numpy(im1), "H W C -> C H W").float()
        im2 = einops.rearrange(torch.from_numpy(im2), "H W C -> C H W").float()
        flow = einops.rearrange(torch.from_numpy(flow), "H W C -> C H W").float()
        occl = einops.rearrange(torch.from_numpy(occl), "H W C -> C H W").float()
        sigma = einops.rearrange(torch.from_numpy(sigma), "H W C -> C H W").float()
        valid = einops.rearrange(torch.from_numpy(valid), "H W C -> C H W").float()

        return im1, im2, flow, valid, occl, sigma
