import os
import cv2

def load_sequence(root, seq_name, version="clean"):
    seq_path = os.path.join(root, "training", version, seq_name)
    frames = sorted(os.listdir(seq_path))
    images = []

    for f in frames:
        img = cv2.imread(os.path.join(seq_path, f))
        images.append(img)

    return images

import numpy as np

def read_flo(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise Exception('Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    return np.resize(data, (h, w, 2))
