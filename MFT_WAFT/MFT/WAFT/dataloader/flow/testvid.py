import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class testvid(Dataset):
    def __init__(self, root='/Datasets/TestVideo', transform=None):
        self.root = root
        self.transform = transform or transforms.ToTensor()

        # Explicitly store the video path
        self.video_path = os.path.join(self.root, 'test_dot.mp4')

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video not found at {self.video_path}")

    def __len__(self):
        # Only one video
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("This dataset only has one video.")

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 2:
            raise RuntimeError(f"Video {self.video_path} has less than 2 frames!")

        frames = []

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not read frame {frame_idx} from {self.video_path}")

            # Convert BGR -> RGB -> PIL -> Tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(frame_pil)
            frames.append(frame_tensor)

        cap.release()

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        return frames
            