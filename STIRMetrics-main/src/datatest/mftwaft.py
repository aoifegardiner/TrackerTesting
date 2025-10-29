from pathlib import Path
#HERE = Path(__file__).resolve().parent  
#PROJECT_ROOT = HERE.parent.parent.parent
#print("PROJECT_ROOT:", PROJECT_ROOT)
#CONFIG = PROJECT_ROOT / "MFT_WAFT" / "MFT" / "MFT_files" / "configs" / "MFT_cfg.py"
PROJECT_ROOT = Path("/workspace/agardiner_STIR_submission")
CONFIG = PROJECT_ROOT / "MFT_WAFT" / "MFT" / "MFT_files" / "configs" / "MFT_cfg.py"

from MFT.config import load_config
from MFT.point_tracking import convert_to_point_tracking
from MFT.waft import FlowOUTrackingResult
from MFT.waft import WAFTWrapper


import cv2
import torch
import numpy as np

class MFTWAFTTracker:
    def __init__(self):
        # Load config and initialize tracker
        config = load_config(CONFIG)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.modeltype = "MFTWAFT"
        self.tracker = config.tracker_class(config)
        self.initialized = False
        self.queries = None
        self.scale = 2
        self.internalimsize = (1280 // self.scale, 1024 // self.scale)

    def preprocess_frame(self, frame):
        """
        Convert input frame to HxWxC numpy array and resize to internalimsize.
        Accepts numpy arrays or torch tensors.
        """
        # Remove batch dimension if present
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.ndim == 4 and frame.shape[0] == 1:  # (1,H,W,C)
            frame = frame[0]
        elif frame.ndim != 3:
            raise ValueError(f"Expected frame with 3 dimensions HxWxC, got shape {frame.shape}")

        # Ensure size tuple is valid
        H_int, W_int = self.internalimsize[1], self.internalimsize[0]
        if not (isinstance(H_int, int) and isinstance(W_int, int)):
            raise ValueError(f"Invalid internalimsize: {self.internalimsize}")

        # Resize
        frame_resized = cv2.resize(frame, (W_int, H_int))  # note: (width, height)
        return frame_resized

    def trackpoints2D(self, pointlist, frames):
        """
        Track points between frames using WAFT

        pointlist: numpy array (N,2) with x,y coordinates
        frames: tuple/list of two frames (frame0, frame1), HxWxC numpy arrays
        """

        frame0 = self.preprocess_frame(frames[0])
        frame1 = self.preprocess_frame(frames[1])

        device = self.device
        if isinstance(pointlist, torch.Tensor):
            queries = pointlist.float().to(device) / self.scale
        else:
            queries = torch.from_numpy(np.array(pointlist)).float().to(device) / self.scale

        #queries = torch.from_numpy(pointlist).float().to(device) / self.scale

        if not self.initialized:
            self.tracker.init(frame0)
            self.initialized = True

        flow_result = self.tracker.track(frame1, queries)

        #print("track() returned:", type(flow_result))
        import inspect

        print("WAFTWrapper file:", inspect.getfile(type(self.tracker)))
        #print("DEBUG: WAFTWrapper defined in", inspect.getfile(WAFTWrapper))

        if not isinstance(flow_result, FlowOUTrackingResult):
            raise TypeError("WAFTWrapper.track must return a FlowOUTrackingResult")

        # Warp points using flow
        coords_internal = flow_result.warp_forward_points(queries)

        coords = coords_internal * self.scale

        #print("First query before:", queries[0].cpu().numpy())
        #print("First query after :", coords_internal[0].cpu().numpy())
        #print("Coords :", coords[0].cpu().numpy())


        return coords, flow_result.occlusion