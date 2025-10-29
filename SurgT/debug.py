import cv2
import torch
from SurgT.src.MFT_WAFT import MFTWAFTTrackerSurgT

def debug_tracker():
    # Path to your video file
    video_path = "/Workspace/agardiner_STIR_submission/data/validation/case_1/1/video.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    # Read first two frames
    ret1, im1 = cap.read()
    ret2, im2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        raise RuntimeError("Could not read two frames from video")

    # Fake bbox (whole image) for debugging
    H, W, _ = im1.shape
    bbox = [0, 0, W-1, H-1]

    tracker = MFTWAFTTrackerSurgT(im1, im2, bbox, bbox)
    bbox1, bbox2 = tracker.tracker_update(im1, im2)

    print("Predicted bbox1:", bbox1)
    print("Predicted bbox2:", bbox2)

if __name__ == "__main__":
    debug_tracker()
