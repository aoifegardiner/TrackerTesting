import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from MFT.config import load_config
from MFT.point_tracking import convert_to_point_tracking
import MFT.utils.io as io_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Run MFT-WAFT on SurgT dataset")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to a SurgT video folder (with frames)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Where to save the tracking results (e.g. .npy)")
    parser.add_argument("--config", type=str, default="MFT/MFT_files/configs/MFT_cfg.py",
                        help="Path to MFT-WAFT config file")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load WAFT/MFT config
    config = load_config(args.config)
    tracker = config.tracker_class(config)

    # Collect frames from SurgT sequence
    frame_paths = sorted(list(Path(args.input_dir).glob("*.jpg")))
    frames = [cv2.imread(str(f)) for f in frame_paths]

    # Initialize tracker
    tracker.init(frames[0])
    queries = np.array([[x, y] for y in range(0, frames[0].shape[0], 30)
                                  for x in range(0, frames[0].shape[1], 30)])

    # Track over sequence
    results = []
    for frame in frames[1:]:
        result = tracker.track(frame, queries)
        coords, occlusions = convert_to_point_tracking(result, queries)
        results.append({"coords": coords, "occlusions": occlusions})

    # Save results in benchmark-readable format
    np.save(args.output_file, results)

if __name__ == "__main__":
    main()
