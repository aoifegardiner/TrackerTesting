import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_predicted_vs_gt(predictions_pkl, gt_yaml, output_dir):
    """
    Plot predicted trajectory (left camera) vs ground-truth and save to results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load predictions ---
    with open(predictions_pkl, "rb") as f:
        preds = pickle.load(f)
    frame_indices = np.array(preds["frame_indices"])
    pred_bboxes = np.array(preds["predicted_bboxes"])  # shape (N, 4): [x, y, w, h]

    # compute predicted centers
    pred_centers_x = pred_bboxes[:, 0] + pred_bboxes[:, 2] / 2
    pred_centers_y = pred_bboxes[:, 1] + pred_bboxes[:, 3] / 2

    # --- Load ground truth ---
    with open(gt_yaml, "r") as f:
        gt_data = yaml.load(f, Loader=yaml.FullLoader)

    gt_centers_x, gt_centers_y, gt_frames = [], [], []

    # Filter GT values to only frames where we have predictions
    valid_mask = np.isin(gt_frames, frame_indices)
    gt_frames_sync = gt_frames[valid_mask]
    gt_centers_x_sync = gt_centers_x[valid_mask]
    gt_centers_y_sync = gt_centers_y[valid_mask]

    # Align predictions to these frames
    pred_idx_map = {f: i for i, f in enumerate(frame_indices)}
    pred_centers_x_sync = np.array([pred_centers_x[pred_idx_map[f]] for f in gt_frames_sync])
    pred_centers_y_sync = np.array([pred_centers_y[pred_idx_map[f]] for f in gt_frames_sync])

    # Compute magnitude error per frame
    diff_x = pred_centers_x_sync - gt_centers_x_sync
    diff_y = pred_centers_y_sync - gt_centers_y_sync
    magnitude_error = np.sqrt(diff_x**2 + diff_y**2)

    # --- Plot magnitude error ---
    plt.figure(figsize=(10, 5))
    plt.plot(gt_frames_sync, magnitude_error, label="Center Error Magnitude")
    plt.xlabel("Frame Number")
    plt.ylabel("Error Magnitude (pixels)")
    plt.title("Tracking Error Magnitude per Frame")
    plt.grid(True)
    plt.legend()

    out_path_mag = output_dir / "trajectory_error_magnitude_case2.png"
    plt.savefig(out_path_mag, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✅ Magnitude error plot saved to: {out_path_mag}")

    # Iterate through each frame index and entry
    for frame_id, frame_entry in gt_data.items():
        # Each entry is (visible, difficult, (bbox_left, bbox_right))
        if not isinstance(frame_entry, (list, tuple)) or len(frame_entry) < 3:
            continue

        visible, difficult, bbxs = frame_entry
        if bbxs is None or difficult or not visible:
            continue

        bbox_left = bbxs[0]  # left camera bbox
        if bbox_left is not None and len(bbox_left) == 4:
            x, y, w, h = bbox_left
            gt_centers_x.append(x + w / 2)
            gt_centers_y.append(y + h / 2)
            gt_frames.append(int(frame_id))


    # --- Plot trajectories ---
    plt.figure(figsize=(8, 6))
    plt.plot(gt_centers_x, gt_centers_y, "g-", label="Ground Truth (left)")
    plt.plot(pred_centers_x, pred_centers_y, "r--", label="MFT Prediction")
    plt.scatter(gt_centers_x[0], gt_centers_y[0], c="g", marker="o", label="GT start")
    plt.scatter(pred_centers_x[0], pred_centers_y[0], c="r", marker="x", label="Pred start")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("MFT Left-Camera Trajectory vs Ground Truth")
    plt.legend()
    plt.grid(True)

    out_path = output_dir / "trajectory_comparison_case2.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✅ Trajectory plot saved to: {out_path}")

if __name__ == "__main__":
    plot_predicted_vs_gt(
        predictions_pkl="results/case1_video1_left/case1_video1_left_predictions_waft_new.pkl",
        gt_yaml="/Workspace/agardiner_STIR_submission/data/test/case_2/1/gt_rectified_0.yaml",
        output_dir="results/case1_video1_left"
    )
