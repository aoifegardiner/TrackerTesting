import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_predicted_vs_gt(predictions_pkl, gt_yaml, output_dir):
    """
    Compare predicted vs GT centers and plot:
    1. Trajectory comparison
    2. Magnitude of center error per frame
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. LOAD PREDICTIONS
    # ============================================================
    with open(predictions_pkl, "rb") as f:
        preds = pickle.load(f)

    frame_indices = np.array(preds["frame_indices"])
    pred_bboxes = np.array(preds["predicted_bboxes"])  # shape (N,4)

    pred_centers_x = pred_bboxes[:, 0] + pred_bboxes[:, 2] / 2
    pred_centers_y = pred_bboxes[:, 1] + pred_bboxes[:, 3] / 2

    # ============================================================
    # 2. LOAD GROUND TRUTH
    # ============================================================
    with open(gt_yaml, "r") as f:
        gt_data = yaml.load(f, Loader=yaml.FullLoader)

    gt_centers_x = []
    gt_centers_y = []
    gt_frames = []

    # Extract GT centers
    for frame_id, frame_entry in gt_data.items():

        if (not isinstance(frame_entry, (list, tuple))) or (len(frame_entry) < 3):
            continue

        visible, difficult, bbxs = frame_entry
        if not visible or difficult or bbxs is None:
            continue

        bbox_left = bbxs[0]  # left image bbox
        if bbox_left is None or len(bbox_left) != 4:
            continue

        x, y, w, h = bbox_left
        gt_centers_x.append(x + w / 2)
        gt_centers_y.append(y + h / 2)
        gt_frames.append(int(frame_id))

    gt_centers_x = np.array(gt_centers_x)
    gt_centers_y = np.array(gt_centers_y)
    gt_frames = np.array(gt_frames)

    # ============================================================
    # 3. FRAME ALIGNMENT
    # ============================================================
    # Keep only GT frames that appear in predictions
    valid_mask = np.isin(gt_frames, frame_indices)

    gt_frames_sync = gt_frames[valid_mask]
    gt_centers_x_sync = gt_centers_x[valid_mask]
    gt_centers_y_sync = gt_centers_y[valid_mask]

    if len(gt_frames_sync) == 0:
        print("❌ ERROR: No overlapping frames between predictions and GT!")
        print("GT frames:", gt_frames[:20])
        print("Pred frames:", frame_indices[:20])
        return

    # Map frame → index in prediction arrays
    pred_idx = {frame: i for i, frame in enumerate(frame_indices)}

    # Sync predicted centers to the matching GT frames
    pred_centers_x_sync = np.array([pred_centers_x[pred_idx[f]] for f in gt_frames_sync])
    pred_centers_y_sync = np.array([pred_centers_y[pred_idx[f]] for f in gt_frames_sync])

    # ============================================================
    # 4. MAGNITUDE ERROR COMPUTATION
    # ============================================================
    diff_x = pred_centers_x_sync - gt_centers_x_sync
    diff_y = pred_centers_y_sync - gt_centers_y_sync
    magnitude_error = np.sqrt(diff_x**2 + diff_y**2)

    # ============================================================
    # 5. PLOT MAGNITUDE ERROR
    # ============================================================
    plt.figure(figsize=(10, 5))
    plt.plot(gt_frames_sync, magnitude_error, label="Center Error Magnitude", linewidth=1.5)
    plt.xlabel("Frame Number")
    plt.ylabel("Error Magnitude (pixels)")
    plt.title("Tracking Error Magnitude per Frame")
    plt.grid(True)
    plt.legend()

    out_mag = output_dir / "trajectory_error_magnitude_2711_3.png"
    plt.savefig(out_mag, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✅ Magnitude error plot saved to: {out_mag}")

    # ============================================================
    # 6. PLOT TRAJECTORY COMPARISON
    # ============================================================
    plt.figure(figsize=(8, 6))
    plt.plot(gt_centers_x, gt_centers_y, "g-", label="Ground Truth")
    plt.plot(pred_centers_x, pred_centers_y, "r--", label="Prediction")
    plt.scatter(gt_centers_x[0], gt_centers_y[0], c="g", marker="o", label="GT Start")
    plt.scatter(pred_centers_x[0], pred_centers_y[0], c="r", marker="x", label="Pred Start")

    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Trajectory Comparison (Left Camera)")
    plt.legend()
    plt.grid(True)

    out_traj = output_dir / "trajectory_comparison_2711_3.png"
    plt.savefig(out_traj, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✅ Trajectory plot saved to: {out_traj}")


# ============================================================
# RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    plot_predicted_vs_gt(
        predictions_pkl="results/case1_video1_left/case1_video1_left_predictions_2411.pkl",
        gt_yaml="/Workspace/agardiner_STIR_submission/data/test/case_1/1/gt_rectified_0.yaml",
        output_dir="results/case1_video1_left"
    )
