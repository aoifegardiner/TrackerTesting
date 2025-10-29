import cv2
import yaml
import os

# Paths (adjust to your setup)
video_path = "/Workspace/agardiner_STIR_submission/data/validation/case_1/1/video.mp4"
gt_path = "/Workspace/agardiner_STIR_submission/data/validation/case_1/1/gt_rectified_0.yaml"

# Example anchors to test (frame indices from config)
anchors = [0, 10, 20, 30, 50, 100]  # replace with your actual anchors

def load_gt(gt_path):
    """Load GT data safely, converting tuples if needed."""
    with open(gt_path, "r") as f:
        gt_data = yaml.load(f, Loader=yaml.FullLoader)

    # Convert tuples to lists if present
    for frame, ann in gt_data.items():
        if isinstance(ann, dict) and "bbox" in ann:
            ann["bbox"] = list(ann["bbox"])
    return gt_data

def is_valid_bbox(bbox, frame_w, frame_h, ann):
    """Check if bbox is valid."""
    if bbox is None:
        return False, "No bbox in GT"

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False, "Zero size bbox"
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return False, "BBox out of frame"

    if ann.get("visible", True) is False:
        return False, "Marked not visible"
    if ann.get("difficult", False) is True:
        return False, "Marked difficult"

    return True, "Valid"

def check_anchors(video_path, gt_path, anchors, save_frames=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video size: {frame_w}x{frame_h}, total frames: {total_frames}")

    gt_data = load_gt(gt_path)

    for anchor in anchors:
        print(f"\n🔍 Checking anchor at frame {anchor}...")

        if str(anchor) not in gt_data:
            print(f"⚠️ No GT entry for frame {anchor}")
            continue

        ann = gt_data[str(anchor)]
        bbox = ann.get("bbox", None)

        valid, reason = is_valid_bbox(bbox, frame_w, frame_h, ann)
        print(f" GT bbox: {bbox}, status: {reason}")

        # Try to grab frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, anchor)
        ret, frame = cap.read()
        if not ret:
            print(" ⚠️ Could not read video frame at anchor")
            continue

        if save_frames:
            out_dir = "anchor_checks"
            os.makedirs(out_dir, exist_ok=True)
            frame_copy = frame.copy()
            if bbox:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0) if valid else (0, 0, 255), 2)
            out_path = os.path.join(out_dir, f"anchor_{anchor}.jpg")
            cv2.imwrite(out_path, frame_copy)
            print(f" Saved annotated frame to {out_path}")

    cap.release()

if __name__ == "__main__":
    check_anchors(video_path, gt_path, anchors)
