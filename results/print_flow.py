import pickle

# Path to your results file
pkl_path = "./results/case1_video1_left/case1_video1_left_predictions_waft.pkl"  # <-- adjust path if needed

# Load the .pkl file
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

frame_indices = data["frame_indices"]
predicted_bboxes = data["predicted_bboxes"]

# Print results for frames between 500 and 600
print(f"Tracking results for frames 500–600 from {pkl_path}:\n")
for idx, frame in zip(frame_indices, predicted_bboxes):
    if 500 <= idx <= 600:
        print(f"Frame {idx}: {frame}")
