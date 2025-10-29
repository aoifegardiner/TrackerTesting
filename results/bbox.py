import pickle

with open("results/case1_video1_left/case1_video1_left_predictions_waft.pkl", "rb") as f:
    data = pickle.load(f)

frames = data["frame_indices"]
bboxes = data["predicted_bboxes"]

for frame, box in zip(frames, bboxes):
    if 450 <= frame <= 550:
        print(f"Frame {frame}: bbox={box}")
