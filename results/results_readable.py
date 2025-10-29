import pickle, json

with open("results/case1_video1_left/case1_video1_left_predictions.pkl", "rb") as f:
    data = pickle.load(f)

with open("results/case1_video1_left/predictions_readable.json", "w") as f:
    json.dump(data, f, indent=2)
