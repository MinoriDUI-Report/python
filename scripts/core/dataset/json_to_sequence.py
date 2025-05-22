import os
import json
import numpy as np

json_dir = "/Users/jiwonkim/Desktop/GradProj/json_features"
output_path = "/Users/jiwonkim/Desktop/GradProj/final_sequence.npy"

json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

sequence_data = []

for json_file in json_files:
    with open(os.path.join(json_dir, json_file)) as f:
        frame_objects = json.load(f)

    frame_features = []
    for obj in frame_objects:
        feature_vector = [
            obj["x_min"],
            obj["y_min"],
            obj["x_max"],
            obj["y_max"],
            obj["x_center"],
            obj["y_center"],
            obj["area"]
        ]
        frame_features.append(feature_vector)

    if len(frame_features) == 0:
        padded_features = np.zeros((1, 7))
    else:
        padded_features = np.array(frame_features)

    sequence_data.append(padded_features)

# ▶ padding: 모든 프레임에 대해 max_objects 맞추기
max_objects = max(f.shape[0] for f in sequence_data)
padded_sequence = []
for frame in sequence_data:
    pad_size = max_objects - frame.shape[0]
    if pad_size > 0:
        pad = np.zeros((pad_size, 7))
        padded_frame = np.vstack([frame, pad])
    else:
        padded_frame = frame
    padded_sequence.append(padded_frame)

final_sequence = np.array(padded_sequence, dtype=np.float32)  # shape: (T, max_objects, 7)

# 저장
np.save(output_path, final_sequence)
print(f"✅ Saved final_sequence.npy to {output_path}")
print("shape:", final_sequence.shape)