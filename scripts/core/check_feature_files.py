import os
import json
import numpy as np
import pandas as pd

def check_feature_files(folder_path):
    result = []
    subfolders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])

    for subfolder in subfolders:
        full_path = os.path.join(folder_path, subfolder)
        pose_path = os.path.join(full_path, "pose_features.npy")
        face_path = os.path.join(full_path, "face_features.json")
        velocity_path = os.path.join(full_path, "velocity_features.csv")

        status = {
            "folder": subfolder,
            "pose_exists": os.path.isfile(pose_path),
            "face_exists": os.path.isfile(face_path),
            "velocity_exists": os.path.isfile(velocity_path),
            "pose_valid": False,
            "face_valid": False,
            "velocity_valid": False,
            "pose_length": None,
            "face_length": None,
            "velocity_length": None,
        }

        if status["pose_exists"]:
            try:
                pose = np.load(pose_path, allow_pickle=True)
                status["pose_valid"] = isinstance(pose, np.ndarray) and pose.shape[0] > 0
                status["pose_length"] = pose.shape[0]
            except:
                pass

        if status["face_exists"]:
            try:
                with open(face_path, 'r') as f:
                    face = json.load(f)
                    status["face_valid"] = isinstance(face, list) and len(face) > 0
                    status["face_length"] = len(face)
            except:
                pass

        if status["velocity_exists"]:
            try:
                df = pd.read_csv(velocity_path)
                status["velocity_valid"] = len(df) > 0 and "velocity" in df.columns
                status["velocity_length"] = len(df)
            except:
                pass

        result.append(status)

    return pd.DataFrame(result)

if __name__ == "__main__":
    root_dir = "/Users/jiwonkim/Desktop/GradProj/sober_file"
    df = check_feature_files(root_dir)
    print(df)
    df.to_csv("feature_integrity_check.csv", index=False)