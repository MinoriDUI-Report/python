# === augment_sequences.py ===
#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd

# === 설정 ===
HOME       = os.path.expanduser("~")
BASE_DIR   = os.path.join(HOME, "Desktop", "GradProj")
SOBER_DIR  = os.path.join(BASE_DIR, "sober_file")
DRUNK_DIR  = os.path.join(BASE_DIR, "drunk_file")
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented_dataset")

# 슬라이딩 윈도우 파라미터
WINDOW_LEN = 10  # 시퀀스 길이 L
STRIDE     = 2   # 스트라이드 S

# === 시퀀스 로드 함수 ===
def load_sequence(folder_path):
    # Pose
    pose = np.load(os.path.join(folder_path, "pose_features.npy"))
    # Face
    with open(os.path.join(folder_path, "face_features.json"), "r") as f:
        face_json = json.load(f)
    face = np.array([item.get("eye_aspect_ratio", 0.0) for item in face_json]).reshape(-1, 1)
    # Velocity
    df_vel = pd.read_csv(os.path.join(folder_path, "velocity_features.csv"))
    vel = df_vel["velocity"].fillna(0.0).values.reshape(-1, 1)
    # 최소 길이에 맞춰 자르기
    T = min(pose.shape[0], face.shape[0], vel.shape[0])
    return np.hstack([pose[:T], face[:T], vel[:T]])  # (T, features)

# === 슬라이딩 윈도우 함수 ===
def sliding_windows(seq, L, S):
    windows = []
    if seq.shape[0] < L:
        pad_len = L - seq.shape[0]
        pad_block = np.tile(seq[-1:], (pad_len, 1))
        windows.append(np.vstack([seq, pad_block]))
    else:
        for start in range(0, seq.shape[0] - L + 1, S):
            windows.append(seq[start:start + L])
    return windows

# === 메인 실행 (증강) ===
if __name__ == '__main__':
    X_list, y_list = [], []

    for label, base_dir in [(0, SOBER_DIR), (1, DRUNK_DIR)]:
        for folder in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                seq = load_sequence(folder_path)
                if seq.shape[0] < WINDOW_LEN:
                    print(f"⚠️ Skipping {folder}: insufficient sequence length {seq.shape[0]} < {WINDOW_LEN}")
                    continue
            except Exception as e:
                print(f"⚠️ Skipping {folder}: {e}")
                continue

            for win in sliding_windows(seq, WINDOW_LEN, STRIDE):
                X_list.append(win)
                y_list.append(label)

    X_aug = np.stack(X_list)  # (N, L, features)
    y_aug = np.array(y_list)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "augmented.npz")
    np.savez(out_file, X=X_aug, y=y_aug)
    print(f"✅ 증강 완료: {X_aug.shape[0]}개의 시퀀스 (길이 {WINDOW_LEN}) → {out_file}")
