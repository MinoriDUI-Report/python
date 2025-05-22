# === realtime_inference_test.py ===
import os
import json
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt

# === 사용자 지정 경로 ===
FOLDER_PATH = "/Users/jiwonkim/Desktop/GradProj/drunk_file/drunk_01"  # 테스트 영상의 feature 저장 폴더
ONNX_MODEL_PATH = "/Users/jiwonkim/Desktop/GradProj/onnx/lstm_model.onnx"  # ONNX 모델 경로
WINDOW_LEN = 10
STRIDE = 2

# === Feature 불러오기 ===
def load_features(folder_path):
    pose = np.load(os.path.join(folder_path, "pose_features.npy"))
    with open(os.path.join(folder_path, "face_features.json")) as f:
        face_json = json.load(f)
    face = np.array([item.get("eye_aspect_ratio", 0.0) for item in face_json]).reshape(-1, 1)
    vel = pd.read_csv(os.path.join(folder_path, "velocity_features.csv"))["velocity"].fillna(0.0).values.reshape(-1, 1)

    T = min(pose.shape[0], face.shape[0], vel.shape[0])
    return np.hstack([pose[:T], face[:T], vel[:T]])

# === 슬라이딩 윈도우 ===
def sliding_window(seq, L, S):
    windows = []
    if seq.shape[0] < L:
        return []
    for start in range(0, seq.shape[0] - L + 1, S):
        windows.append(seq[start:start+L])
    return np.stack(windows)

# === ONNX 추론 실행 ===
def run_onnx_inference(X_windows, onnx_model_path):
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: X_windows.astype(np.float32)})[0]
    pred_classes = np.argmax(preds, axis=1)
    return pred_classes

# === 실행 ===
if __name__ == "__main__":
    print("🔍 Loading feature data...")
    features = load_features(FOLDER_PATH)
    print("✅ Feature shape:", features.shape)

    print("🧱 Applying sliding window...")
    windows = sliding_window(features, WINDOW_LEN, STRIDE)
    print("🪟 Window count:", len(windows))

    if len(windows) == 0:
        print("❌ Not enough frames for sliding window.")
        exit()

    print("🚀 Running inference...")
    predictions = run_onnx_inference(windows, ONNX_MODEL_PATH)

    print("🧪 Predictions:", predictions)
    print("🥂 Drunk ratio:", np.mean(predictions))

    # 간단한 시각화
    plt.plot(predictions, marker='o')
    plt.title("Drunk/Sober Prediction per Window")
    plt.xlabel("Window Index")
    plt.ylabel("Prediction (0: Sober, 1: Drunk)")
    plt.ylim(-0.5, 1.5)
    plt.grid(True)
    plt.show()