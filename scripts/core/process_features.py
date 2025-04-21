import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# Mediapipe 초기화
mp_pose = mp.solutions.pose.Pose(static_image_mode=True)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def extract_pose(image):
    result = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks:
        return np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark]).flatten()
    else:
        return None

def extract_face(image):
    result = mp_face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if result.multi_face_landmarks:
        # EAR 계산을 위해 눈 일부만 추출 (예: 왼쪽 눈 33, 160, 158, 133, 153, 144)
        eye_landmarks = [33, 160, 158, 133, 153, 144]
        pts = [result.multi_face_landmarks[0].landmark[i] for i in eye_landmarks]
        EAR = np.linalg.norm([pts[1].x - pts[5].x, pts[1].y - pts[5].y]) / (
              np.linalg.norm([pts[0].x - pts[3].x, pts[0].y - pts[3].y]) + 1e-6)
        return EAR
    else:
        return None

def extract_velocity(centers):
    velocities = [0.0]
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i-1][0]
        dy = centers[i][1] - centers[i-1][1]
        velocities.append(np.sqrt(dx**2 + dy**2))
    return velocities

def process_folder(folder_path):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    pose_features = []
    face_features = []
    centers = []

    for fname in tqdm(frame_files, desc=f"Processing {os.path.basename(folder_path)}"):
        fpath = os.path.join(folder_path, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"❌ 이미지 로드 실패: {fpath}")
            continue

        # Pose
        pose = extract_pose(img)
        if pose is not None and pose.shape == (66,):  # Expecting 33 landmarks * 2
            pose_features.append(pose)
            x_coords = pose[::2]
            y_coords = pose[1::2]
            center = [np.mean(x_coords), np.mean(y_coords)]
        else:
            pose_features.append(np.zeros(66))
            center = [0, 0]
        centers.append(center)

        # Face
        ear = extract_face(img)
        face_features.append({
            "frame": fname,
            "eye_aspect_ratio": float(ear) if ear is not None else None
        })

    # Save pose
    np.save(os.path.join(folder_path, "pose_features.npy"), np.array(pose_features))

    # Save face
    with open(os.path.join(folder_path, "face_features.json"), 'w') as f:
        json.dump(face_features, f, indent=2)

    # Save velocity
    velocities = extract_velocity(centers)
    valid_frame_names = [f["frame"] for f in face_features]
    with open(os.path.join(folder_path, "velocity_features.csv"), 'w') as f:
        f.write("frame,velocity\n")
        for i, fname in enumerate(valid_frame_names):
            vel = velocities[i] if i < len(velocities) else 0.0
            f.write(f"{fname},{vel:.6f}\n")

def run_all(root_folder):
    subfolders = sorted([os.path.join(root_folder, d) for d in os.listdir(root_folder)
                         if os.path.isdir(os.path.join(root_folder, d))])
    for folder in subfolders:
        process_folder(folder)

if __name__ == "__main__":
    run_all("/Users/jiwonkim/Desktop/GradProj/sober_file")
    run_all("/Users/jiwonkim/Desktop/GradProj/drunk_file")