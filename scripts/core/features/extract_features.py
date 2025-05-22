# === extract_features.py ===
import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp

# === 경로 설정 ===
HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME, "Desktop", "GradProj")
SOBER_DIR = os.path.join(BASE_DIR, "sober_file")
DRUNK_DIR = os.path.join(BASE_DIR, "drunk_file")

# === Mediapipe 초기화 ===
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose(static_image_mode=True)
face = mp_face.FaceMesh(static_image_mode=True)


# === 처리 함수 ===
def process_folder(folder_path):
    img_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")])

    pose_features = []
    face_features = []
    velocity_list = []
    prev_landmark = [0.0, 0.0]  # 초기값 (x, y)

    for i, img_file in enumerate(img_files):
        image = cv2.imread(img_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # === Pose ===
        pose_result = pose.process(image_rgb)
        if pose_result.pose_landmarks is not None:
            landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_result.pose_landmarks.landmark])
            pose_features.append(landmarks.flatten())

            # === velocity 계산 ===
            dx = pose_result.pose_landmarks.landmark[0].x - prev_landmark[0]
            dy = pose_result.pose_landmarks.landmark[0].y - prev_landmark[1]
            velocity = (dx ** 2 + dy ** 2) ** 0.5
            prev_landmark = [pose_result.pose_landmarks.landmark[0].x, pose_result.pose_landmarks.landmark[0].y]
        else:
            pose_features.append(np.zeros(33 * 3))  # 33 landmarks x (x,y,z)
            velocity = 0.0  # 검출 실패 시 velocity 0

        velocity_list.append(velocity)

        # === Face (EAR 계산) ===
        face_result = face.process(image_rgb)
        if face_result.multi_face_landmarks:
            landmarks = face_result.multi_face_landmarks[0].landmark
            # EAR 계산용 좌표 (left eye: 33, 160, 158, 133, 153, 144)
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            A = np.linalg.norm([left_eye[1].x - left_eye[5].x, left_eye[1].y - left_eye[5].y])
            B = np.linalg.norm([left_eye[2].x - left_eye[4].x, left_eye[2].y - left_eye[4].y])
            C = np.linalg.norm([left_eye[0].x - left_eye[3].x, left_eye[0].y - left_eye[3].y])
            EAR = (A + B) / (2.0 * C) if C != 0 else 0.0
        else:
            EAR = 0.0  # 얼굴 못 잡으면 0

        face_features.append({"eye_aspect_ratio": EAR})

    # === 저장 ===
    np.save(os.path.join(folder_path, "pose_features.npy"), np.array(pose_features))
    with open(os.path.join(folder_path, "face_features.json"), "w") as f:
        json.dump(face_features, f)
    pd.DataFrame({"velocity": velocity_list}).to_csv(os.path.join(folder_path, "velocity_features.csv"), index=False)

    print(f"✅ {os.path.basename(folder_path)}: feature saved")


# === 메인 실행 ===
if __name__ == "__main__":
    for base_dir in [SOBER_DIR, DRUNK_DIR]:
        for folder in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                process_folder(folder_path)
            except Exception as e:
                print(f"⚠️ Error in {folder}: {e}")