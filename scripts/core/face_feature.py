import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Face Mesh 초기화 (테스트 모드)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def extract_face_features(frame):
    """
    주어진 프레임에서 Mediapipe Face Mesh를 사용해 얼굴 landmark를 추출하고,
    왼쪽 눈의 landmark를 기반으로 Eye Aspect Ratio(EAR)를 계산합니다.

    Returns:
      dict: {'eye_aspect_ratio': value} 또는 None (얼굴 검출 실패 시)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # 왼쪽 눈 landmark 인덱스 예시 (mediapipe 기준)
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    try:
        left_eye = [pts[i] for i in left_eye_indices]
    except IndexError:
        return None

    def euclidean(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    A = euclidean(left_eye[1], left_eye[5])
    B = euclidean(left_eye[2], left_eye[4])
    C = euclidean(left_eye[0], left_eye[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return {"eye_aspect_ratio": round(ear, 4)}