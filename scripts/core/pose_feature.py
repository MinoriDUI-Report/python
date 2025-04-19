import cv2
import mediapipe as mp

# Mediapipe Pose 초기화 (이미지 처리 용)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


def extract_pose_features(frame):
    """
    주어진 프레임에서 Mediapipe Pose 솔루션을 활용하여,
    33개의 landmark 중 필요한 좌표를 추출합니다.

    여기서는 예시로 (x, y) 좌표만 추출하며, 전체 좌표는 flatten하여 34차원 벡터로 처리할 수 있도록 구성합니다.

    Returns:
      list: 각 landmark의 (x, y) 좌표 리스트 (예: [(x1, y1), (x2, y2), ...]) 또는 None (검출 실패 시)
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if not results.pose_landmarks:
        return None
    h, w, _ = frame.shape
    pose_coords = []
    for lm in results.pose_landmarks.landmark:
        x = round(lm.x * w, 2)
        y = round(lm.y * h, 2)
        pose_coords.append((x, y))
    return pose_coords