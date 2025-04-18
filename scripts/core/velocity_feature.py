import numpy as np


def compute_pose_center(pose_features):
    """
    주어진 pose landmark 좌표 리스트의 중심(centroid)을 계산합니다.

    Parameters:
      pose_features: list of (x,y) 튜플

    Returns:
      tuple: (center_x, center_y) 평균 값 (실수)
    """
    if pose_features is None or len(pose_features) == 0:
        return None
    xs = [pt[0] for pt in pose_features]
    ys = [pt[1] for pt in pose_features]
    return (np.mean(xs), np.mean(ys))


def compute_velocities(centers, fps=30):
    """
    연속된 프레임의 pose 중심 좌표를 기반으로, 각 프레임 간 속도를 계산합니다.

    Parameters:
      centers: list of (center_x, center_y) 중심 좌표 (순차적으로)
      fps: 초당 프레임 수

    Returns:
      list: 각 인접 프레임 간의 속도 (픽셀/초)
    """
    velocities = []
    dt = 1.0 / fps
    for i in range(1, len(centers)):
        if centers[i - 1] is None or centers[i] is None:
            velocities.append(None)
            continue
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        dist = np.sqrt(dx * dx + dy * dy)
        vel = dist / dt
        velocities.append(round(vel, 2))
    return velocities