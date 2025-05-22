import cv2
import numpy as np
from ultralytics import YOLO

# 설정
VIDEO_PATH = "/Users/jiwonkim/Desktop/GradProj/input_video.mp4"
OUTPUT_PATH = "/Users/jiwonkim/Desktop/GradProj/output_overlay.mp4"
YOLO_MODEL_PATH = "/Users/jiwonkim/Desktop/GradProj/last.pt"
PREDICTION_CSV = "/Users/jiwonkim/Desktop/GradProj/predictions.csv"

# 색상 정의 (BGR)
COLOR_SOBER = (0, 255, 0)     # 초록
COLOR_DRUNK = (0, 255, 255)   # 노랑
COLOR_DANGER = (0, 0, 255)    # 빨강

# 슬라이딩 윈도우 설정
WINDOW_LEN = 10
STRIDE = 2

# 예측 결과 로드 (0=sober, 1=drunk)
predictions = np.loadtxt(PREDICTION_CSV, delimiter=",", dtype=int)

# 비디오 설정
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# YOLO 모델 로드
model = YOLO(YOLO_MODEL_PATH)

# 예측값을 프레임 단위로 확장
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_labels = np.zeros(total_frames, dtype=int)
for i, pred in enumerate(predictions):
    start = i * STRIDE
    end = min(start + WINDOW_LEN, total_frames)
    frame_labels[start:end] = pred

# 프레임 처리 루프
frame_idx = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    label = frame_labels[frame_idx]
    label_text = "DRUNK" if label == 1 else "SOBER"

    # YOLO 탐지
    results = model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy
    classes = results.boxes.cls.cpu().numpy().astype(int)

    person_detected = 4 in classes
    door_opened = 1 in classes or 3 in classes

    # 음주자 상태 판단
    if label == 1:
        if person_detected and door_opened:
            person_color = COLOR_DANGER
        else:
            person_color = COLOR_DRUNK
    else:
        person_color = COLOR_SOBER

    # 상태 텍스트 오버레이
    cv2.putText(frame, label_text, (30, 40), font, 1.2, person_color, 3)

    # 객체별 박스 그리기
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cls_name = model.names[cls_id]

        # 사람만 상태에 따라 색 변경
        if cls_id == 4:  # person
            color = person_color
        else:
            color = COLOR_SOBER  # 항상 초록색

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, cls_name, (x1, y1 - 5), font, 0.6, color, 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("✅ output_overlay.mp4 생성 완료")