# 1. 필요한 패키지는 터미널에서 직접 설치:
# pip install ultralytics opencv-python-headless

# 2. 데이터셋 파일은 이미 로컬에 있다고 가정하고, 압축 해제 (zipfile 모듈 사용)
import zipfile
import os

zip_path = 'car_driver_seat.v13i.yolov8.zip'
extract_dir = 'dataset'
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 3. YOLOv8 모델 학습 (Baseline)
from ultralytics import YOLO

# 사전 학습된 모델 로드 (yolov8n.pt 파일이 로컬에 있어야 함)
model = YOLO('yolov8n.pt')
# 데이터셋 설정 파일 경로 (압축 해제된 dataset 폴더 내 data.yaml)
data_config = os.path.join(extract_dir, 'data.yaml')
# 학습 에폭 설정
epochs = 50

# 학습 실행 (기본 증강, Mosaic 적용)
results = model.train(data=data_config, epochs=epochs, imgsz=640)

# 4. 학습 완료 후 결과 파일을 확인하거나 복사
# 로컬에서는 파일들이 이미 지정된 디렉토리에 저장됩니다.
print("학습 완료, 결과 폴더: runs/detect/train/")