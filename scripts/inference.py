# 1. 필요한 패키지 설치
# !pip install ultralytics opencv-python-headless

# 2. 데이터셋 파일 업로드 (로컬에서 car_driver_seat.v13i.yolov8.zip 파일 선택)
# from google.colab import files
# uploaded = files.upload()

# 3. 업로드한 zip 파일을 'dataset' 폴더에 압축 해제
# !unzip -q car_driver_seat.v13i.yolov8.zip -d dataset

# 4. YOLOv8 모델 학습 (Baseline)
# from ultralytics import YOLO

# 사전 학습된 모델 로드 (여기서는 yolov8n.pt 사용)
# model = YOLO('yolov8n.pt')
# 데이터셋 설정 파일 경로 (압축 해제된 dataset 폴더 내 data.yaml)
# data_config = '/content/dataset/data.yaml'
# 학습 에폭 설정
# epochs = 50

# 학습 실행 (기본 증강, Mosaic 적용)
# results = model.train(data=data_config, epochs=epochs, imgsz=640)

# 5. 학습 완료 후 모델 가중치(best.pt) 다운로드
# from google.colab import files
# files.download('runs/detect/train/weights/best.pt')

# 6. 학습 결과 CSV 파일 다운로드
# files.download('runs/detect/train/results.csv')

# 7. 전체 결과 폴더를 압축하여 로컬로 다운로드
# import shutil
# runs/detect/train 폴더를 results.zip으로 압축
# shutil.make_archive("/content/results", "zip", "runs/detect/train")
# files.download('/content/results.zip')