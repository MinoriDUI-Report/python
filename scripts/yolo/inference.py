import os
import zipfile
import shutil
from ultralytics import YOLO


def extract_dataset(zip_path, extract_dir):
    """
    주어진 ZIP 파일을 지정된 디렉토리로 압축 해제합니다.
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Dataset extracted to '{extract_dir}'.")


def train_baseline(data_config, epochs, project_dir):
    """
    YOLOv8n 모델을 사용하여 Baseline 학습을 수행합니다.
    결과는 지정된 project_dir에 저장됩니다.
    """
    model = YOLO('yolov8n.pt')
    print("Starting Baseline training...")
    results = model.train(data=data_config, epochs=epochs, imgsz=640, project=project_dir)
    print(f"Baseline training complete. Results stored in '{project_dir}'.")
    return results


def train_mixup(data_config, epochs, project_dir):
    """
    YOLOv8n 모델을 사용하여 MixUp 증강을 적용한 학습을 수행합니다.
    결과는 지정된 project_dir에 저장됩니다.
    """
    model = YOLO('yolov8n.pt')
    print("Starting MixUp training...")
    results = model.train(data=data_config, epochs=epochs, imgsz=640, augment=True, mixup=0.2, project=project_dir)
    print(f"MixUp training complete. Results stored in '{project_dir}'.")
    return results


def zip_results(result_dir, output_zip):
    """
    지정된 결과 폴더(result_dir)를 ZIP 파일로 압축합니다.
    결과 파일은 output_zip.zip으로 생성됩니다.
    """
    shutil.make_archive(output_zip, 'zip', result_dir)
    print(f"Results zipped as '{output_zip}.zip'.")


def main():
    # 1. 데이터셋 압축 해제
    zip_path = 'car_driver_seat.v13i.yolov8.zip'
    extract_dir = 'dataset'
    extract_dataset(zip_path, extract_dir)

    # 2. Baseline 학습 실행 (Mosaic 기본 증강)
    data_config = os.path.join(extract_dir, 'data.yaml')
    epochs = 50
    baseline_project = 'runs/baseline'
    train_baseline(data_config, epochs, baseline_project)

    # 3. Baseline 결과 폴더 압축 (예: baseline_results.zip)
    zip_results(baseline_project, 'baseline_results')

    # 4. MixUp 적용 학습 실행 (Mosaic + MixUp)
    mixup_project = 'runs/mixup'
    train_mixup(data_config, epochs, mixup_project)

    # 5. MixUp 결과 폴더 압축 (예: mixup_results.zip)
    zip_results(mixup_project, 'mixup_results')


if __name__ == '__main__':
    main()