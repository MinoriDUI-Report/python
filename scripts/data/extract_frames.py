import cv2
import os


def extract_frames_from_video(video_path, output_folder, interval=3):
    """
    영상 파일에서 일정 간격(interval)마다 프레임을 추출하여 output_folder에 저장합니다.

    Parameters:
    - video_path (str): 처리할 영상 파일의 전체 경로
    - output_folder (str): 추출된 프레임을 저장할 폴더 경로
    - interval (int): 몇 프레임마다 저장할지 (기본값: 3)
    """
    # 파일 존재 여부 확인
    if not os.path.exists(video_path):
        print("Error: Video file not found:", video_path)
        return

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 비디오 캡쳐 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file:", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing '{video_path}': Total frames = {total_frames}")

    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"✅ {saved_count} frames saved in {output_folder}")


# -------------------------
# 기본 경로 설정
input_folder = "/Users/jiwonkim/Desktop/GradProj/drunk"
output_base_folder = "/Users/jiwonkim/Desktop/GradProj/drunk_file"

# 입력 폴더 내에 있는 모든 .mp4 파일 찾기
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]
print("Detected video files:", video_files)

# 각 비디오 파일에 대해 프레임 추출 진행
for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    # 파일명에서 확장자를 제거한 폴더 이름 생성 (예: "sober_01")
    folder_name = os.path.splitext(video_file)[0]
    output_folder = os.path.join(output_base_folder, folder_name)
    print("Processing video file:", video_path)
    extract_frames_from_video(video_path, output_folder, interval=3)