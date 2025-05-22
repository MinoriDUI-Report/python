import numpy as np
import os

# 설정
INPUT_PATH = "/Users/jiwonkim/Desktop/GradProj/final_sequence.npy"  # 7차원 feature가 저장된 파일
OUTPUT_PATH = "/Users/jiwonkim/Desktop/GradProj/augmented_dataset/single_test.npz"
WINDOW_SIZE = 10
STRIDE = 2
LABEL = 1  # drunk = 1, sober = 0 (적절히 변경)

# 1. 데이터 로드 (T, N_objects, 7)
X_raw = np.load(INPUT_PATH)
print("✅ Loaded final_sequence.npy, shape:", X_raw.shape)

# 2. 프레임별 평균 → (T, 7)
X_avg = np.mean(X_raw, axis=1)
print("✅ Averaged per frame, shape:", X_avg.shape)

# 3. 슬라이딩 윈도우 → (N, 10, 7)
windows = []
for i in range(0, len(X_avg) - WINDOW_SIZE + 1, STRIDE):
    windows.append(X_avg[i:i + WINDOW_SIZE])

X_seq = np.stack(windows)
print("✅ Sliding window complete, shape:", X_seq.shape)

# 4. 라벨 생성 (모두 drunk 가정)
y_seq = np.full(X_seq.shape[0], LABEL, dtype=np.int64)

# 5. 저장
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.savez(OUTPUT_PATH, X=X_seq, y=y_seq)
print(f"✅ Saved augmented data to {OUTPUT_PATH}")