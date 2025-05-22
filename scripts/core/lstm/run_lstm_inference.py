import numpy as np
import onnxruntime as ort

# 1. 입력 로드
X_raw = np.load("/Users/jiwonkim/Desktop/GradProj/final_sequence.npy")  # shape: (T, N_objects, 7)

# 2. 프레임 평균 (-> shape: (T, 7))
X_avg = np.mean(X_raw, axis=1)

# 3. 슬라이딩 윈도우로 시퀀스 구성 (-> shape: (N, 10, 7))
WINDOW_SIZE = 10
STRIDE = 2

windows = []
for start in range(0, len(X_avg) - WINDOW_SIZE + 1, STRIDE):
    windows.append(X_avg[start:start+WINDOW_SIZE])

X_input = np.stack(windows).astype(np.float32)  # (N, 10, 7)

# 4. ONNX 세션 실행
onnx_path = "/Users/jiwonkim/Desktop/GradProj/onnx/lstm_model.onnx"
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 5. 추론
outputs = session.run([output_name], {input_name: X_input})
predictions = np.argmax(outputs[0], axis=1)  # shape: (N,)

# 6. 저장
np.savetxt("/Users/jiwonkim/Desktop/GradProj/predictions.csv", predictions, fmt="%d")
print("✅ Saved predictions to predictions.csv, shape:", predictions.shape)