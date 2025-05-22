import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score
import os

# === LSTM 모델 정의 ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# === 데이터 로드 ===
data = np.load("/Users/jiwonkim/Desktop/GradProj/augmented_dataset/augmented.npz")
X, y = data["X"], data["y"]

# 데이터 나누기
num_samples = len(X)
train_size = int(0.8 * num_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Tensor 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# === 학습 ===
model = LSTMClassifier(input_size=X.shape[2])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# === 평가 ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = torch.argmax(y_pred, dim=1).numpy()

print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Recall:", recall_score(y_test, y_pred_class))
print("Precision:", precision_score(y_test, y_pred_class))

# === ONNX Export ===
onnx_path = "/Users/jiwonkim/Desktop/GradProj/lstm_model.onnx"
dummy_input = torch.randn(1, X.shape[1], X.shape[2])
torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
print(f"✅ Exported ONNX model to {onnx_path}")