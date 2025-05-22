#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# === 경로 설정 ===
HOME       = os.path.expanduser("~")
BASE_DIR   = os.path.join(HOME, "Desktop", "GradProj")
AUG_PATH   = os.path.join(BASE_DIR, "augmented_dataset", "augmented.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset_augmented")

def main():
    # 1) 증강 데이터 로드
    data = np.load(AUG_PATH, allow_pickle=True)
    X_aug, y_aug = data["X"], data["y"]

    # 2) 클래스 균형 조정 (Drunk ↓ to match Sober count)
    idx_sober = np.where(y_aug == 0)[0]
    idx_drunk = np.where(y_aug == 1)[0]
    n_sober  = len(idx_sober)
    np.random.seed(42)
    if len(idx_drunk) > n_sober:
        idx_drunk = np.random.choice(idx_drunk, size=n_sober, replace=False)
    idx_all = np.concatenate([idx_sober, idx_drunk])
    np.random.shuffle(idx_all)

    X_bal = X_aug[idx_all]
    y_bal = y_aug[idx_all]

    # 3) Train/Val/Test split (80:10:10)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    # 4) 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(OUTPUT_DIR, "val.npz"),   X=X_val,   y=y_val)
    np.savez(os.path.join(OUTPUT_DIR, "test.npz"),  X=X_test,  y=y_test)

    # 5) 분포 요약 출력
    summary = pd.DataFrame({
        "Split":       ["train", "val", "test"],
        "Samples":     [len(y_train), len(y_val), len(y_test)],
        "Drunk_Ratio": [float(y_train.mean()), float(y_val.mean()), float(y_test.mean())]
    })
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()