import os
import shutil
import random
from tqdm import tqdm

def downsample_dataset(source_dir, target_dir, max_samples=None, seed=42):
    os.makedirs(target_dir, exist_ok=True)
    subfolders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])

    if max_samples:
        random.seed(seed)
        subfolders = random.sample(subfolders, min(max_samples, len(subfolders)))

    for folder in tqdm(subfolders, desc=f"Copying to {os.path.basename(target_dir)}"):
        src_path = os.path.join(source_dir, folder)
        tgt_path = os.path.join(target_dir, folder)
        os.makedirs(tgt_path, exist_ok=True)

        for file in os.listdir(src_path):
            if file.endswith((".npy", ".json", ".csv")):
                shutil.copy(os.path.join(src_path, file), os.path.join(tgt_path, file))

    print(f"✅ Copied {len(subfolders)} folders to {target_dir}")


downsample_dataset(
    source_dir="/Users/jiwonkim/Desktop/GradProj/drunk_file",
    target_dir="/Users/jiwonkim/Desktop/GradProj/downsampled_drunk_file",
    max_samples=74
)

downsample_dataset(
    source_dir="/Users/jiwonkim/Desktop/GradProj/sober_file",
    target_dir="/Users/jiwonkim/Desktop/GradProj/downsampled_sober_file",
    max_samples=None
)