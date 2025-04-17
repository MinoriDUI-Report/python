import os
import glob
import struct


def extract_seq(seq_path, output_dir):
    with open(seq_path, 'rb') as f:
        content = f.read()

    header_size = 1024
    data = content[header_size:]
    i = 0
    frame = 0
    os.makedirs(output_dir, exist_ok=True)

    while i < len(data):
        try:
            size = struct.unpack('<I', data[i:i + 4])[0]
            jpg = data[i + 4:i + 4 + size]
            with open(os.path.join(output_dir, f"frame_{frame:05d}.jpg"), 'wb') as out:
                out.write(jpg)
            i += 4 + size
            frame += 1
        except:
            break

    print(f"✅ {frame} frames extracted to {output_dir}")


def batch_extract_all(start_index=4):
    # 🔁 Caltech Train sets
    base_sets = [
        "/Users/jiwonkim/Downloads/data_and_labels/Train/set00/set00",
        "/Users/jiwonkim/Downloads/data_and_labels/Train/set01/set01",
        "/Users/jiwonkim/Downloads/data_and_labels/Train/set02/set02",
        "/Users/jiwonkim/Downloads/data_and_labels/Train/set03/set03",
        "/Users/jiwonkim/Downloads/data_and_labels/Train/set04/set04",
        "/Users/jiwonkim/Downloads/data_and_labels/Train/set05/set05"
    ]

    target_base = "/Users/jiwonkim/Desktop/GradProj/sober_file"
    seq_index = start_index

    for set_path in base_sets:
        seq_files = sorted(glob.glob(os.path.join(set_path, "V*.seq")))
        for seq_path in seq_files:
            out_dir = os.path.join(target_base, f"sober_{seq_index:02d}")
            print(f"▶️ Extracting {seq_path} → {out_dir}")
            extract_seq(seq_path, out_dir)
            seq_index += 1


if __name__ == "__main__":
    batch_extract_all(start_index=4)