"""A3 + A4 + B1 + B2: dựng dataset chuẩn Ultralytics. Chỉ đọc archive, sửa nhãn trên bản copy.
Dùng: python3 -I build_dataset.py <archive_dir> <splits_dir> <dst_dir>
"""
import json, os, subprocess, sys

src, splits, dst = sys.argv[1:4]
if os.path.exists(dst):
    sys.exit(f"Dừng: {dst} đã tồn tại, không ghi đè")


def fix(stem, lines):
    if stem == "1965":  # B1: 2 box trùng nhau, đã xác nhận bằng mắt là lớp 23
        new = [l for l in lines if l.split()[0] != "14"]
        assert len(new) == len(lines) - 1
    elif stem == "2095":  # B1: như trên
        new = [l for l in lines if l.split()[0] != "34"]
        assert len(new) == len(lines) - 1
    elif stem == "0248":  # B2: cắt box vào trong mép phải ảnh
        new, n = [], 0
        for l in lines:
            p = l.split()
            x, w = float(p[1]), float(p[3])
            if p[0] == "49" and x + w / 2 > 1:
                left = x - w / 2
                p[1], p[3] = f"{(left + 1) / 2:.6f}", f"{1 - left:.6f}"
                n += 1
            new.append(" ".join(p))
        assert n == 1
    else:
        return lines
    print(f"sửa {stem}:\n  trước: {lines}\n  sau:   {new}")
    return new


for split in ("train", "val", "test"):
    os.makedirs(f"{dst}/images/{split}")
    os.makedirs(f"{dst}/labels/{split}")
    for s in open(f"{splits}/{split}.txt").read().split():
        # cp -c: bản clone APFS, không tốn thêm ổ đĩa và độc lập với ảnh gốc
        subprocess.run(["cp", "-c", f"{src}/images/{s}.jpg", f"{dst}/images/{split}/{s}.jpg"], check=True)
        lines = fix(s, [l.strip() for l in open(f"{src}/labels/{s}.txt") if l.strip()])
        with open(f"{dst}/labels/{split}/{s}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

names = open(f"{src}/classes_vie.txt", encoding="utf-8").read().splitlines()
assert len(names) == 52
with open(f"{dst}/data.yaml", "w", encoding="utf-8") as f:
    # Không ghi "path": Ultralytics lấy thư mục chứa file yaml làm gốc, nên chạy được cả trên Colab/Kaggle
    f.write("train: images/train\nval: images/val\ntest: images/test\nnames:\n")
    for i, n in enumerate(names):
        f.write(f"  {i}: {json.dumps(n, ensure_ascii=False)}\n")
print("xong")
