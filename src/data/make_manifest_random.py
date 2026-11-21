"""Dựng manifest_random.csv cho Thí nghiệm 5 (bản R: chia ngẫu nhiên từng ảnh, như split gốc của archive).

Kế hoạch: ke_hoach/nguoi_1_dung_du_lieu_va_A0-A2.md, mục "Bản ảnh cắt chia ngẫu nhiên cho Thí nghiệm 5".
- Test: ảnh trong archive/split_dataset/test_files.txt.
- Train và val: ảnh trong train_files.txt. Lấy ngẫu nhiên 15% số ảnh làm val (seed 0, chia theo ảnh).
  Cố ý giữ kiểu chia ngẫu nhiên này để thấy rò rỉ (NHAT_KY mục A1).
- Bỏ 25 ảnh nhãn rỗng (du_lieu/splits/excluded_empty.txt). Thực tế split gốc không chứa ảnh nào trong 25 ảnh này.

Không cắt lại ảnh: dùng đúng các ảnh cắt của data/vn_signs_crops (cùng nhãn đã sửa, cùng md5 nhãn với bản G),
chỉ đổi cột split. Cột path giữ nguyên, nên manifest_random.csv phải nằm cùng thư mục với manifest.csv.
Bản R và bản G chỉ khác nhau ở cách chia tập.

Ghi ra:
  data/vn_signs_crops/manifest_random.csv   cùng cột với manifest.csv (git ignore)
  du_lieu/crops_info_random.json             mean/std tính trên train của bản R (có trong git)
Chỉ đọc archive và manifest.csv, không sửa.

Dùng (chạy từ gốc repo):
  python -m src.data.make_manifest_random
Train trên bản R:
  python -m src.train ... --manifest data/vn_signs_crops/manifest_random.csv --info du_lieu/crops_info_random.json
"""
import argparse
import collections
import csv
import json
import os
import random
import sys

from src.data.make_crops import train_mean_std

SPLITS = ("train", "val", "test")


def read_list(path):
    """Đọc danh sách tên ảnh, bỏ đuôi .jpg nếu có."""
    out = []
    for line in open(path):
        s = line.strip()
        if s:
            out.append(s[:-4] if s.endswith(".jpg") else s)
    return out


def neighbor_leak(stems_eval, stems_train):
    """Số ảnh trong stems_eval có ảnh liền số (stem ± 1) nằm trong stems_train (cách đo ở NHAT_KY mục A1)."""
    tr = {int(s) for s in stems_train}
    return sum(1 for s in stems_eval if int(s) - 1 in tr or int(s) + 1 in tr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-split", default="/Users/tdu/Downloads/archive/split_dataset")
    ap.add_argument("--manifest", default="data/vn_signs_crops/manifest.csv")
    ap.add_argument("--empty", default="du_lieu/splits/excluded_empty.txt")
    ap.add_argument("--out", default="data/vn_signs_crops/manifest_random.csv")
    ap.add_argument("--info-g", default="du_lieu/crops_info.json")
    ap.add_argument("--info", default="du_lieu/crops_info_random.json")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"Dừng: {args.out} đã tồn tại, không ghi đè. Xoá file đó nếu muốn dựng lại.")

    empty = set(read_list(args.empty))
    train_files = read_list(os.path.join(args.archive_split, "train_files.txt"))
    test_files = read_list(os.path.join(args.archive_split, "test_files.txt"))
    print(f"split gốc: train_files {len(train_files)}, test_files {len(test_files)}; "
          f"ảnh nhãn rỗng có trong split gốc: {len(empty & set(train_files + test_files))}")
    train_pool = sorted(s for s in train_files if s not in empty)
    test_set = {s for s in test_files if s not in empty}

    # Chia train/val theo ảnh, ngẫu nhiên, seed cố định. Sắp xếp trước khi xáo để kết quả không phụ thuộc thứ tự file.
    rng = random.Random(args.seed)
    shuffled = train_pool[:]
    rng.shuffle(shuffled)
    n_val = round(args.val_frac * len(shuffled))
    val_set = set(shuffled[:n_val])
    train_set = set(shuffled[n_val:])
    split_of = {**{s: "train" for s in train_set}, **{s: "val" for s in val_set}, **{s: "test" for s in test_set}}

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    missing = sorted({r["stem"] for r in rows} - set(split_of))
    if missing:
        sys.exit(f"Dừng: {len(missing)} ảnh trong manifest.csv không có trong split gốc, ví dụ {missing[:5]}")
    new_rows = [{**r, "split": split_of[r["stem"]]} for r in rows]

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(new_rows)

    # Kiểm tra
    stems = {sp: {r["stem"] for r in new_rows if r["split"] == sp} for sp in SPLITS}
    n_crop = {sp: sum(r["split"] == sp for r in new_rows) for sp in SPLITS}
    n_cls = {sp: len({r["class_id"] for r in new_rows if r["split"] == sp}) for sp in SPLITS}
    assert not (stems["train"] & stems["val"]) and not (stems["train"] & stems["test"]) and not (stems["val"] & stems["test"])
    groups = collections.defaultdict(set)
    for r in new_rows:
        groups[r["group"]].add(r["split"])
    split_groups = sum(len(v) > 1 for v in groups.values())

    g_rows = rows
    g_stems = {sp: {r["stem"] for r in g_rows if r["split"] == sp} for sp in SPLITS}
    leak = {
        "R_test": (neighbor_leak(stems["test"], stems["train"]), len(stems["test"])),
        "R_val": (neighbor_leak(stems["val"], stems["train"]), len(stems["val"])),
        "G_test": (neighbor_leak(g_stems["test"], g_stems["train"]), len(g_stems["test"])),
        "G_val": (neighbor_leak(g_stems["val"], g_stems["train"]), len(g_stems["val"])),
    }
    crop_leak_test = sum(1 for r in new_rows if r["split"] == "test"
                         and (str(int(r["stem"]) - 1).zfill(4) in stems["train"]
                              or str(int(r["stem"]) + 1).zfill(4) in stems["train"]))

    mean, std = train_mean_std(new_rows, os.path.dirname(os.path.abspath(args.out)))
    with open(args.info_g, encoding="utf-8") as f:
        info_g = json.load(f)
    info = dict(
        nguon=args.manifest,
        manifest=args.out,
        cach_chia="split gốc của archive (ngẫu nhiên từng ảnh); val = 15% ảnh của train_files, seed 0",
        md5_nhan=info_g["md5_nhan"],
        margin=info_g["margin"],
        so_anh={sp: len(stems[sp]) for sp in SPLITS},
        so_anh_cat=n_crop,
        so_lop=n_cls,
        mean_std_size=64,
        mean=mean,
        std=std,
    )
    with open(args.info, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False))
    print(f"nhóm frame bị tách ra nhiều tập: {split_groups}/{len(groups)}")
    for k, (a, b) in leak.items():
        print(f"ảnh {k} có ảnh train liền số: {a}/{b} ({a / b:.1%})")
    print(f"ảnh cắt test (R) có ảnh gốc liền số với một ảnh train: {crop_leak_test}/{n_crop['test']} "
          f"({crop_leak_test / n_crop['test']:.1%})")


if __name__ == "__main__":
    main()
