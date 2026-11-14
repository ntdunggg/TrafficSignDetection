"""Dựng bộ ảnh cắt cho hướng A (giai đoạn 0.2). Chỉ đọc dataset YOLO, ghi ra thư mục mới.

Mỗi box trong data/vn_signs_yolo thành một ảnh PNG, cắt bằng crop_box() (xem docstring của hàm).
Ghi kèm:
  <out>/manifest.csv       mỗi ảnh cắt một dòng: path,split,class_id,stem,line,group,box_w_px,box_h_px
  du_lieu/crops_info.json  md5 nhãn, số ảnh cắt từng tập, mean/std của train ở 64x64 (có trong git)

Dùng (chạy từ gốc repo):
  python -m src.data.make_crops --yolo data/vn_signs_yolo --out data/vn_signs_crops
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

from src.utils import label_md5

SPLITS = ("train", "val", "test")
MARGIN = 1.2  # cạnh hình vuông = 1,2 × cạnh dài của box, tức lề 10% mỗi phía (giống GTSRB)


def crop_box(img, box, margin=MARGIN):
    """Cắt một biển báo ra khỏi ảnh gốc.

    img: PIL.Image RGB (ảnh gốc, kích thước bất kỳ).
    box: (xc, yc, w, h) chuẩn hoá 0..1 theo định dạng YOLO.
    Trả về PIL.Image vuông, cạnh = round(margin × max(w, h)) pixel (ít nhất 1 pixel).

    - Vuông, tâm là tâm box: đổi kích thước về 64×64 sau này không làm méo biển.
    - Phần vượt ra ngoài ảnh được đệm bằng cách lặp lại pixel ở mép (edge padding).
    Hệ hai giai đoạn B1 (Anh Đức) dùng lại đúng hàm này cho box do YOLO dự đoán.
    """
    W, H = img.size
    xc, yc, w, h = box
    side = max(1, round(margin * max(w * W, h * H)))
    x0 = round(xc * W - side / 2)
    y0 = round(yc * H - side / 2)
    x1, y1 = x0 + side, y0 + side
    a = np.asarray(img)
    pad = ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)), (0, 0))
    if any(p for pair in pad[:2] for p in pair):
        a = np.pad(a, pad, mode="edge")
        x0, y0 = x0 + pad[1][0], y0 + pad[0][0]
    return Image.fromarray(a[y0:y0 + side, x0:x0 + side])


def read_groups(path):
    with open(path, newline="") as f:
        return {r["stem"]: r["group"] for r in csv.DictReader(f)}


def train_mean_std(rows, out_dir, size=64):
    """Mean/std từng kênh của ảnh cắt train sau khi đổi về size×size (giáo trình 4.2)."""
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for r in rows:
        if r["split"] != "train":
            continue
        with Image.open(os.path.join(out_dir, r["path"])) as im:
            a = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float64) / 255
        s += a.sum((0, 1))
        s2 += (a ** 2).sum((0, 1))
        n += size * size
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return [round(float(v), 6) for v in mean], [round(float(v), 6) for v in std]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", default="data/vn_signs_yolo")
    ap.add_argument("--out", default="data/vn_signs_crops")
    ap.add_argument("--groups", default="du_lieu/splits/groups.csv")
    ap.add_argument("--info", default="du_lieu/crops_info.json")
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"Dừng: {args.out} đã tồn tại, không ghi đè. Xoá thư mục đó nếu muốn dựng lại.")
    groups = read_groups(args.groups)
    rows = []
    for sp in SPLITS:
        os.makedirs(os.path.join(args.out, sp))
        for f in sorted(os.listdir(os.path.join(args.yolo, "labels", sp))):
            if not f.endswith(".txt"):
                continue
            stem = f[:-4]
            lines = open(os.path.join(args.yolo, "labels", sp, f)).read().splitlines()
            with Image.open(os.path.join(args.yolo, "images", sp, stem + ".jpg")) as im:
                img = im.convert("RGB")
            W, H = img.size
            for ln, l in enumerate(lines):  # ln đếm từ 0, khớp khóa stem#line của catalog
                if not l.strip():
                    continue
                c, xc, yc, w, h = l.split()
                box = tuple(map(float, (xc, yc, w, h)))
                rel = f"{sp}/{stem}_{ln}.png"
                crop_box(img, box).save(os.path.join(args.out, rel))
                rows.append(dict(path=rel, split=sp, class_id=int(c), stem=stem, line=ln,
                                 group=groups[stem], box_w_px=round(box[2] * W), box_h_px=round(box[3] * H)))
        print(sp, sum(r["split"] == sp for r in rows), "ảnh cắt")

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    mean, std = train_mean_std(rows, args.out)
    info = dict(
        nguon=args.yolo,
        md5_nhan=label_md5(os.path.join(args.yolo, "labels")),
        margin=MARGIN,
        so_anh_cat={sp: sum(r["split"] == sp for r in rows) for sp in SPLITS},
        mean_std_size=64,
        mean=mean,
        std=std,
    )
    with open(args.info, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
