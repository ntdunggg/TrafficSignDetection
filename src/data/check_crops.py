"""Kiểm tra bộ ảnh cắt (giai đoạn 0.2) và tạo trang ảnh ghép để xem bằng mắt.

Kiểm tra:
  1. Số ảnh cắt từng tập = số box (train 5621, val 1078, test 1630).
  2. Không stem nào nằm ở hai tập; mỗi group chỉ ở một tập.
  3. Số ảnh cắt theo lớp và tập khớp du_lieu/splits/class_counts.csv, sau khi tính box 1193#2 (test) từ lớp 28 sang 30.
  4. Mọi file trong manifest tồn tại, ảnh vuông, cạnh = round(1,2 × cạnh dài box).
Ảnh ghép: <crops>/_xem/lop_XX.jpg, 10 ảnh cắt ngẫu nhiên mỗi lớp (seed 0), phóng lên 64×64.

Dùng: python -m src.data.check_crops --crops data/vn_signs_crops
"""
import argparse
import collections
import csv
import os
import random

from PIL import Image, ImageDraw

SPLITS = ("train", "val", "test")
EXPECTED = {"train": 5621, "val": 1078, "test": 1630}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/vn_signs_crops")
    ap.add_argument("--class-counts", default="du_lieu/splits/class_counts.csv")
    ap.add_argument("--per-class", type=int, default=10)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.crops, "manifest.csv"), newline="")))
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("OK   " if cond else "LỖI ") + msg)

    n = collections.Counter(r["split"] for r in rows)
    check(dict(n) == EXPECTED, f"1. Số ảnh cắt {dict(n)}")

    stem_sp = collections.defaultdict(set)
    group_sp = collections.defaultdict(set)
    for r in rows:
        stem_sp[r["stem"]].add(r["split"])
        group_sp[r["group"]].add(r["split"])
    check(all(len(v) == 1 for v in stem_sp.values()), f"2a. {len(stem_sp)} stem, không stem nào ở hai tập")
    check(all(len(v) == 1 for v in group_sp.values()), f"2b. {len(group_sp)} nhóm frame, không nhóm nào ở hai tập")

    got = collections.Counter((int(r["class_id"]), r["split"]) for r in rows)
    exp = collections.Counter()
    for r in csv.DictReader(open(args.class_counts, newline="")):
        for sp in SPLITS:
            exp[(int(r["id"]), sp)] = int(r[sp])
    # class_counts.csv đếm trên nhãn archive lúc chia tập, trước các lần sửa nhãn sau đây:
    exp[(14, "train")] -= 1  # B1: bỏ box trùng 1965 (lớp 14), du_lieu/log/build_log.txt
    exp[(34, "train")] -= 1  # B1: bỏ box trùng 2095 (lớp 34), du_lieu/log/build_log.txt
    exp[(16, "train")] -= 3  # catalog: xoá 2088#1, 2278#1, 2279#1, du_lieu/log/apply_catalog_log.txt
    exp[(28, "test")] -= 1  # C5: 1193#2 chuyển từ lớp 28 sang 30
    exp[(30, "test")] += 1
    diff = {k: (got[k], exp[k]) for k in set(got) | set(exp) if got[k] != exp[k]}
    check(not diff, f"3. Số ảnh cắt theo lớp khớp class_counts.csv sau các sửa nhãn đã ghi. Lệch: {diff}")

    bad, sides = [], []
    for r in rows:
        p = os.path.join(args.crops, r["path"])
        if not os.path.exists(p):
            bad.append((r["path"], "không có file"))
            continue
        with Image.open(p) as im:
            w, h = im.size
        want = max(1, round(1.2 * max(int(r["box_w_px"]), int(r["box_h_px"]))))
        if w != h or abs(w - want) > 1:  # box_w_px đã làm tròn nên cho lệch 1 px
            bad.append((r["path"], (w, h, want)))
        sides.append(w)
    check(not bad, f"4. File tồn tại, ảnh vuông, cạnh đúng 1,2 × cạnh dài box. Lỗi: {bad[:5]}")

    sides.sort()
    q = lambda f: sides[int(f * (len(sides) - 1))]
    print(f"Cạnh ảnh cắt (px): nhỏ nhất {sides[0]}, p25 {q(.25)}, trung vị {q(.5)}, p75 {q(.75)}, lớn nhất {sides[-1]}; "
          f"≤ 64 px: {sum(s <= 64 for s in sides) / len(sides):.1%}")

    out = os.path.join(args.crops, "_xem")
    os.makedirs(out, exist_ok=True)
    by_class = collections.defaultdict(list)
    for r in rows:
        by_class[int(r["class_id"])].append(r)
    rng = random.Random(0)
    for c, rs in sorted(by_class.items()):
        pick = rng.sample(rs, min(args.per_class, len(rs)))
        sheet = Image.new("RGB", (len(pick) * 68, 84), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(pick):
            with Image.open(os.path.join(args.crops, r["path"])) as im:
                sheet.paste(im.convert("RGB").resize((64, 64), Image.BILINEAR), (i * 68 + 2, 2))
            d.text((i * 68 + 2, 68), f"{r['stem']}#{r['line']}", fill="black")
        sheet.save(os.path.join(out, f"lop_{c:02d}.jpg"), quality=90)
    print(f"Ảnh ghép: {out}/lop_00.jpg ... lop_51.jpg")
    print("KẾT LUẬN:", "đạt" if ok else "CÓ LỖI")


if __name__ == "__main__":
    main()
