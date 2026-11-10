"""Áp dụng quyết định sau catalog (2026-11-09) lên data/vn_signs_yolo. Không đụng archive.
- Xoay trái 90° 39 ảnh dọc 1319-1357 và đổi tọa độ nhãn theo.
- Xoá box 2088#1, 2278#1, 2279#1 (lớp 16, bạn đánh dấu Sai).
- Đổi tên lớp 26 trong data.yaml.
Sao lưu nhãn, ảnh và data.yaml trước khi sửa. Dừng nếu đã chạy rồi.
Dùng: python3 -I apply_catalog.py <dataset_dir> <backup_dir>
"""
import os, shutil, sys
from PIL import Image

ds, bk = sys.argv[1:3]
if os.path.exists(bk):
    sys.exit(f"Dừng: {bk} đã tồn tại, có vẻ đã chạy rồi")

ROTATE = [f"{i:04d}" for i in range(1319, 1358)]
DELETE = {"2088": (1, "16"), "2278": (1, "16"), "2279": (1, "16")}
NEW26 = "Chú ý chướng ngại vật, vòng tránh hai bên"


def where(s):
    hits = [sp for sp in ("train", "val", "test") if os.path.exists(f"{ds}/images/{sp}/{s}.jpg")]
    assert len(hits) == 1, s
    return hits[0]


targets = {s: where(s) for s in ROTATE + list(DELETE)}
for s, sp in targets.items():
    os.makedirs(f"{bk}/labels/{sp}", exist_ok=True)
    shutil.copy2(f"{ds}/labels/{sp}/{s}.txt", f"{bk}/labels/{sp}/{s}.txt")
for s in ROTATE:
    sp = targets[s]
    os.makedirs(f"{bk}/images/{sp}", exist_ok=True)
    shutil.copy2(f"{ds}/images/{sp}/{s}.jpg", f"{bk}/images/{sp}/{s}.jpg")
shutil.copy2(f"{ds}/data.yaml", f"{bk}/data.yaml")

for s in ROTATE:
    sp = targets[s]
    p = f"{ds}/images/{sp}/{s}.jpg"
    with Image.open(p) as im:
        assert im.size == (540, 960), (s, im.size)
        out = im.rotate(90, expand=True)  # xoay ngược chiều kim đồng hồ
        out.save(p + ".tmp", "JPEG", quality=95)
    os.replace(p + ".tmp", p)
    lines = []
    for l in open(f"{ds}/labels/{sp}/{s}.txt"):
        c, x, y, w, h = l.split()
        x, y, w, h = map(float, (x, y, w, h))
        lines.append(f"{c} {y:.6f} {1 - x:.6f} {h:.6f} {w:.6f}")
    open(f"{ds}/labels/{sp}/{s}.txt", "w").write("\n".join(lines) + "\n")
print("xoay", len(ROTATE), "ảnh")

for s, (ln, cls) in DELETE.items():
    sp = targets[s]
    lines = [l.strip() for l in open(f"{ds}/labels/{sp}/{s}.txt") if l.strip()]
    assert lines[ln].split()[0] == cls, (s, lines[ln])
    print(f"xoá {s}#{ln}: {lines[ln]}")
    del lines[ln]
    open(f"{ds}/labels/{sp}/{s}.txt", "w").write("\n".join(lines) + "\n")

y = open(f"{ds}/data.yaml", encoding="utf-8").read()
old = '  26: "Chướng ngoại vật phía trước"\n'
assert y.count(old) == 1
open(f"{ds}/data.yaml", "w", encoding="utf-8").write(y.replace(old, f'  26: "{NEW26}"\n'))
print("đổi tên lớp 26 ->", NEW26)
