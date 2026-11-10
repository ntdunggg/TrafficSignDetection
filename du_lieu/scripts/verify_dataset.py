"""Kiểm tra dataset đã dựng so với archive. Chỉ đọc.
Dùng: python3 -I verify_dataset.py <archive_dir> <dataset_dir>
"""
import collections, hashlib, os, sys
import yaml

src, dst = sys.argv[1:3]
data = yaml.safe_load(open(f"{dst}/data.yaml", encoding="utf-8"))
print("data.yaml:", {k: data[k] for k in ("train", "val", "test")}, "| số tên lớp:", len(data["names"]), "| lớp 23:", data["names"][23])
assert sorted(data["names"]) == list(range(52))

seen, boxes, bad, corrupt, same_inode, diff_bytes = set(), 0, [], [], 0, 0
per = collections.Counter()
for sp in ("train", "val", "test"):
    im = {f[:-4] for f in os.listdir(f"{dst}/images/{sp}")}
    lb = {f[:-4] for f in os.listdir(f"{dst}/labels/{sp}")}
    assert im == lb, sp
    assert not (seen & im), "ảnh nằm ở 2 tập"
    seen |= im
    nb = 0
    for s in im:
        for l in open(f"{dst}/labels/{sp}/{s}.txt"):
            p = l.split()
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            nb += 1
            per[c] += 1
            if not (0 <= c < 52 and x - w / 2 >= -1e-6 and x + w / 2 <= 1 + 1e-6 and y - h / 2 >= -1e-6 and y + h / 2 <= 1 + 1e-6 and w > 0 and h > 0):
                bad.append((s, l.strip()))
        a, b = f"{src}/images/{s}.jpg", f"{dst}/images/{sp}/{s}.jpg"
        same_inode += os.stat(a).st_ino == os.stat(b).st_ino
        ba, bb = open(a, "rb").read(), open(b, "rb").read()
        diff_bytes += ba != bb
        corrupt += [s] if bb[-2:] != b"\xff\xd9" else []
    boxes += nb
    print(f"{sp:5s}: {len(im)} cặp ảnh/nhãn, {nb} box")

orig = {f[:-4] for f in os.listdir(f"{src}/images") if f.endswith(".jpg")}
empty = {s for s in orig if not open(f"{src}/labels/{s}.txt").read().strip()}
print("tổng ảnh:", len(seen), "| archive:", len(orig), "| rỗng trong archive:", len(empty), "| khớp archive - rỗng:", seen == orig - empty)
print("tổng box:", boxes, "(archive 8334, mong đợi 8332)")
print("box lỗi biên/định dạng:", bad)
print("ảnh dùng chung inode với archive:", same_inode, "| ảnh khác nội dung:", diff_bytes)
print("JPEG thiếu marker kết thúc (Ultralytics sẽ tự ghi lại):", len(corrupt), corrupt[:10])
print("lớp 14:", per[14], "lớp 23:", per[23], "lớp 34:", per[34], "lớp 49:", per[49])
for sp, s in (("1965", "1965"), ("2095", "2095"), ("0248", "0248")):
    for t in ("train", "val", "test"):
        p = f"{dst}/labels/{t}/{s}.txt"
        if os.path.exists(p):
            print(s, t, open(p).read().split("\n")[:-1])
