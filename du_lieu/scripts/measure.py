"""Đo bằng chứng cho B3, C3, D2, D3 và tạo tấm ảnh mẫu cho B4. Chỉ đọc dataset.
Dùng: python3 -I measure.py <dataset_dir> <b4_out_prefix>
"""
import collections, io, os, random, sys
import numpy as np
from PIL import Image, ImageDraw

ds, b4 = sys.argv[1:3]
rows, files = [], []
for sp in ("train", "val", "test"):
    for f in sorted(os.listdir(f"{ds}/images/{sp}")):
        s = f[:-4]
        p = f"{ds}/images/{sp}/{f}"
        with Image.open(p) as im:
            W, H = im.size
        files.append((s, sp, W, H, os.path.getsize(p), p))
        for ln, l in enumerate(open(f"{ds}/labels/{sp}/{s}.txt")):
            q = l.split()
            rows.append((s, sp, ln, int(q[0]), float(q[3]) * W, float(q[4]) * H, W, H))

print("=== Độ phân giải ảnh")
for k, v in collections.Counter((W, H) for _, _, W, H, _, _ in files).most_common():
    print(" ", k, v)

w = np.array([r[4] for r in rows]); h = np.array([r[5] for r in rows]); L = np.array([max(r[6], r[7]) for r in rows])
print("\n=== D2: kích thước box sau khi đưa về imgsz (cạnh dài ảnh = imgsz)")
print("imgsz | median cạnh ngắn box | cạnh ngắn <8px | <16px | <32px")
for z in (640, 960, 1280):
    m = np.minimum(w, h) * z / L
    print(f"{z:5d} | {np.median(m):6.1f} px | {(m<8).mean()*100:5.1f}% | {(m<16).mean()*100:5.1f}% | {(m<32).mean()*100:5.1f}%")

small = w * h < 32 * 32
print(f"\n=== B3: box 'nhỏ' theo COCO (diện tích <32x32 px trên ảnh gốc): {small.sum()} / {len(rows)} ({small.mean()*100:.1f}%), trong {len({r[0] for r, f in zip(rows, small) if f})} ảnh")
m640 = np.minimum(w, h) * 640 / L
print(f"box có cạnh ngắn <8px ở imgsz 640: {(m640<8).sum()}, trong {len({r[0] for r, f in zip(rows, m640<8) if f})} ảnh")
cs = collections.Counter(r[3] for r, f in zip(rows, small) if f)
ca = collections.Counter(r[3] for r in rows)
print("10 lớp có tỉ lệ box nhỏ cao nhất (id: nhỏ/tổng):", [(c, f"{cs[c]}/{ca[c]}") for c in sorted(ca, key=lambda c: -cs[c] / ca[c])[:10]])

print("\n=== C3: ảnh 0961-0964 và lớp 26, 51")
for s in ("0961", "0962", "0963", "0964"):
    for r in rows:
        if r[0] == s:
            z = 960 / max(r[6], r[7])
            print(f"  {s} ({r[1]}) lớp {r[3]}: {r[4]:.0f}x{r[5]:.0f} px trên ảnh {r[6]}x{r[7]} | ở imgsz 960: {r[4]*z:.0f}x{r[5]*z:.0f} | COCO nhỏ: {r[4]*r[5]<1024}")
for c in (26, 51):
    a = [r[4] * r[5] for r in rows if r[3] == c]
    print(f"  lớp {c}: {len(a)} box, median {np.sqrt(np.median(a)):.0f}x~ px, số box COCO nhỏ: {sum(x<1024 for x in a)}")

print("\n=== D3: dung lượng")
total = sum(f[4] for f in files)
print(f"tổng ảnh: {total/1e6:.0f} MB, {len(files)} ảnh")
big = [f for f in files if max(f[2], f[3]) > 1280]
print(f"ảnh có cạnh dài >1280: {len(big)} ảnh, {sum(f[4] for f in big)/1e6:.0f} MB")
random.seed(0)
samp = random.sample(big, min(100, len(big)))
for q in (95, 90):
    old = new = 0
    for f in samp:
        im = Image.open(f[5]).convert("RGB")
        k = 1280 / max(im.size)
        b = io.BytesIO()
        im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS).save(b, "JPEG", quality=q)
        old += f[4]
        new += b.tell()
    r = new / old
    bigmb = sum(f[4] for f in big)
    print(f"resize về 1280, JPEG q{q}: mẫu {len(samp)} ảnh còn {r*100:.0f}% | ước tính tổng: {(total-bigmb*(1-r))/1e6:.0f} MB")
sample_zip = sum(os.path.getsize(f[5]) for f in samp)
import zlib
comp = sum(len(zlib.compress(open(f[5], "rb").read(), 6)) for f in samp)
print(f"nén zip (deflate) trên JPEG: còn {comp/sample_zip*100:.1f}%")

print("\n=== B4: mẫu ngẫu nhiên")
random.seed(1)
pick = random.sample(files, 24)
print(" ".join(sorted(f[0] for f in pick)))
for sheet in range(2):
    sub = pick[sheet * 12:(sheet + 1) * 12]
    cw, ch = 640, 360
    out = Image.new("RGB", (3 * cw, 4 * (ch + 16)), "white")
    d = ImageDraw.Draw(out)
    for i, f in enumerate(sub):
        im = Image.open(f[5]).convert("RGB")
        k = min(cw / im.width, ch / im.height)
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        dr = ImageDraw.Draw(im)
        for l in open(f"{ds}/labels/{f[1]}/{f[0]}.txt"):
            q = l.split()
            c, x, y, bw, bh = int(q[0]), *map(float, q[1:])
            dr.rectangle([(x - bw / 2) * im.width, (y - bh / 2) * im.height, (x + bw / 2) * im.width, (y + bh / 2) * im.height], outline=(255, 0, 0), width=2)
        x0, y0 = (i % 3) * cw, (i // 3) * (ch + 16)
        out.paste(im, (x0, y0 + 16))
        d.text((x0 + 2, y0 + 2), f[0], fill=(0, 0, 0))
    out.save(f"{b4}_{sheet}.jpg", quality=85)
