"""Tạo catalog biển báo (B4, C2, C3, C4, C5): một file HTML có thể xem và sửa, chỉ đọc dataset.
Dùng: python3 -I build_catalog.py <archive_dir> <dataset_dir> <out.html> <template.html>
"""
import base64, collections, io, json, os, sys
from PIL import Image, ImageDraw

archive, ds, out, tpl = sys.argv[1:5]
read = lambda f: open(f, encoding="utf-8").read().splitlines()
codes, en, vi = read(f"{archive}/classes.txt"), read(f"{archive}/classes_en.txt"), read(f"{archive}/classes_vie.txt")

boxes = collections.defaultdict(list)
counts = collections.defaultdict(collections.Counter)
for sp in ("train", "val", "test"):
    for f in sorted(os.listdir(f"{ds}/labels/{sp}")):
        s = f[:-4]
        with Image.open(f"{ds}/images/{sp}/{s}.jpg") as im:
            W, H = im.size
        for ln, l in enumerate(open(f"{ds}/labels/{sp}/{f}")):
            p = l.split()
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            boxes[c].append(dict(s=s, sp=sp, ln=ln, x=x, y=y, w=w, h=h, W=W, H=H))
            counts[c][sp] += 1

SECTIONS = [
    dict(key="c3", title="C3. W.246a và W.246c", ids=[26, 51], mode="near"),
    dict(key="c4", title="C4. Mã lạ và dấu *", ids=[31, 34, 14, 16, 17, 43], mode="near"),
    dict(key="c5", title="C5. Biển phụ S.505a", ids=[25, 28, 30], mode="above"),
]
tasks = []  # (stem, split, box, mode, nơi lưu)
samples = {c: [] for c in range(52)}
review = {c: [] for sec in SECTIONS for c in sec["ids"]}
for c in range(52):
    bs = sorted(boxes[c], key=lambda b: b["s"])
    area = sorted(b["w"] * b["W"] * b["h"] * b["H"] for b in bs)
    big = [b for b in bs if b["w"] * b["W"] * b["h"] * b["H"] >= area[len(area) // 2]] or bs
    chosen, seen = [], set()
    for i in range(6):
        b = big[i * len(big) // 6] if len(big) >= 6 else (big[i] if i < len(big) else None)
        if b and b["s"] not in seen:
            seen.add(b["s"])
            chosen.append(b)
    for b in chosen:
        tasks.append((b, "near", samples[c]))
for sec in SECTIONS:
    for c in sec["ids"]:
        for b in sorted(boxes[c], key=lambda b: (b["s"], b["ln"])):
            tasks.append((b, sec["mode"], review[c]))


def crop(im, b, mode):
    W, H = im.size
    bw, bh = b["w"] * W, b["h"] * H
    x0, y0, x1, y1 = (b["x"] * W - bw / 2, b["y"] * H - bh / 2, b["x"] * W + bw / 2, b["y"] * H + bh / 2)
    if mode == "above":  # thấy cả biển chính phía trên biển phụ
        L, T, R, B = bw * 1.0, bh * 3.0, bw * 1.0, bh * 0.8
    else:
        p = max(0.6 * max(bw, bh), 10)
        L, T, R, B = p, p, p, p
    a, t, c, d = max(0, x0 - L), max(0, y0 - T), min(W, x1 + R), min(H, y1 + B)
    k = 200 / max(c - a, d - t)
    img = im.crop((int(a), int(t), int(c), int(d))).resize((max(1, round((c - a) * k)), max(1, round((d - t) * k))), Image.LANCZOS)
    ImageDraw.Draw(img).rectangle([(x0 - a) * k, (y0 - t) * k, (x1 - a) * k, (y1 - t) * k], outline=(255, 30, 30), width=2)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), round(bw), round(bh)


tasks.sort(key=lambda t: (t[0]["sp"], t[0]["s"]))
cur, im = None, None
for b, mode, dest in tasks:
    key = (b["sp"], b["s"])
    if key != cur:
        cur, im = key, Image.open(f"{ds}/images/{b['sp']}/{b['s']}.jpg").convert("RGB")
    uri, pw, ph = crop(im, b, mode)
    dest.append(dict(k=f"{b['s']}#{b['ln']}", s=b["s"], sp=b["sp"], img=uri, pw=pw, ph=ph, small=pw * ph < 1024))
for v in list(samples.values()) + list(review.values()):
    v.sort(key=lambda e: e["k"])

classes = []
for c in range(52):
    bs = boxes[c]
    classes.append(dict(id=c, code=codes[c], en=en[c], vi=vi[c], total=len(bs), train=counts[c]["train"], val=counts[c]["val"], test=counts[c]["test"],
                        small=sum(b["w"] * b["W"] * b["h"] * b["H"] < 1024 for b in bs), samples=samples[c]))
data = dict(classes=classes, sections=[dict(key=s["key"], title=s["title"], ids=s["ids"]) for s in SECTIONS], review={str(k): v for k, v in review.items()})
html = open(tpl, encoding="utf-8").read().replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w", encoding="utf-8").write(html)
print("catalog:", out, f"{os.path.getsize(out)/1e6:.1f} MB", "| crop:", len(tasks), "| box cần xem:", sum(len(v) for v in review.values()))
