"""Chia train/val/test theo NHÓM frame (A1, A2). Chỉ đọc archive.

1. Quét ảnh: kích thước + thumbnail xám 64x36.
2. Tạo nhóm: cắt khi tương quan giữa 2 ảnh liền số < cut, hoặc khi đổi kích thước ảnh.
3. Gộp nhóm: 2 nhóm có cặp ảnh tương quan > merge thì thành 1 nhóm (cảnh quay lại, ảnh gần trùng).
4. Gán nguyên nhóm vào tập bằng simulated annealing, ưu tiên: mọi lớp có ở train,
   lớp đủ nhóm thì có mặt ở val và test, tỉ lệ ảnh và box gần 70/15/15.

So sánh:  python3 -I make_split.py <archive> <meta.npz> compare
Xuất ra:  python3 -I make_split.py <archive> <meta.npz> write <out_dir> <cut> <merge>
"""
import math, os, random, sys
import numpy as np
from PIL import Image

archive, meta, mode = sys.argv[1:4]
NC, SEED, RATIOS, TAG = 52, 0, np.array([0.70, 0.15, 0.15]), ["train", "val", "test"]

if not os.path.exists(meta):
    names = sorted(f[:-4] for f in os.listdir(f"{archive}/images") if f.endswith(".jpg"))
    sizes = np.zeros((len(names), 2), dtype=np.int32)
    thumbs = np.zeros((len(names), 36, 64), dtype=np.uint8)
    for i, n in enumerate(names):
        with Image.open(f"{archive}/images/{n}.jpg") as im:
            sizes[i] = im.size
            im.draft("L", (128, 72))
            thumbs[i] = np.asarray(im.convert("L").resize((64, 36), Image.BILINEAR))
    np.savez_compressed(meta, names=np.array(names), sizes=sizes, thumbs=thumbs)

d = np.load(meta)
names = [str(x) for x in d["names"]]
sizes = d["sizes"]
n = len(names)
v = d["thumbs"].astype(np.float32).reshape(n, -1)
v -= v.mean(1, keepdims=True)
v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
corr = v @ v.T
np.fill_diagonal(corr, -1)
adj = np.array([corr[i, i + 1] for i in range(n - 1)])

labels = {s: [int(l.split()[0]) for l in open(f"{archive}/labels/{s}.txt") if l.strip()] for s in names}
keep = [i for i, s in enumerate(names) if labels[s]]
empty = [s for s in names if not labels[s]]


def make_groups(cut, merge):
    g = np.zeros(n, dtype=int)
    for i in range(1, n):
        g[i] = g[i - 1] + int(adj[i - 1] < cut or (sizes[i] != sizes[i - 1]).any())
    parent = list(range(g.max() + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in np.argwhere(np.triu(corr > merge, 1)):
        if g[a] != g[b]:
            parent[find(g[a])] = find(g[b])
    roots = {r: k for k, r in enumerate(sorted({find(x) for x in range(g.max() + 1)}))}
    return np.array([roots[find(x)] for x in g])


def assign(grp, seed=SEED):
    gids = sorted({grp[i] for i in keep})
    gmap = {g: k for k, g in enumerate(gids)}
    GC, GI = np.zeros((len(gids), NC)), np.zeros(len(gids))
    for i in keep:
        k = gmap[grp[i]]
        GI[k] += 1
        for c in labels[names[i]]:
            GC[k, c] += 1
    Cn = GC.sum(0)
    feasible = (GC > 0).sum(0) >= 2  # lớp chỉ có 1 nhóm thì buộc nằm ở train
    tc, ti = RATIOS[:, None] * Cn[None, :], RATIOS * GI.sum()

    def cost(S, I):
        c = ((S - tc) ** 2 / (tc + 1)).sum() + 3.0 * ((I - ti) ** 2 / ti).sum()
        c += 1e4 * (S[0] == 0).sum()
        # Phạt riêng từng tập để chuyển 1 nhóm cũng giảm được phạt; ưu tiên test hơn val
        c += 800.0 * ((S[1] == 0) & feasible).sum() + 1200.0 * ((S[2] == 0) & feasible).sum()
        return c

    rng, best = random.Random(seed), None
    for _ in range(20):
        asg = np.array([rng.choices([0, 1, 2], RATIOS)[0] for _ in gids])
        S = np.stack([GC[asg == s].sum(0) for s in range(3)])
        I = np.array([GI[asg == s].sum() for s in range(3)])
        cur, T = cost(S, I), 20.0
        for _ in range(8000):
            k = rng.randrange(len(gids))
            old = asg[k]
            new = rng.choice([x for x in range(3) if x != old])
            S[old] -= GC[k]; S[new] += GC[k]; I[old] -= GI[k]; I[new] += GI[k]
            c2 = cost(S, I)
            if c2 < cur or rng.random() < math.exp((cur - c2) / T):
                asg[k] = new
                cur = c2
            else:
                S[old] += GC[k]; S[new] -= GC[k]; I[old] += GI[k]; I[new] -= GI[k]
            T = max(0.01, T * 0.999)
        if best is None or cur < best[0]:
            best = (cur, asg.copy())
    return {i: int(best[1][gmap[grp[i]]]) for i in keep}, (GC > 0).sum(0)


def report(grp, split_of, gcount):
    out = []
    sz = np.bincount(grp)
    out.append(f"nhóm: {len(sz)}, cỡ median={int(np.median(sz))}, lớn nhất={sz.max()}")
    tri = [i for i in keep if split_of[i] == 0]
    cnt = np.zeros((3, NC), dtype=int)
    for i in keep:
        for c in labels[names[i]]:
            cnt[split_of[i], c] += 1
    for s in range(3):
        ii = [i for i in keep if split_of[i] == s]
        line = f"{TAG[s]:5s}: {len(ii)} ảnh ({len(ii)/len(keep)*100:.1f}%), {cnt[s].sum()} box ({cnt[s].sum()/cnt.sum()*100:.1f}%)"
        if s:
            m = corr[np.ix_(ii, tri)].max(1)
            line += f" | giống train: median={np.median(m):.3f} p90={np.percentile(m,90):.3f} >0.95={(m>0.95).mean()*100:.1f}%"
        out.append(line)
    miss_val, miss_test = np.where(cnt[1] == 0)[0].tolist(), np.where(cnt[2] == 0)[0].tolist()
    out.append(f"lớp thiếu ở val ({len(miss_val)}): {miss_val}")
    out.append(f"lớp thiếu ở test ({len(miss_test)}): {miss_test}")
    out.append(f"lớp thiếu ở val hoặc test: {len(set(miss_val)|set(miss_test))}")
    out.append(f"lớp chỉ có trong 1-2 nhóm (không thể phủ đủ 3 tập): {np.where(gcount<3)[0].tolist()}")
    return out, cnt


idx = {s: i for i, s in enumerate(names)}
tr0 = [idx[x[:-4]] for x in open(f"{archive}/split_dataset/train_files.txt").read().split()]
te0 = [idx[x[:-4]] for x in open(f"{archive}/split_dataset/test_files.txt").read().split()]
m0 = corr[np.ix_(te0, tr0)].max(1)
head = f"[Split GỐC] test giống train: median={np.median(m0):.3f} p90={np.percentile(m0,90):.3f} >0.95={(m0>0.95).mean()*100:.1f}%"
c0 = np.zeros((2, NC), dtype=int)
for i in tr0:
    for c in labels[names[i]]: c0[0, c] += 1
for i in te0:
    for c in labels[names[i]]: c0[1, c] += 1
head += f" | lớp thiếu ở test: {np.where(c0[1]==0)[0].tolist()}"
print(head)

if mode == "seeds":  # đo độ ổn định theo seed, không ghi split
    cut, merge = float(sys.argv[4]), float(sys.argv[5])
    grp = make_groups(cut, merge)
    base = None
    for seed in range(5):
        sp, gc = assign(grp, seed)
        lines, cnt = report(grp, sp, gc)
        same = sum(sp[i] == base[i] for i in keep) / len(keep) if base else 1.0
        base = base or sp
        none = [c for c in range(NC) if cnt[1, c] == 0 and cnt[2, c] == 0]
        print(f"\n=== seed={seed} | ảnh cùng tập với seed 0: {same*100:.1f}% | lớp không có ở val và test ({len(none)}): {none}")
        print("\n".join(lines[1:]))
elif mode == "compare":
    for cut in (0.3, 0.4, 0.5):
        for merge in (0.95, 0.97):
            grp = make_groups(cut, merge)
            sp, gc = assign(grp)
            print(f"\n=== cut={cut} merge={merge}")
            print("\n".join(report(grp, sp, gc)[0]))
else:
    out, cut, merge = sys.argv[4], float(sys.argv[5]), float(sys.argv[6])
    grp = make_groups(cut, merge)
    sp, gc = assign(grp)
    lines, cnt = report(grp, sp, gc)
    os.makedirs(out, exist_ok=True)
    for s in range(3):
        with open(f"{out}/{TAG[s]}.txt", "w") as f:
            f.write("".join(names[i] + "\n" for i in keep if sp[i] == s))
    open(f"{out}/excluded_empty.txt", "w").write("\n".join(empty) + "\n")
    with open(f"{out}/groups.csv", "w") as f:
        f.write("stem,group,split\n")
        for i in range(n):
            f.write(f"{names[i]},{grp[i]},{TAG[sp[i]] if i in sp else 'loai'}\n")
    with open(f"{out}/class_counts.csv", "w") as f:
        f.write("id,train,val,test,so_nhom\n")
        for c in range(NC):
            f.write(f"{c},{cnt[0,c]},{cnt[1,c]},{cnt[2,c]},{gc[c]}\n")
    txt = "\n".join([head, f"cut={cut} merge={merge} seed={SEED}"] + lines) + "\n"
    open(f"{out}/split_report.txt", "w").write(txt)
    print(txt)
