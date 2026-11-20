"""So sánh các mạng qua 3 seed: trung bình ± độ lệch chuẩn của val macro-F1, vẽ val macro-F1, train loss, val loss.

Dùng (chạy từ gốc repo): python -m src.so_sanh data/runs_a data/runs_a/hinh/a0_a1.png a0 a1
Đọc data/runs_a/<mạng>_adam_lr0.001_s<seed>/log.csv (tên mặc định của train.py với cấu hình nền).
"""
import csv
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs_dir, out_png = sys.argv[1], sys.argv[2]
models = sys.argv[3:]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for mi, m in enumerate(models):
    f1, acc, ep, gap = [], [], [], []
    for s in range(3):
        rows = list(csv.DictReader(open(f"{runs_dir}/{m}_adam_lr0.001_s{s}/log.csv")))
        b = max(rows, key=lambda r: float(r["val_macro_f1"]))
        vl = min(rows, key=lambda r: float(r["val_loss"]))
        f1.append(float(b["val_macro_f1"])); acc.append(float(b["val_acc"])); ep.append(int(b["epoch"]))
        gap.append(float(b["train_acc"]) - float(b["val_acc"]))
        e = [int(r["epoch"]) for r in rows]
        kw = dict(color=colors[mi], alpha=0.8, label=m.upper() if s == 0 else None)
        axes[0].plot(e, [float(r["val_macro_f1"]) for r in rows], **kw)
        axes[1].plot(e, [float(r["train_loss"]) for r in rows], **kw)
        axes[2].plot(e, [float(r["val_loss"]) for r in rows], **kw)
        print(f"{m} seed {s}: best ep {b['epoch']} F1 {float(b['val_macro_f1']):.4f} acc {float(b['val_acc']):.4f} "
              f"train_acc {float(b['train_acc']):.4f} | min val loss ep {vl['epoch']} {float(vl['val_loss']):.3f} "
              f"| ep30 val loss {float(rows[-1]['val_loss']):.3f} | giây/epoch {st.mean(float(r['giay']) for r in rows):.1f}")
        # epoch đầu tiên đạt 0.5 val macro-F1 (tốc độ hội tụ)
        first = next((r["epoch"] for r in rows if float(r["val_macro_f1"]) >= 0.5), "chưa")
        print(f"   epoch đầu tiên val F1 >= 0.5: {first}")
    print(f"{m}: F1 {st.mean(f1):.4f} ± {st.stdev(f1):.4f} | acc {st.mean(acc):.4f} ± {st.stdev(acc):.4f} "
          f"| khoảng cách train-val acc {st.mean(gap):.4f}")
for ax, t in zip(axes, ["val macro-F1 (35 lớp)", "train loss", "val loss"]):
    ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
fig.suptitle("3 seed mỗi mạng, Adam lr 1e-3, batch 128, không tăng cường")
fig.tight_layout()
fig.savefig(out_png, dpi=110)
print("Đã ghi", out_png)
