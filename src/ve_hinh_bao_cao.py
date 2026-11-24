"""Vẽ các hình cho báo cáo (bao_cao/bao_cao.md) từ log đã có. Không train, không chấm lại gì.

Đọc:
  - data/runs_a/<lần chạy>/log.csv          (đường học của mạng phân loại, 3 seed)
  - ket_qua/ket_qua.csv                     (val_chinh, test_chinh từng lần chạy)
  - ket_qua/b0_results.csv, b0_results_s{1,2}.csv  (đường học của B0)
Ghi PNG vào bao_cao/hinh/. Hình chỉ là biểu đồ số liệu, không chứa ảnh của dataset.

Đường cong: đường liền là trung bình 3 seed, dải mờ là khoảng từ seed thấp nhất đến seed cao nhất.
Màu lấy theo thứ tự cố định của bảng màu đã kiểm cho người mù màu (skill dataviz, palette mặc định, 4 màu đầu).

Dùng (từ gốc repo): python -m src.ve_hinh_bao_cao
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "data/runs_a"
OUT = "bao_cao/hinh"
MAU = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CHU, CHU_PHU, LUOI, TRUC = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": TRUC, "axes.labelcolor": CHU_PHU, "xtick.color": CHU_PHU, "ytick.color": CHU_PHU,
    "text.color": CHU, "axes.grid": True, "grid.color": LUOI, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def doc_log(pref):
    """Trả về dict cột → list 3 seed × 30 epoch."""
    out = {}
    for s in range(3):
        for r in csv.DictReader(open(f"{RUNS}/{pref}_s{s}/log.csv")):
            for k, v in r.items():
                out.setdefault(k, [[], [], []])[s].append(float(v))
    return out


def ve_duong(ax, x, ys, mau, nhan):
    tb = [st.mean(c) for c in zip(*ys)]
    ax.fill_between(x, [min(c) for c in zip(*ys)], [max(c) for c in zip(*ys)], color=mau, alpha=0.15, lw=0)
    ax.plot(x, tb, color=mau, label=nhan)


def hinh_duong(ten, cac_ban, o, tieu_de, logy=()):
    """cac_ban: [(tiền tố, nhãn)]; o: [(cột, tiêu đề ô, nhãn trục y)]."""
    fig, axes = plt.subplots(1, len(o), figsize=(4.6 * len(o), 3.6))
    for (pref, nhan), mau in zip(cac_ban, MAU):
        d = doc_log(pref)
        x = [int(e) for e in d["epoch"][0]]
        for ax, (cot, _, _) in zip(axes, o):
            ve_duong(ax, x, d[cot], mau, nhan)
    for i, (ax, (cot, td, nhan_y)) in enumerate(zip(axes, o)):
        ax.set_title(td, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(nhan_y)
        if cot in logy:
            ax.set_yscale("log")
    axes[0].legend(loc="best")
    fig.suptitle(tieu_de, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{ten}.png", dpi=150)
    plt.close(fig)
    print("Đã ghi", f"{OUT}/{ten}.png")


def ket_qua():
    rows = list(csv.DictReader(open("ket_qua/ket_qua.csv", encoding="utf-8")))
    return rows


def hinh_test(rows):
    """Val (35 lớp) và test (41 lớp) macro-F1 của các bậc đã chốt, trung bình ± độ lệch chuẩn 3 seed."""
    bac = [("A0", "A0 LeNet"), ("A1", "A1 VGG"), ("A2", "A2 +BN"), ("A3-a", "A3-a\n+tăng cường"),
           ("A4-res-sau", "A4-res-sâu\n(= A5)"), ("A7-seed", "A7-seed"), ("A7-kientruc", "A7\nkiến trúc")]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for j, (nhan_tap, cot, mau) in enumerate([("Val (35 lớp)", "val_chinh", MAU[0]),
                                              ("Test (41 lớp)", "test_chinh", MAU[1])]):
        xs, tb, sd = [], [], []
        for i, (key, _) in enumerate(bac):
            v = [float(r[cot]) for r in rows if r["thi_nghiem"] == key and r[cot]]
            xs.append(i + (j - 0.5) * 0.3)
            tb.append(st.mean(v))
            sd.append(st.stdev(v) if len(v) > 1 else 0)
        ax.errorbar(xs, tb, yerr=sd, fmt="o", ms=8, color=mau, ecolor=mau, elinewidth=1.5, capsize=4,
                    label=nhan_tap, mec="#fcfcfb", mew=1.5)
        for x, y in zip(xs, tb):
            ax.annotate(f"{y:.3f}".replace(".", ","), (x, y), xytext=(7 if j else -7, 0), va="center",
                        textcoords="offset points", ha="left" if j else "right", fontsize=8, color=CHU_PHU)
    ax.set_xticks(range(len(bac)), [b[1] for b in bac], fontsize=9)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.35, 0.8)
    ax.legend(loc="upper left")
    ax.set_title("Macro-F1 của các bậc đã chốt (trung bình ± độ lệch chuẩn 3 seed; A7-seed chỉ 1 số)", loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/h5_val_test.png", dpi=150)
    plt.close(fig)
    print("Đã ghi", f"{OUT}/h5_val_test.png")


def hinh_tn5():
    """Thí nghiệm 5, số lấy từ NHAT_KY mục 2026-11-23 (macro-F1 trên lớp chung: val 35, test 41)."""
    so = {"G: chia theo nhóm frame": ([0.6044, 0.4667], [0.0141, 0.0408]),
          "R: chia ngẫu nhiên từng ảnh": ([0.9503, 0.9628], [0.0162, 0.0056])}
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    w = 0.36
    for j, ((nhan, (tb, sd)), mau) in enumerate(zip(so.items(), MAU)):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(2)]
        ax.bar(xs, tb, w, yerr=sd, color=mau, label=nhan, capsize=4, ecolor=CHU_PHU)
        for x, y in zip(xs, tb):
            ax.text(x, y + 0.03 + sd[xs.index(x)], f"{y:.3f}".replace(".", ","), ha="center", fontsize=9, color=CHU)
    ax.set_xticks([0, 1], ["Val (35 lớp chung)", "Test (41 lớp chung)"])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title("Thí nghiệm 5: cùng mạng A2, chỉ khác cách chia tập", loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/h6_tn5.png", dpi=150)
    plt.close(fig)
    print("Đã ghi", f"{OUT}/h6_tn5.png")


def hinh_b0():
    files = ["ket_qua/b0_results.csv", "ket_qua/b0_results_s1.csv", "ket_qua/b0_results_s2.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for s, (f, mau) in enumerate(zip(files, MAU)):
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(f))]
        e = [r["epoch"] for r in rows]
        axes[0].plot(e, [r["metrics/mAP50-95(B)"] for r in rows], color=mau, label=f"seed {s}", lw=1.6)
        b = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
        axes[0].plot([b["epoch"]], [b["metrics/mAP50-95(B)"]], "o", ms=8, color=mau, mec="#fcfcfb", mew=1.5)
        axes[1].plot(e, [r["val/cls_loss"] for r in rows], color=mau, lw=1.6, label=f"val, seed {s}")
        axes[1].plot(e, [r["train/cls_loss"] for r in rows], color=mau, lw=1.2, ls="--")
    axes[0].set_title("Val mAP50-95 (35 lớp); chấm tròn = best.pt", loc="left")
    axes[0].set_ylabel("mAP50-95")
    axes[1].set_title("cls loss: liền = val, nét đứt = train", loc="left")
    axes[1].set_ylabel("cls loss")
    axes[1].set_ylim(0, 3)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].legend(loc="lower right")
    fig.suptitle("B0: YOLOv8s train từ đầu, 3 seed", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/h8_b0.png", dpi=150)
    plt.close(fig)
    print("Đã ghi", f"{OUT}/h8_b0.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    o3 = [("train_loss", "Train loss (thang log)", "loss"), ("val_loss", "Val loss", "loss"),
          ("val_macro_f1", "Val macro-F1 (35 lớp)", "macro-F1")]
    hinh_duong("h1_a0_a1_a2", [("a0_adam_lr0.001", "A0 LeNet"), ("a1_adam_lr0.001", "A1 VGG"),
                               ("a2_adam_lr0.001", "A2 VGG + BN")], o3,
               "A0, A1, A2: train loss về gần 0, val loss không giảm theo", logy=("train_loss",))
    hinh_duong("h2_a3", [("a2_adam_lr0.001", "A3-0 (= A2)"), ("a3d_adam_lr0.001", "A3-d dropout"),
                         ("a2_adam_lr0.001_auga3", "A3-a tăng cường"), ("a3d_adam_lr0.001_auga3", "A3-da cả hai")],
               o3, "A3: dropout và tăng cường dữ liệu", logy=("train_loss",))
    hinh_duong("h3_a4", [("a4pn_adam_lr0.001_auga3", "plain-nông (5 conv)"),
                         ("a4ps_adam_lr0.001_auga3", "plain-sâu (9 conv)"),
                         ("a4rs_adam_lr0.001_auga3", "res-sâu (9 conv + tắt)")],
               o3, "A4: mạng sâu hơn có train tệ hơn không?", logy=("train_loss",))
    hinh_duong("h4_a5_lich_lr", [("a4rs_adam_lr0.001_auga3", "lr không đổi"),
                                 ("a4rs_adam_lr0.001_step_auga3", "step"),
                                 ("a4rs_adam_lr0.001_cosine_auga3", "cosine"),
                                 ("a4rs_adam_lr0.001_linear_auga3", "linear")],
               [("lr", "Learning rate (thang log)", "lr"), ("train_loss", "Train loss (thang log)", "loss"),
                ("val_macro_f1", "Val macro-F1 (35 lớp)", "macro-F1")],
               "A5: lịch lr với Adam 1e-3", logy=("lr", "train_loss"))
    rows = ket_qua()
    hinh_test(rows)
    hinh_tn5()
    hinh_b0()


if __name__ == "__main__":
    main()
