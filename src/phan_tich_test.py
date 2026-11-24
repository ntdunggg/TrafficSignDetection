"""Thí nghiệm 3: phân tích lỗi của mạng đã chốt trên test (kế hoạch mục 2.5, NHAT_KY mục C6).

Không dùng để chọn cấu hình: chỉ đọc lại dự đoán của các best.pt đã chấm test ở bước 2.4.
Gộp 3 seed (cộng confusion matrix) để bớt nhiễu, rồi:
  1. Các cặp nhầm nhiều nhất (lớp thật → lớp dự đoán).
  2. Nhầm trong các nhóm dễ nhầm của mục C6: bao nhiêu phần lỗi của nhóm rơi vào lớp khác cùng nhóm.
  3. Độ chính xác theo cỡ box (cạnh dài của box trên ảnh gốc 960×540, cột box_w_px, box_h_px của manifest).
Ghi hình confusion matrix gộp vào data/runs_a/hinh/ (git ignore).

Dùng: python -m src.phan_tich_test --runs data/runs_a/a4rs_adam_lr0.001_auga3_s0 ..._s1 ..._s2
"""
import argparse
import collections
import os

import torch

from src.data.dataset import NUM_CLASSES, CropDataset
from src.evaluate import predict
from src.models import get_model
from src.utils import pick_device

# Nhóm lấy theo mục C6 của NHAT_KY (id lớp đếm từ 0, theo du_lieu/classes.txt)
NHOM = {
    "Cấm rẽ, cấm quay đầu (vòng tròn đỏ, khác mũi tên)": [7, 17, 19, 32, 35, 36, 47, 49],
    "Cấm xe (vòng tròn đỏ, khác hình xe)": [14, 18, 20, 23, 29, 34],
    "Giới hạn tốc độ (khác chữ số)": [38, 39, 40, 41],
    "Biển phụ S.505a": [25, 28, 30],
    "Cặp trái/phải 6/50 (chỗ ngoặt nguy hiểm)": [6, 50],
    "Cặp trái/phải 1/37 (ngã ba)": [1, 37],
    "Cặp trái/phải 15/21 (đường thu hẹp)": [15, 21],
}
CO = [(0, 12), (13, 16), (17, 24), (25, 32), (33, 48), (49, 10_000)]


def ten_lop(path="du_lieu/classes.txt"):
    return [line.strip() for line in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--hinh", default="data/runs_a/hinh/cm_a4rs_test_3seed.png")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    names = ten_lop()
    device = pick_device("auto")

    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    dung_theo_co = collections.defaultdict(lambda: [0, 0])  # (đúng, tổng) cộng qua các seed
    for r in args.runs:
        ck = torch.load(os.path.join(r, "best.pt"), map_location="cpu", weights_only=True)
        cfg = ck["config"]
        model = get_model(cfg["model"], size=cfg["size"])
        model.load_state_dict(ck["model"])
        model.to(device)
        ds = CropDataset(cfg["manifest"], "test", size=cfg["size"], mean=ck["mean"], std=ck["std"])
        y, p, _ = predict(model, torch.utils.data.DataLoader(ds, batch_size=256), device)
        cm.index_put_((y, p), torch.ones_like(y), accumulate=True)
        for row, yi, pi in zip(ds.rows, y.tolist(), p.tolist()):
            canh = max(int(row["box_w_px"]), int(row["box_h_px"]))
            k = next(i for i, (a, b) in enumerate(CO) if a <= canh <= b)
            dung_theo_co[k][0] += int(yi == pi)
            dung_theo_co[k][1] += 1
    n = len(args.runs)
    tong = cm.sum(1)

    print(f"== Cặp nhầm nhiều nhất (cộng {n} seed; 'TB' = trung bình mỗi seed; % = phần của lớp thật) ==")
    off = cm.clone()
    off.fill_diagonal_(0)
    idx = torch.argsort(off.flatten(), descending=True)[: args.top]
    for i in idx.tolist():
        t, q = divmod(i, NUM_CLASSES)
        print(f"{t:2d} {names[t][:45]:45s} → {q:2d} {names[q][:45]:45s} TB {off[t, q].item() / n:5.1f} "
              f"({off[t, q].item() / tong[t].item():.0%} của {tong[t].item() // n} ảnh)")

    print("\n== Nhóm dễ nhầm (mục C6) ==")
    for ten, lops in NHOM.items():
        co = [c for c in lops if tong[c] > 0]
        if not co:
            print(f"{ten}: không có lớp nào ở test")
            continue
        sai = sum((tong[c] - cm[c, c]).item() for c in co)
        trong = sum(cm[c, d].item() for c in co for d in lops if d != c)
        acc = sum(cm[c, c].item() for c in co) / sum(tong[c].item() for c in co)
        chi_tiet = ", ".join(f"{c}: {tong[c].item() // n} ảnh, recall {cm[c, c].item() / tong[c].item():.2f}" for c in co)
        print(f"{ten}\n   lớp có ở test: {chi_tiet}\n   acc cả nhóm {acc:.3f}; lỗi rơi vào lớp khác cùng nhóm: "
              f"{trong}/{sai} ({trong / max(sai, 1):.0%})")
        if len(lops) == 2 and len(co) == 2:
            a, b = lops
            print(f"   {a}→{b}: TB {cm[a, b].item() / n:.1f}; {b}→{a}: TB {cm[b, a].item() / n:.1f}")

    print("\n== Độ chính xác theo cạnh dài của box trên ảnh gốc (px) ==")
    for k, (a, b) in enumerate(CO):
        d, t = dung_theo_co[k]
        nhan = f"{a}–{b}" if b < 10_000 else f"> {a - 1}"
        print(f"{nhan:>8s} px: {t // n:4d} ảnh cắt, acc {d / t:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        present = (tong > 0).nonzero().flatten().tolist()
        # Hàng: 41 lớp có ở test. Cột: đủ 52 lớp, vì mạng có thể đoán ra lớp chỉ có ở train (ví dụ 13 → 48).
        sub = cm[present].float()
        sub = sub / sub.sum(1, keepdim=True)
        fig, ax = plt.subplots(figsize=(14, 11))
        im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), range(NUM_CLASSES), fontsize=6)
        for c in range(NUM_CLASSES):
            if c not in present:
                ax.get_xticklabels()[c].set_color("#e34948")
        ax.set_yticks(range(len(present)), [f"{c} {names[c][:28]}" for c in present], fontsize=7)
        ax.set_xlabel("Lớp dự đoán")
        ax.set_ylabel("Lớp thật")
        ax.set_title(f"Test: hàng là {len(present)} lớp có ở test, cột là đủ {NUM_CLASSES} lớp (số đỏ: lớp không có ở test); "
                     f"cộng {n} seed, chuẩn hoá theo hàng", fontsize=10)
        fig.colorbar(im)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.hinh), exist_ok=True)
        fig.savefig(args.hinh, dpi=120)
        print("\nĐã ghi", args.hinh)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
