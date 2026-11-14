"""Đánh giá mạng phân loại: accuracy, macro-F1 trên các lớp có mặt, recall từng lớp, confusion matrix.

Dùng (chạy từ gốc repo):
  python -m src.evaluate --run data/runs_a/<tên lần chạy> --split val
  python -m src.evaluate --run data/runs_a/<tên lần chạy> --split test --cham-test

Test chỉ chấm một lần cho cấu hình đã chốt, nên phải thêm --cham-test (AGENTS.md mục 2).
Kết quả ghi vào <run>/danh_gia_<split>/.
"""
import argparse
import csv
import json
import os

import torch

from src.data.dataset import NUM_CLASSES, CropDataset
from src.models import get_model
from src.utils import pick_device


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        ys.append(y.cpu())
        ps.append(logits.argmax(1).cpu())
    y, p = torch.cat(ys), torch.cat(ps)
    return y, p, loss_sum / len(y)


def metrics(y, p, num_classes=NUM_CLASSES):
    """Accuracy và macro-F1. Macro-F1 chỉ lấy trung bình trên các lớp có trong y (val 35, test 41 lớp).

    F1 của lớp c = 2·TP / (2·TP + FP + FN).
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)  # hàng: nhãn thật, cột: dự đoán
    cm.index_put_((y, p), torch.ones_like(y), accumulate=True)
    tp = cm.diag().float()
    fn = cm.sum(1).float() - tp
    fp = cm.sum(0).float() - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    present = (cm.sum(1) > 0).nonzero().flatten().tolist()
    return dict(
        accuracy=(y == p).float().mean().item(),
        macro_f1=f1[present].mean().item(),
        so_lop=len(present),
        present=present,
        f1=f1.tolist(),
        recall=recall.tolist(),
        cm=cm,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="thư mục lần chạy, có best.pt")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--cham-test", action="store_true", help="xác nhận đã chốt cấu hình, cho phép chấm test")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.split == "test" and not args.cham_test:
        raise SystemExit("Test chỉ chấm một lần cho cấu hình đã chốt. Nếu đúng vậy, thêm --cham-test.")

    ck = torch.load(os.path.join(args.run, "best.pt"), map_location="cpu", weights_only=True)
    cfg = ck["config"]
    device = pick_device(args.device)
    model = get_model(cfg["model"], size=cfg["size"])
    model.load_state_dict(ck["model"])
    model.to(device)
    ds = CropDataset(cfg["manifest"], args.split, size=cfg["size"], mean=ck["mean"], std=ck["std"])
    loader = torch.utils.data.DataLoader(ds, batch_size=256)
    y, p, loss = predict(model, loader, device)
    m = metrics(y, p)

    out = os.path.join(args.run, f"danh_gia_{args.split}")
    os.makedirs(out, exist_ok=True)
    counts = m["cm"].sum(1).tolist()
    with open(os.path.join(out, "tung_lop.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "so_anh", "recall", "f1"])
        for c in m["present"]:
            w.writerow([c, counts[c], round(m["recall"][c], 4), round(m["f1"][c], 4)])
    with open(os.path.join(out, "confusion_matrix.csv"), "w", newline="") as f:
        csv.writer(f).writerows(m["cm"].tolist())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 10))
        cmn = m["cm"].float() / m["cm"].sum(1, keepdim=True).clamp(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("Lớp dự đoán")
        ax.set_ylabel("Lớp thật")
        ax.set_title(f"{os.path.basename(args.run)}, {args.split}: chuẩn hoá theo hàng (recall)")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.tick_params(labelsize=6)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=120)
    except ImportError:
        pass
    tom_tat = dict(split=args.split, loss=round(loss, 4), accuracy=round(m["accuracy"], 4),
                   macro_f1=round(m["macro_f1"], 4), so_lop=m["so_lop"], epoch=ck["epoch"])
    with open(os.path.join(out, "tom_tat.json"), "w") as f:
        json.dump(tom_tat, f, indent=2)
    print(json.dumps(tom_tat, ensure_ascii=False))
    print("Đã ghi:", out)


if __name__ == "__main__":
    main()
