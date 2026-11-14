"""Train một mạng phân loại của hướng A.

Ví dụ (chạy từ gốc repo):
  python -m src.train --model a0 --epochs 1                       # chạy thử, không ghi ket_qua.csv
  python -m src.train --model a0 --seed 0 --nguoi Dũng --thi-nghiem A0

Mặc định = cấu hình nền của A0–A4 (ke_hoach/nguoi_1...): Adam, lr 1e-3, batch 128, 30 epoch,
cross-entropy, không tăng cường dữ liệu, chọn epoch tốt nhất theo val macro-F1.

Mỗi lần chạy ghi vào <out>/<name>/:
  config.json   toàn bộ tham số
  log.csv       mỗi epoch: train_loss, train_acc, val_loss, val_acc, val_macro_f1, lr, giây
  best.pt       trọng số ở epoch có val macro-F1 cao nhất (không commit vào git)
Có --nguoi thì thêm một dòng vào ket_qua/ket_qua.csv.
"""
import argparse
import csv
import json
import os
import time

import torch
from torch import nn

from src.data.augment import AUGMENTS
from src.data.dataset import CropDataset, load_mean_std
from src.evaluate import metrics, predict
from src.models import get_model
from src.utils import append_ket_qua, count_params, pick_device, set_seed


def make_optimizer(name, params, lr, momentum, wd):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    raise SystemExit(f"Không có optimizer '{name}'")


def make_scheduler(name, opt, epochs):
    """Lịch lr theo epoch (giáo trình 4.5). Minh Quân dùng ở A5."""
    if name == "none":
        return None
    if name == "step":  # ×0,1 ở 50% và 75% số epoch
        return torch.optim.lr_scheduler.MultiStepLR(opt, [epochs // 2, epochs * 3 // 4], gamma=0.1)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    raise SystemExit(f"Không có lịch lr '{name}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--opt", default="adam", choices=["adam", "adamw", "sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9, help="chỉ dùng với sgd")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay")
    ap.add_argument("--sched", default="none", choices=["none", "step", "cosine", "linear"])
    ap.add_argument("--augment", default="none", choices=list(AUGMENTS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="data/vn_signs_crops/manifest.csv")
    ap.add_argument("--info", default="du_lieu/crops_info.json")
    ap.add_argument("--out", default="data/runs_a")
    ap.add_argument("--name", default=None, help="mặc định <model>_<opt>_lr<lr>_s<seed>")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps hoặc cpu")
    ap.add_argument("--nguoi", default=None, help="có thì ghi một dòng vào ket_qua/ket_qua.csv")
    ap.add_argument("--thi-nghiem", default="", help="ví dụ A0, A3, A5-opt")
    ap.add_argument("--ghi-chu", default="")
    ap.add_argument("--ket-qua", default="ket_qua/ket_qua.csv")
    args = ap.parse_args()

    name = args.name or f"{args.model}_{args.opt}_lr{args.lr:g}_s{args.seed}"
    run_dir = os.path.join(args.out, name)
    if os.path.exists(os.path.join(run_dir, "best.pt")):
        raise SystemExit(f"Dừng: {run_dir} đã có kết quả. Đặt --name khác hoặc xoá thư mục đó.")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)
    device = pick_device(args.device)
    with open(args.info, encoding="utf-8") as f:
        info = json.load(f)
    mean, std = load_mean_std(args.info)
    train_ds = CropDataset(args.manifest, "train", args.size, AUGMENTS[args.augment], mean, std)
    val_ds = CropDataset(args.manifest, "val", args.size, None, mean, std)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256)

    model = get_model(args.model, size=args.size).to(device)
    n_params = count_params(model)
    opt = make_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd)
    sched = make_scheduler(args.sched, opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    config = dict(vars(args), name=name, device=str(device), so_tham_so=n_params, md5_nhan=info["md5_nhan"],
                  torch=str(torch.__version__))  # str: để best.pt nạp được với weights_only=True
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"{name}: {n_params:,} tham số | thiết bị {device} | train {len(train_ds)}, val {len(val_ds)} "
          f"({len(val_ds.classes_present())} lớp)")

    cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "giay"]
    log = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(cols)
    best = dict(f1=-1.0, epoch=0, acc=0.0)
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        lr_now = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step()
        yv, pv, val_loss = predict(model, val_dl, device)
        m = metrics(yv, pv)
        row = [ep, loss_sum / n, correct / n, val_loss, m["accuracy"], m["macro_f1"], lr_now, time.time() - t0]
        wr.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
        log.flush()
        mark = ""
        if m["macro_f1"] > best["f1"]:
            best = dict(f1=m["macro_f1"], epoch=ep, acc=m["accuracy"])
            torch.save(dict(model=model.state_dict(), config=config, epoch=ep, mean=mean, std=std,
                            val_macro_f1=m["macro_f1"]), os.path.join(run_dir, "best.pt"))
            mark = " *"
        print(f"epoch {ep:3d} | train loss {row[1]:.4f} acc {row[2]:.4f} | val loss {val_loss:.4f} "
              f"acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} | {row[7]:.1f}s{mark}")
    log.close()
    minutes = (time.time() - t_start) / 60
    print(f"Tốt nhất: epoch {best['epoch']}, val macro-F1 {best['f1']:.4f} (trên {m['so_lop']} lớp), "
          f"val acc {best['acc']:.4f}. Thời gian {minutes:.1f} phút. Kết quả ở {run_dir}")

    if args.nguoi:
        append_ket_qua(args.ket_qua, nguoi=args.nguoi, thi_nghiem=args.thi_nghiem, ban=name, seed=args.seed,
                       md5_nhan=info["md5_nhan"], so_tham_so=n_params, epoch=best["epoch"],
                       thoi_gian_phut=round(minutes, 1), val_chinh=round(best["f1"], 4),
                       ghi_chu=f"val acc {best['acc']:.4f}; {args.opt} lr {args.lr:g} {args.sched} "
                               f"aug {args.augment}; {device}. {args.ghi_chu}".strip())
    else:
        print("Không có --nguoi nên không ghi ket_qua/ket_qua.csv (chạy thử).")


if __name__ == "__main__":
    main()
