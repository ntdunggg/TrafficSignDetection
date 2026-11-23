"""Phép thử ngoài phân phối rẻ cho H3 (kế hoạch mục 2.5): đưa vùng nền không có biển qua mạng phân loại.

Cách làm:
- Cắt ngẫu nhiên N vùng vuông (mặc định 500, seed 0) từ ảnh test của data/vn_signs_yolo.
  Cạnh vùng lấy ngẫu nhiên từ cạnh các ảnh cắt test thật (round(1,2 × cạnh dài box), như crop_box),
  để vùng nền cùng cỡ với vùng biển thật. Vùng phải nằm trọn trong ảnh và không chạm box nào.
- Đổi về 64×64, chuẩn hoá bằng mean/std của mạng, giống hệt ảnh cắt thật (CropDataset).
- Mạng phân loại luôn chọn 1 trong 52 lớp. Đo xác suất softmax lớn nhất (MSP) trên vùng nền,
  so với MSP trên ảnh cắt biển thật của test. Đếm tỉ lệ vùng nền có MSP > 0,5 và > 0,9.
- AUROC của MSP khi dùng nó để tách biển thật với nền (Hendrycks & Gimpel 2017, arXiv 1610.02136):
  0,5 là không tách được, 1 là tách hoàn hảo.
Không ghi ảnh vào repo. --xem ghi một ảnh ghép 60 vùng nền đầu tiên để kiểm bằng mắt (đặt đường dẫn ngoài git).
Lưu ý: dataset có thể sót nhãn (NHAT_KY mục B4), nên một số vùng "nền" có thể chứa biển chưa gán nhãn.

Dùng: python -m src.ood_nen --runs data/runs_a/<run1> ... [--n 500] [--xem <file.jpg>]
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from PIL import Image

from src.data.dataset import CropDataset
from src.models import get_model
from src.utils import pick_device


def boxes_px(label_path, W, H):
    out = []
    for line in open(label_path):
        if line.strip():
            _, xc, yc, w, h = map(float, line.split())
            out.append(((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
    return out


def lay_nen(yolo, manifest, n, seed):
    sides = [round(1.2 * max(int(r["box_w_px"]), int(r["box_h_px"])))
             for r in csv.DictReader(open(manifest)) if r["split"] == "test"]
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(yolo, "images", "test")) if f.endswith(".jpg"))
    rng = random.Random(seed)
    crops, info, thu = [], [], 0
    while len(crops) < n:
        thu += 1
        stem = rng.choice(stems)
        side = max(8, rng.choice(sides))
        with Image.open(os.path.join(yolo, "images", "test", stem + ".jpg")) as im:
            img = im.convert("RGB")
        W, H = img.size
        if side >= min(W, H):
            continue
        x0, y0 = rng.randint(0, W - side), rng.randint(0, H - side)
        x1, y1 = x0 + side, y0 + side
        bx = boxes_px(os.path.join(yolo, "labels", "test", stem + ".txt"), W, H)
        if any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in bx):
            continue
        crops.append(img.crop((x0, y0, x1, y1)))
        info.append((stem, x0, y0, side))
    return crops, info, thu


def auroc(pos, neg):
    """Xác suất một mẫu dương (biển thật) có điểm cao hơn một mẫu âm (nền); hoà tính 0,5."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--yolo", default="data/vn_signs_yolo")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xem", default=None)
    args = ap.parse_args()
    names = [line.strip() for line in open("du_lieu/classes.txt", encoding="utf-8")]
    device = pick_device("auto")

    ck0 = torch.load(os.path.join(args.runs[0], "best.pt"), map_location="cpu", weights_only=True)
    manifest = ck0["config"]["manifest"]
    crops, info, thu = lay_nen(args.yolo, manifest, args.n, args.seed)
    print(f"Lấy {len(crops)} vùng nền sau {thu} lần thử (bỏ vùng chạm box hoặc quá lớn); "
          f"cạnh trung vị {int(np.median([i[3] for i in info]))} px")
    if args.xem:
        g = Image.new("RGB", (64 * 10, 64 * 6), "white")
        for i, c in enumerate(crops[:60]):
            g.paste(c.resize((64, 64)), (64 * (i % 10), 64 * (i // 10)))
        g.save(args.xem)
        print("Đã ghi ảnh ghép để xem:", args.xem)

    size = ck0["config"]["size"]
    raw = torch.from_numpy(np.stack([np.asarray(c.resize((size, size), Image.BILINEAR)) for c in crops]))
    raw = raw.permute(0, 3, 1, 2).float() / 255

    for r in args.runs:
        ck = torch.load(os.path.join(r, "best.pt"), map_location="cpu", weights_only=True)
        model = get_model(ck["config"]["model"], size=size)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        mean, std = torch.tensor(ck["mean"]).view(1, 3, 1, 1), torch.tensor(ck["std"]).view(1, 3, 1, 1)
        with torch.no_grad():
            p_nen = torch.softmax(model(((raw - mean) / std).to(device)), 1).cpu()
            ds = CropDataset(manifest, "test", size=size, mean=ck["mean"], std=ck["std"])
            p_that = torch.cat([torch.softmax(model(x.to(device)), 1).cpu()
                                for x, _ in torch.utils.data.DataLoader(ds, batch_size=256)])
        msp_nen, lop_nen = p_nen.max(1)
        msp_that = p_that.max(1).values
        top = torch.bincount(lop_nen, minlength=len(names)).topk(3)
        print(f"\n{os.path.basename(r)}:")
        print(f"  Nền ({len(crops)} vùng): MSP trung vị {msp_nen.median():.3f}; "
              f"> 0,5: {(msp_nen > 0.5).float().mean():.1%}; > 0,9: {(msp_nen > 0.9).float().mean():.1%}")
        print(f"  Biển thật ({len(ds)} ảnh cắt test): MSP trung vị {msp_that.median():.3f}; "
              f"> 0,5: {(msp_that > 0.5).float().mean():.1%}; > 0,9: {(msp_that > 0.9).float().mean():.1%}")
        print(f"  AUROC của MSP (biển thật so với nền): {auroc(msp_that.numpy(), msp_nen.numpy()):.3f}")
        print("  Nền hay bị gán vào: " + "; ".join(
            f"{names[c][:40]} ({v} vùng)" for v, c in zip(top.values.tolist(), top.indices.tolist())))


if __name__ == "__main__":
    main()
