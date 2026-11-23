"""A7: ensemble, lấy trung bình softmax của vài mạng đã train (giáo trình 4.9).

Dùng (chạy từ gốc repo):
  python -m src.ensemble --name A7-seed --runs data/runs_a/<run1> data/runs_a/<run2> data/runs_a/<run3>
  thêm --split test --cham-test khi chấm test (chỉ một lần, cho cấu hình đã chốt)
  thêm --nguoi Dũng --thi-nghiem A7-... để ghi một dòng vào ket_qua/ket_qua.csv

Cách làm:
- Nạp best.pt của từng lần chạy (epoch tốt nhất theo val macro-F1), cho chạy trên cùng tập ảnh.
- Mỗi mạng cho xác suất softmax 52 lớp. Ensemble lấy trung bình cộng các xác suất đó, rồi lấy lớp lớn nhất.
- Các mạng phải dùng cùng manifest, cùng kích thước ảnh, cùng mean/std, để cùng một ảnh vào.
In ra: macro-F1 và accuracy của từng mạng và của ensemble, tổng số tham số, thời gian suy diễn.
Thời gian: mỗi mạng chạy 1 lượt làm nóng rồi đo 3 lượt trên cả tập (batch 256), lấy trung vị, quy ra ms mỗi ảnh.
Ensemble chạy tuần tự từng mạng nên thời gian bằng tổng thời gian các mạng.
Kết quả ghi vào data/runs_a/ensemble/<name>_<split>/tom_tat.json.
"""
import argparse
import json
import os
import statistics as st
import time

import torch

from src.data.dataset import CropDataset
from src.evaluate import metrics
from src.models import get_model
from src.utils import append_ket_qua, count_params, pick_device


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def probs_of(model, loader, device):
    model.eval()
    out = [torch.softmax(model(x.to(device)), 1).cpu() for x, _ in loader]
    return torch.cat(out)


def time_ms_per_image(model, loader, device, n_images, repeats=3):
    probs_of(model, loader, device)  # làm nóng
    ts = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        probs_of(model, loader, device)
        sync(device)
        ts.append(time.perf_counter() - t0)
    return st.median(ts) * 1000 / n_images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cham-test", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="data/runs_a/ensemble")
    ap.add_argument("--nguoi", default=None)
    ap.add_argument("--thi-nghiem", default="A7")
    ap.add_argument("--seed", default="")
    ap.add_argument("--ket-qua", default="ket_qua/ket_qua.csv")
    args = ap.parse_args()
    if args.split == "test" and not args.cham_test:
        raise SystemExit("Test chỉ chấm một lần cho cấu hình đã chốt. Nếu đúng vậy, thêm --cham-test.")

    device = pick_device(args.device)
    cks = [torch.load(os.path.join(r, "best.pt"), map_location="cpu", weights_only=True) for r in args.runs]
    c0 = cks[0]
    for r, ck in zip(args.runs, cks):
        same = (ck["config"]["manifest"], ck["config"]["size"], ck["mean"], ck["std"]) == \
               (c0["config"]["manifest"], c0["config"]["size"], c0["mean"], c0["std"])
        if not same:
            raise SystemExit(f"{r} khác manifest, kích thước ảnh hoặc mean/std với {args.runs[0]}")

    ds = CropDataset(c0["config"]["manifest"], args.split, size=c0["config"]["size"], mean=c0["mean"], std=c0["std"])
    loader = torch.utils.data.DataLoader(ds, batch_size=256)
    y = ds.labels

    members, all_probs, total_params, total_ms = [], [], 0, 0.0
    for r, ck in zip(args.runs, cks):
        model = get_model(ck["config"]["model"], size=ck["config"]["size"])
        model.load_state_dict(ck["model"])
        model.to(device)
        pr = probs_of(model, loader, device)
        m = metrics(y, pr.argmax(1))
        n_par = count_params(model)
        ms = time_ms_per_image(model, loader, device, len(ds))
        members.append(dict(run=os.path.basename(r), model=ck["config"]["model"], epoch=ck["epoch"],
                            so_tham_so=n_par, macro_f1=round(m["macro_f1"], 4),
                            accuracy=round(m["accuracy"], 4), ms_moi_anh=round(ms, 4)))
        all_probs.append(pr)
        total_params += n_par
        total_ms += ms
        print(f"{members[-1]['run']}: F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} | {n_par:,} tham số | "
              f"{ms:.4f} ms/ảnh")

    ens_p = torch.stack(all_probs).mean(0)
    me = metrics(y, ens_p.argmax(1))
    f1s = [mb["macro_f1"] for mb in members]
    tom_tat = dict(
        name=args.name, split=args.split, so_lop=me["so_lop"], so_mang=len(members),
        ensemble=dict(macro_f1=round(me["macro_f1"], 4), accuracy=round(me["accuracy"], 4),
                      so_tham_so=total_params, ms_moi_anh=round(total_ms, 4)),
        mang_don_tot_nhat=max(members, key=lambda mb: mb["macro_f1"]),
        mang_don_trung_binh_f1=round(st.mean(f1s), 4),
        thanh_vien=members, thiet_bi=str(device),
    )
    out = os.path.join(args.out, f"{args.name}_{args.split}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tom_tat.json"), "w", encoding="utf-8") as f:
        json.dump(tom_tat, f, ensure_ascii=False, indent=2)
    print(f"ENSEMBLE {args.name}: F1 {me['macro_f1']:.4f} (trên {me['so_lop']} lớp) acc {me['accuracy']:.4f} | "
          f"{total_params:,} tham số | {total_ms:.4f} ms/ảnh | mạng đơn: TB F1 {st.mean(f1s):.4f}, "
          f"tốt nhất {max(f1s):.4f}")
    print("Đã ghi:", out)

    if args.nguoi:
        key = "val_chinh" if args.split == "val" else "test_chinh"
        append_ket_qua(args.ket_qua, nguoi=args.nguoi, thi_nghiem=args.thi_nghiem, ban=args.name, seed=args.seed,
                       md5_nhan=c0["config"]["md5_nhan"], so_tham_so=total_params, **{key: round(me["macro_f1"], 4)},
                       ghi_chu=f"ensemble trung bình softmax {len(members)} mạng: "
                               f"{' + '.join(mb['run'] for mb in members)}; val acc {me['accuracy']:.4f}; "
                               f"{total_ms:.3f} ms/ảnh trên {device}")


if __name__ == "__main__":
    main()
