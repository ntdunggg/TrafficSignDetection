"""Hàm dùng chung: seed, chọn thiết bị, md5 nhãn, ghi ket_qua.csv."""
import csv
import datetime
import hashlib
import os
import pathlib
import random

import numpy as np
import torch

KET_QUA_COLS = ["ngay", "nguoi", "thi_nghiem", "ban", "seed", "md5_nhan", "so_tham_so", "epoch",
                "thoi_gian_phut", "val_chinh", "test_chinh", "ghi_chu"]


def label_md5(labels_dir):
    """md5 của thư mục nhãn (bỏ qua .cache). Giống hàm trong ke_hoach/00_chung.md."""
    h, root = hashlib.md5(), pathlib.Path(labels_dir)
    for p in sorted(root.rglob("*.txt")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # gồm cả cuda và mps


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def append_ket_qua(path, **row):
    """Thêm một dòng vào ket_qua/ket_qua.csv. Không có file thì dừng (Dũng tạo file, không tự tạo cột khác)."""
    if not os.path.exists(path):
        print(f"Chưa có {path}, không ghi kết quả. Báo Dũng.")
        return
    row.setdefault("ngay", datetime.date.today().isoformat())
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    if header != KET_QUA_COLS:
        raise SystemExit(f"Cột của {path} khác quy định trong ke_hoach/00_chung.md: {header}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=KET_QUA_COLS).writerow({k: row.get(k, "") for k in KET_QUA_COLS})
