"""Dataset ảnh cắt cho hướng A.

Đọc manifest.csv (không dùng ImageFolder: val chỉ có 35 lớp, test 41 lớp, ImageFolder sẽ đánh số lớp
khác nhau ở mỗi tập). class_id lấy thẳng từ manifest, luôn là 0..51.

Ảnh được đổi về size×size một lần khi khởi tạo và giữ trong RAM dạng uint8 (8329 ảnh 64×64 chỉ ~100 MB),
nên mỗi epoch không phải đọc lại đĩa.
"""
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 52


def load_mean_std(info_path="du_lieu/crops_info.json"):
    """Mean/std tính trên ảnh cắt của train (giáo trình 4.2), do make_crops.py ghi ra."""
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    return info["mean"], info["std"]


class CropDataset(Dataset):
    """
    manifest: đường dẫn manifest.csv (ảnh nằm cùng thư mục với manifest).
    split:    "train", "val" hoặc "test".
    size:     cạnh ảnh đưa vào mạng (mặc định 64).
    augment:  hàm nhận tensor float 3×size×size (giá trị 0..1) và trả về tensor cùng cỡ, hoặc None.
              Chỉ truyền cho train. Không được lật ngang (AGENTS.md mục 2).
    mean, std: chuẩn hoá sau augment. None thì đọc từ du_lieu/crops_info.json.
    """

    def __init__(self, manifest, split, size=64, augment=None, mean=None, std=None):
        root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"Không có ảnh nào của tập {split} trong {manifest}")
        self.split, self.size, self.augment = split, size, augment
        imgs = np.empty((len(self.rows), size, size, 3), dtype=np.uint8)
        for i, r in enumerate(self.rows):
            with Image.open(os.path.join(root, r["path"])) as im:
                imgs[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
        self.images = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # N×3×size×size
        self.labels = torch.tensor([int(r["class_id"]) for r in self.rows])
        if mean is None or std is None:
            mean, std = load_mean_std()
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.images[i].float() / 255
        if self.augment is not None:
            x = self.augment(x)
        return (x - self.mean) / self.std, self.labels[i]

    def classes_present(self):
        """Các lớp có mặt trong tập này (val 35, test 41). Macro-F1 chỉ tính trên các lớp này."""
        return sorted(set(self.labels.tolist()))
