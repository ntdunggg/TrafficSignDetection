"""A2: A1 thêm BatchNorm sau mỗi conv, trước ReLU (Ioffe & Szegedy 2015, giáo trình 4.4).

Chỉ đổi một yếu tố so với A1: mỗi khối thành [conv 3×3 → BN → ReLU → conv 3×3 → BN → ReLU → max pool 2].
Phần FC giữ nguyên.

- Conv đặt bias=False: BN trừ đi trung bình theo từng kênh, nên bias của conv bị triệt tiêu,
  vai trò dịch chuyển do β của BN đảm nhận.
- Mỗi BatchNorm2d(C) có 2C tham số học được (γ, β) và 2C thống kê không học (running mean, running var),
  cộng một bộ đếm num_batches_tracked.
- Lúc train BN chuẩn hóa bằng trung bình và phương sai của batch hiện tại; lúc suy diễn (model.eval())
  dùng running mean/var tích lũy khi train. evaluate.predict gọi model.eval() nên đúng chế độ.
Tính tay ở ket_qua/bang_tham_so.md.
"""
from torch import nn


def vgg_bn_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class A2VGGBN(nn.Module):
    def __init__(self, num_classes=52, size=64, channels=(32, 64, 128), fc=256):
        super().__init__()
        n = size // 2 ** len(channels)  # 8
        blocks, c_in = [], 3
        for c in channels:
            blocks.append(vgg_bn_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*blocks)  # 3×64×64 -> 32×32×32 -> 64×16×16 -> 128×8×8
        self.classifier = nn.Sequential(        # giữ nguyên như A1
            nn.Flatten(),
            nn.Linear(c_in * n * n, fc),
            nn.ReLU(),
            nn.Linear(fc, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
