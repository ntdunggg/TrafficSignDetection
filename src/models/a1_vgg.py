"""A1: mạng kiểu VGG cho ảnh 3×64×64, 52 lớp.

Ý tưởng VGG (Simonyan & Zisserman 2014): chỉ dùng conv 3×3, padding 1, chồng hai lớp rồi mới max pool.
Hai conv 3×3 có receptive field 5×5 như một conv 5×5 nhưng ít trọng số hơn (18C² so với 25C²)
và có thêm một lần ReLU. Tính tay đầy đủ ở ket_qua/bang_tham_so.md.

Kích thước không gian, công thức (N + 2P − F) / S + 1:
  conv 3×3, P = 1, S = 1 giữ nguyên cạnh: (N + 2 − 3) / 1 + 1 = N.
  max pool 2, S = 2 chia đôi cạnh: 64 -> 32 -> 16 -> 8, flatten 128×8×8 = 8192.
"""
from torch import nn


def vgg_block(c_in, c_out):
    """[conv 3×3 -> ReLU -> conv 3×3 -> ReLU -> max pool 2]: giữ cạnh qua 2 conv, chia đôi ở pool."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class A1VGG(nn.Module):
    def __init__(self, num_classes=52, size=64, channels=(32, 64, 128), fc=256):
        super().__init__()
        n = size // 2 ** len(channels)  # cạnh sau 3 lần pool: 64 / 8 = 8
        blocks, c_in = [], 3
        for c in channels:
            blocks.append(vgg_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*blocks)  # 3×64×64 -> 32×32×32 -> 64×16×16 -> 128×8×8
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # -> 8192
            nn.Linear(c_in * n * n, fc),
            nn.ReLU(),
            nn.Linear(fc, num_classes),         # logits, CrossEntropyLoss tự làm softmax
        )

    def forward(self, x):
        return self.classifier(self.features(x))
