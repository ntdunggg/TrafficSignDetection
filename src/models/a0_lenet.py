"""A0: mạng kiểu LeNet-5 cho ảnh 3×64×64, 52 lớp.

Khác LeNet-5 gốc (LeCun 1998): dùng ReLU thay tanh, max pool thay average pool, ảnh màu 3 kênh,
đầu ra 52 lớp thay 10.

Kích thước không gian, công thức (N + 2P − F) / S + 1:
  64 --conv 5×5--> 60 --pool 2--> 30 --conv 5×5--> 26 --pool 2--> 13, flatten 16×13×13 = 2704.
"""
from torch import nn


class A0LeNet(nn.Module):
    def __init__(self, num_classes=52, size=64):
        super().__init__()
        n = ((size - 4) // 2 - 4) // 2  # cạnh sau khối conv thứ hai (13 khi size = 64)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),   # 3×64×64 -> 6×60×60
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 6×30×30
            nn.Conv2d(6, 16, kernel_size=5),  # -> 16×26×26
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 16×13×13
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # -> 2704
            nn.Linear(16 * n * n, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),       # logits, CrossEntropyLoss tự làm softmax
        )

    def forward(self, x):
        return self.classifier(self.features(x))
