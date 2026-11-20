"""A3-d: A2 thêm Dropout p = 0,5 trước lớp FC cuối (Srivastava và cộng sự 2014, giáo trình 4.7).

Chỉ đổi một yếu tố so với A2: classifier thành Flatten → FC 256 → ReLU → Dropout(0,5) → FC 52.
Phần conv giữ nguyên (cùng vgg_bn_block của A2).

- Dropout không có tham số: số tham số bằng đúng A2 (2.398.228).
- Lúc train (model.train()): mỗi lượt, mỗi phần tử của vector 256 bị đặt về 0 với xác suất p,
  phần còn lại nhân 1/(1−p) = 2 để giữ nguyên kỳ vọng (inverted dropout của PyTorch).
- Lúc suy diễn (model.eval()): Dropout không làm gì. evaluate.predict gọi model.eval().
- Đặt ở FC vì FC 8192→256 chiếm 87,5% tham số (ket_qua/bang_tham_so.md), là chỗ dễ học thuộc nhất.
"""
from torch import nn

from .a2_vgg_bn import vgg_bn_block


class A3Dropout(nn.Module):
    def __init__(self, num_classes=52, size=64, channels=(32, 64, 128), fc=256, p=0.5):
        super().__init__()
        n = size // 2 ** len(channels)  # 8
        blocks, c_in = [], 3
        for c in channels:
            blocks.append(vgg_bn_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * n * n, fc),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(fc, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
