"""A4: mạng trơn nông, mạng trơn sâu, mạng sâu có kết nối tắt (He và cộng sự 2016, slide ResNet).

Kế hoạch: ke_hoach/nguoi_3_minh_quan_A3-A7.md, mục A4. Câu cần trả lời: mạng trơn sâu hơn có tệ hơn
cả trên TRAIN không (suy biến, khác overfitting)? Kết nối tắt có sửa được không?

Khung chung của cả ba bản (chỉ khác ở phần thân của mỗi tầng):
- Stem: conv 3×3, 3→32, stride 1 → BN → ReLU. Đầu ra 32×64×64.
- 4 tầng, số kênh (32, 64, 128, 256). Tầng 1 giữ kích thước, tầng 2–4 giảm một nửa bằng conv stride 2
  (không dùng max pool, đúng kế hoạch). Kích thước: 64 → 32 → 16 → 8.
- Đầu phân loại: global average pooling → FC 256→52. Không dùng FC 8192→256 như A2, để số tham số
  nằm ở phần conv, là phần A4 muốn so.
- Mọi conv đặt bias=False vì có BN ngay sau (như A2).

Ba bản:
- plain_nong ("a4pn"): mỗi tầng 1 conv 3×3 + BN + ReLU. Tổng 1 + 4 = 5 conv.
- plain_sau  ("a4ps"): mỗi tầng 2 conv 3×3 + BN + ReLU. Tổng 1 + 8 = 9 conv.
- res_sau    ("a4rs"): như plain_sau, nhưng 2 conv của mỗi tầng là một basic block có kết nối tắt:
      y = ReLU( BN(conv(ReLU(BN(conv(x))))) + tắt(x) )
  tắt(x) = x nếu số kênh và kích thước không đổi (tầng 1),
  tắt(x) = BN(conv 1×1 stride s (x)) nếu đổi (tầng 2–4), đúng kế hoạch (He 2016 gọi là phương án B).
  Nhánh tắt 1×1 thêm 43.008 trọng số + 896 tham số BN = 43.904, tức 3,7% so với plain_sau (1.195.860).

plain_nong và plain_sau chỉ khác độ sâu. plain_sau và res_sau chỉ khác kết nối tắt.
Khởi tạo: mặc định của PyTorch cho cả ba (A5 mới thử He, Xavier).
Tính tay ở ket_qua/bang_tham_so.md, mục A4.
"""
from torch import nn

CHANNELS = (32, 64, 128, 256)


def conv_bn_relu(c_in, c_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class BasicBlock(nn.Module):
    """Hai conv 3×3 và một kết nối tắt. ReLU cuối đặt sau phép cộng (như bài báo ResNet)."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class A4Net(nn.Module):
    def __init__(self, kind, num_classes=52, size=64, channels=CHANNELS):
        super().__init__()
        assert kind in ("plain_nong", "plain_sau", "res_sau")
        self.stem = conv_bn_relu(3, channels[0])
        stages, c_in = [], channels[0]
        for i, c in enumerate(channels):
            s = 1 if i == 0 else 2
            if kind == "plain_nong":
                stages.append(conv_bn_relu(c_in, c, s))
            elif kind == "plain_sau":
                stages.append(nn.Sequential(conv_bn_relu(c_in, c, s), conv_bn_relu(c, c)))
            else:
                stages.append(BasicBlock(c_in, c, s))
            c_in = c
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class A4PlainNong(A4Net):
    def __init__(self, num_classes=52, size=64):
        super().__init__("plain_nong", num_classes, size)


class A4PlainSau(A4Net):
    def __init__(self, num_classes=52, size=64):
        super().__init__("plain_sau", num_classes, size)


class A4ResSau(A4Net):
    def __init__(self, num_classes=52, size=64):
        super().__init__("res_sau", num_classes, size)
