"""Danh sách mạng của hướng A. Thêm mạng mới: tạo file trong src/models/ rồi thêm một dòng vào MODELS.

Mỗi lớp mạng nhận (num_classes=52, size=64) và khởi tạo ngẫu nhiên.
Không dùng trọng số đã học trước (AGENTS.md mục 2).
"""
from .a0_lenet import A0LeNet

MODELS = {
    "a0": A0LeNet,
    # "a1": A1VGG,      # Dũng, giai đoạn 1
    # "a2": A2VGGBN,    # Dũng, giai đoạn 1
}


def get_model(name, num_classes=52, size=64):
    if name not in MODELS:
        raise SystemExit(f"Không có mạng '{name}'. Có: {', '.join(MODELS)}")
    return MODELS[name](num_classes=num_classes, size=size)
