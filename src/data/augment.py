"""Các kiểu tăng cường dữ liệu, chọn bằng --augment trong train.py.

Minh Quân thêm bản cho A3 vào đây (giáo trình 4.8). Quy tắc:
- Nhận và trả về tensor float 3×H×W, giá trị 0..1 (chưa chuẩn hoá).
- KHÔNG lật ngang hay lật dọc: nhiều lớp chỉ khác nhau ở chiều trái/phải (NHAT_KY.md mục D1).
- Chỉ dùng cho train; val và test không tăng cường.
"""

AUGMENTS = {
    "none": None,
    # "a3": torchvision.transforms.v2.Compose([...]),  # Minh Quân viết ở giai đoạn 0
}
