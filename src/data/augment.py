"""Các kiểu tăng cường dữ liệu, chọn bằng --augment trong train.py.

Quy tắc:
- Nhận và trả về tensor float 3×H×W, giá trị 0..1 (chưa chuẩn hoá).
- KHÔNG lật ngang hay lật dọc: nhiều lớp chỉ khác nhau ở chiều trái/phải (NHAT_KY.md mục D1).
- Chỉ dùng cho train; val và test không tăng cường.
- Ngẫu nhiên lấy từ bộ sinh số chung của torch, set_seed trong train.py cố định nó
  (DataLoader chạy num_workers=0 nên thứ tự gọi cố định, chạy lại cùng seed ra cùng ảnh).

"a3" (giáo trình 4.8, kế hoạch A3 trong ke_hoach/nguoi_3_minh_quan_A3-A7.md):
- RandomAffine: xoay ±10°, dịch tối đa 10% cạnh ảnh mỗi chiều, co giãn 0,9–1,1.
  Mô phỏng biển chụp hơi nghiêng, lệch tâm box, xa gần khác nhau. Góc lộ ra sau khi xoay/dịch
  được tô đen (fill=0, mặc định của torchvision).
- ColorJitter: độ sáng và độ tương phản nhân hệ số ngẫu nhiên trong 0,7–1,3.
  Mô phỏng nắng, bóng râm, ngược sáng. Không đổi sắc độ (hue), vì màu đỏ/xanh/vàng mang nghĩa của biển.
"""
from torchvision.transforms import v2

AUGMENTS = {
    "none": None,
    "a3": v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.3, contrast=0.3),
    ]),
}
