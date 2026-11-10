# Người 3: mạng phân loại A3–A5, A7 (hướng A)

Đọc trước:
- [`00_chung.md`](00_chung.md) và [`nguoi_1_du_lieu_va_A0-A2.md`](nguoi_1_du_lieu_va_A0-A2.md). Bạn dùng bộ ảnh cắt, khung code `src/` và mạng A2 của Người 1.
- [`NHAT_KY.md`](../NHAT_KY.md): D1 (không lật ngang), C6 (nhóm lớp dễ nhầm), mục E phần "Thí nghiệm 7". Phần này là cơ sở lý thuyết và quy tắc chọn cho A5.
- Slide bài 4, bài 5 và giáo trình chương 4 (4.1–4.9).

Vai trò: phần "huấn luyện mạng" của môn. Mỗi bậc đổi **một yếu tố** so với bậc trước, chạy 3 seed, rồi trả lời câu hỏi bằng số liệu.

Cấu hình nền (giữ cố định cho A3, A4, chỉ đổi ở A5):
- Adam, lr 1e-3, batch 128, 30 epoch, đầu vào 64×64.
- Chọn epoch tốt nhất theo val macro-F1.

Đo cho mỗi bậc:
- Val accuracy và macro-F1 (35 lớp có ở val).
- Khoảng cách giữa train và val.
- Số tham số, thời gian train.

**Không chấm test cho đến giai đoạn 3.**

## Giai đoạn 0

- Chạy thử khung code của Người 1 trên máy bạn: `python -m src.train --model a0 --epochs 1`.
- Đọc lại phần lý thuyết bên dưới.
- Viết sẵn phần tăng cường dữ liệu và khối residual để vào giai đoạn 1 là chạy ngay.

## Giai đoạn 1

### A3: chống overfitting (giáo trình 4.7, 4.8)

Xuất phát từ A2. Chạy 4 bản:

| Bản | Dropout | Tăng cường dữ liệu |
|---|---|---|
| A3-0 | không | không (đây chính là A2) |
| A3-d | p = 0.5 trước lớp FC cuối | không |
| A3-a | không | có |
| A3-da | có | có |

**Tăng cường dữ liệu:**
- `RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))`.
- `ColorJitter(brightness=0.3, contrast=0.3)`.
- **Không dùng `RandomHorizontalFlip` hay lật dọc.** Nhiều lớp chỉ khác nhau ở chiều trái/phải, lật ảnh là tạo mẫu sai nhãn (D1).
- Chỉ áp dụng cho train. Val và test chỉ đổi kích thước và chuẩn hóa.

**Đo thêm:** đường cong train loss và val loss của 4 bản trên cùng một hình, để thấy overfitting giảm.

### A4: khối residual (slide ResNet)

Xuất phát từ bản tốt nhất của A3.
- Dựng một mạng sâu "trơn": khoảng 8 khối conv 3×3 + BN + ReLU, giảm kích thước bằng conv stride 2.
- Dựng mạng cùng độ sâu nhưng có **kết nối tắt** kiểu ResNet basic block: 2 conv 3×3, cộng đầu vào. Khi đổi số kênh hoặc stride thì dùng conv 1×1 trên nhánh tắt.

| Bản | Nội dung |
|---|---|
| A4-plain-nông | Mạng trơn, khoảng 4 khối |
| A4-plain-sâu | Mạng trơn, khoảng 8 khối |
| A4-res-sâu | Như A4-plain-sâu nhưng có kết nối tắt |

**Câu cần trả lời:** mạng trơn sâu hơn có tệ hơn cả trên **train** không? Đó là hiện tượng suy biến mà slide nêu, khác với overfitting. Kết nối tắt có sửa được không?

## Giai đoạn 2

### A5: khởi tạo, hàm kích hoạt, optimizer, lịch lr (thay cho Thí nghiệm 7)

Xuất phát từ bản tốt nhất của A4. Đổi **từng yếu tố một**, các yếu tố khác giữ ở giá trị tốt nhất đã có. Mỗi cấu hình chạy 3 seed.

| Yếu tố | Các mức | Bài giảng |
|---|---|---|
| Khởi tạo | mặc định của PyTorch, He (Kaiming normal), Xavier, N(0; 0,01) | 4.3. Mức N(0; 0,01) dùng để thấy gradient bị triệt tiêu |
| Hàm kích hoạt | ReLU, LeakyReLU (0,01), Swish (`nn.SiLU`) | 4.1 |
| Optimizer và lr | SGD + momentum 0,9 lr 0,01; Adam lr 1e-3; Adam lr 5e-4; AdamW lr 1e-3 | 4.6, slide bài 5 |
| Lịch lr | giữ nguyên, giảm theo bậc (×0,1 ở 50% và 75% số epoch), cosin, tuyến tính | 4.5 |

**Quy tắc chọn** (đặt trước khi chạy, lấy từ Thí nghiệm 7):
- Chọn cấu hình có trung bình val macro-F1 cao nhất qua 3 seed.
- Nếu chênh lệch nhỏ hơn độ dao động giữa các seed thì coi là hoà, và chọn cấu hình có độ lệch chuẩn nhỏ hơn.
- Slide khuyên dùng random search. Ở đây dùng lưới nhỏ, đổi từng yếu tố một, vì cần giải thích được từng yếu tố. Nêu hạn chế này trong báo cáo.

**Đo thêm:** đường cong loss và lr theo epoch của từng optimizer và từng lịch lr. Đây là phần nối với lý thuyết: lr lớn thì loss dao động, lr nhỏ thì loss giảm chậm.

### A7: ensemble (giáo trình 4.9)

- Lấy 3–5 mạng tốt nhất (khác seed hoặc khác kiến trúc), lấy trung bình softmax.
- So với mạng đơn tốt nhất về macro-F1, số tham số và thời gian suy diễn.

### Bàn giao cho Người 4 (hết giai đoạn 2)

Giao mạng phân loại tốt nhất cho B1:
- Code mạng.
- File trọng số.
- Kích thước đầu vào, mean và std dùng để chuẩn hóa.

## Giai đoạn 3

- **Chấm test một lần** cho các mạng đã chốt: bản tốt nhất của mỗi bậc A0–A5, và A7. Báo cáo macro-F1 trên 41 lớp có ở test, kèm recall từng lớp.
- **Phân tích confusion matrix (Thí nghiệm 3)** của mạng tốt nhất, theo các nhóm ở mục C6 của sổ:
  - Nhóm cấm rẽ và cấm quay đầu.
  - Các cặp trái/phải: 6/50 (test có 100 và 36 vùng cắt), 1/37 (test chỉ có 2 và 5, quá ít để kết luận). Cặp 15/21 không có ở test.
  - Biển phụ S.505a: 25 và 30. Lớp 28 không có ở test.

  Biển nào hay bị nhầm với biển nào, và có phải vì ảnh cắt quá nhỏ không? Đối chiếu với cột `box_w_px` trong `manifest.csv`.

## Phần báo cáo của bạn

1. **A3–A5, A7:** mỗi bậc có bảng kết quả 3 seed và hình đường cong.
2. **Câu hỏi và trả lời** (mỗi câu kèm số liệu):
   - Overfitting là gì, nhận ra từ đường cong nào? Dropout và tăng cường dữ liệu giảm khoảng cách train và val bao nhiêu?
   - Dropout lúc train và lúc suy diễn khác nhau thế nào?
   - Mạng sâu hơn có luôn tốt hơn không? Hiện tượng suy biến khác overfitting ở đâu? Kết nối tắt giải quyết thế nào?
   - Khởi tạo quá nhỏ thì sao? He và Xavier hợp với hàm kích hoạt nào?
   - ReLU có nhược điểm gì (nơ-ron chết)? LeakyReLU sửa thế nào?
   - SGD + momentum và Adam khác nhau thế nào trên bài toán này? Câu "SGD tốt hơn nhưng khó chỉnh" có đúng không?
   - Lịch lr nào tốt nhất, vì sao?
   - Ensemble tăng được bao nhiêu, đổi lại tốn gì?
   - **Transfer learning (chỉ lý thuyết, giáo trình 4.10):** cô không cho dùng mạng đã học trước. Nếu được dùng thì kỳ vọng gì với dữ liệu ít như thế này?
