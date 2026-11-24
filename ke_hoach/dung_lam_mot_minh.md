# Kế hoạch khi chỉ còn Dũng (từ 2026-11-19)

Từ ngày 2026-11-19 nhóm chỉ còn Dũng. Hạn nộp còn khoảng 2–3 tuần, tức trong khoảng 03/12 đến 10/12/2026. File này **thay cho** phân công 4 người ở [`00_chung.md`](00_chung.md).

Các file `nguoi_2`, `nguoi_3`, `nguoi_4` và thư mục `nhat_ky/` giữ lại làm lịch sử. Không làm theo phân công trong đó nữa, nhưng vẫn dùng phần mô tả chi tiết của từng thí nghiệm. File này trỏ tới đúng mục cần đọc.

Đọc trước:
- [`00_chung.md`](00_chung.md): mục "Quy tắc chung", "Môi trường", "Chạy trên Kaggle". Bảng phân công và bàn giao ở đó không còn dùng.
- [`NHAT_KY.md`](../NHAT_KY.md): mục A1 (rò rỉ), D1 (không lật ngang), C6 (nhóm dễ nhầm), mục E "Kế hoạch chính", mục H (câu hỏi báo cáo).

## Cắt bớt theo nguyên tắc nào

Cô chấm nặng nhất hai phần: tự xây mạng, và trả lời câu hỏi có số liệu (`00_chung.md`, đầu file).

- **Giữ toàn bộ hướng A** (CNN tự viết bằng PyTorch).
  - Đây đúng là phần "tự xây".
  - Rẻ: A0 mất khoảng 1–2 giây mỗi epoch trên M2 (NHAT_KY, lịch sử 2026-11-14), nên một lần chạy 30 epoch chỉ vài phút.
- **Cắt gần hết hướng B** (YOLO).
  - YOLO tạo từ file `yolov8s.yaml` của Ultralytics, nên không phải mạng do mình tự xây.
  - Rất tốn: 551 giây mỗi epoch trên M2 (NHAT_KY, mục "Chạy thử 1 epoch"). B0, B1, B2a cộng lại khoảng 900 epoch GPU.
- **Giữ các câu hỏi H1–H4.** Trả lời bằng thí nghiệm rẻ nhất có số liệu. Câu nào không có thí nghiệm thì trả lời bằng lý thuyết và ghi rõ là lý thuyết.

| Việc | Quyết định | Lý do |
|---|---|---|
| A0, A1, A2, A3, A4 | Giữ đủ | Phần xây mạng, rẻ |
| A5 | Rút gọn: optimizer và lịch lr. Khởi tạo và hàm kích hoạt là phần làm thêm | Cần thêm code cho khởi tạo và hàm kích hoạt (`train.py` chưa có tùy chọn) |
| A7 ensemble | Giữ | Chỉ lấy trung bình softmax của các lần chạy đã có, không train thêm |
| Thí nghiệm 5 (chia ngẫu nhiên so với chia theo nhóm) | Giữ | Bằng chứng cho H1, H4. Chỉ cần dựng `manifest_random.csv` rồi train lại |
| Thí nghiệm 3 (confusion matrix) | Giữ | `evaluate.py` đã ghi confusion matrix, không train thêm |
| B0 (YOLO 52 lớp từ đầu) | Làm thêm, 1 seed, nếu đo thời gian thấy kịp | Cho một con số để so với hướng A |
| B1, B2a, chọn lr cho YOLO | Bỏ | Mỗi việc cần thêm nhiều lần train YOLO |
| Thí nghiệm 6 (tự chụp ảnh ngoài phân phối) | Bỏ, H3 trả lời bằng lý thuyết và một phép thử rẻ (bước 2.5) | Chụp và gán nhãn 150–200 ảnh tốn nhiều ngày công |
| Thí nghiệm 1, 2, 4 | Bỏ | Đã là phần làm thêm từ trước |

## Cấu hình cố định

- Cấu hình nền giữ cố định từ A0 đến A4: Adam, lr 1e-3, batch 128, 30 epoch, cross-entropy, đầu vào 64×64. Chọn epoch tốt nhất theo val macro-F1.
- Mỗi cấu hình chạy 3 seed (0, 1, 2), báo cáo trung bình ± độ lệch chuẩn.
- Đổi một yếu tố mỗi lần so với bản trước.
- Không lật ảnh.
- Chọn trên val (35 lớp). Test (41 lớp) chỉ chấm một lần, ở bước 2.4.
- Chạy với `--nguoi Dũng --thi-nghiem <tên bậc>` để kết quả tự ghi vào `ket_qua/ket_qua.csv`.
- Mỗi việc một nhánh `dung/<việc>`, gộp vào `main` qua pull request.

## Tuần 1 (19/11 – 25/11): xây mạng

### 1.1. A0, A1, A2 (ngày 1–3)

Chi tiết kiến trúc, đáp số tham số, công thức receptive field: [`nguoi_1_dung_du_lieu_va_A0-A2.md`](nguoi_1_dung_du_lieu_va_A0-A2.md), mục "Giai đoạn 1".

1. **A0:**
   - Tính tay kích thước đầu ra và số tham số từng lớp.
   - Kiểm bằng `python -m src.summary --model a0`. Phải ra 342.056 tham số.
   - Chạy 3 seed.
2. **A1:**
   - Tính tay trước. Phải ra 2.397.780 tham số.
   - Viết `src/models/a1_vgg.py`, thêm vào `MODELS` trong `src/models/__init__.py`.
   - Kiểm bằng `src.summary`, rồi chạy 3 seed.
3. **A2:**
   - A1 thêm BatchNorm sau mỗi conv, conv đặt `bias=False`.
   - Đếm tham số học được (γ, β) và buffer (running mean, var).
   - Chạy 3 seed.
   - So đường cong val macro-F1 theo epoch với A1.

**Xong khi:** có 9 dòng trong `ket_qua.csv` và bảng tham số tính tay của 3 mạng.

### 1.2. A3: chống overfitting (ngày 4–5)

Bốn bản A3-0, A3-d, A3-a, A3-da. Cách tăng cường dữ liệu và cách đo: [`nguoi_3_minh_quan_A3-A7.md`](nguoi_3_minh_quan_A3-A7.md), mục "A3".
- Thêm `"a3"` vào `AUGMENTS` trong `src/data/augment.py`.
- Chỉ dùng `RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))` và `ColorJitter(brightness=0.3, contrast=0.3)`. Không lật.
- Dropout p = 0,5 trước lớp FC cuối: viết thành mạng mới (ví dụ `a3d`) từ A2.
- A3-0 chính là A2, không chạy lại.

**Đo thêm:** khoảng cách train và val của 4 bản, vẽ train loss và val loss trên cùng một hình.

### 1.3. A4: khối residual (ngày 6–7)

Ba bản A4-plain-nông, A4-plain-sâu, A4-res-sâu, xuất phát từ bản tốt nhất của A3: [`nguoi_3_minh_quan_A3-A7.md`](nguoi_3_minh_quan_A3-A7.md), mục "A4".

**Câu cần trả lời:** mạng trơn sâu hơn có tệ hơn cả trên **train** không (suy biến, khác overfitting)? Kết nối tắt có sửa được không?

### 1.4. `manifest_random.csv` cho Thí nghiệm 5 (xen vào ngày 6–7)

Cách dựng và cách kiểm: [`nguoi_1_dung_du_lieu_va_A0-A2.md`](nguoi_1_dung_du_lieu_va_A0-A2.md), mục "Bản ảnh cắt chia ngẫu nhiên cho Thí nghiệm 5".

### 1.5. B0 trên Kaggle: chỉ đo thời gian (chạy nền)

- Chạy 1 epoch `yolov8s.yaml`, `imgsz=960`, `fliplr=0` trên GPU của Kaggle. Ghi lại loại GPU, số giây mỗi epoch, bộ nhớ.
- **Quy tắc quyết định, đặt trước khi đo:**
  - Nếu 100 epoch vừa trong một phiên Kaggle (dưới khoảng 10 giờ, còn dư so với giới hạn 12 giờ) thì làm B0 ở tuần 2.
  - Nếu không vừa thì bỏ B0, ghi số đã đo làm lý do.
- Cấu hình B0 nếu chạy: 1 seed, đặt cố định `optimizer=SGD lr0=0.01 momentum=0.937`, không để `optimizer=auto`.
  - Lý do: `auto` sẽ tự đổi optimizer theo số bước (NHAT_KY, "Chạy thử 1 epoch", phát hiện 1).
  - Train từ đầu cần lr lớn hơn mức 0,000179 dành cho tinh chỉnh ngắn (NHAT_KY, "Chọn lr cho YOLO train từ đầu").
  - Hạn chế phải ghi trong báo cáo: không tìm lr, chỉ 1 seed.
- **Đã đo (2026-11-22):** Tesla T4, 214 giây mỗi epoch, ước tính 100 epoch mất 5,95 giờ. **Làm B0 ở tuần 2.** Chi tiết ở NHAT_KY, mục 2026-11-22.
- **Đã chạy B0 (2026-11-23):** 1,69 giờ (chép dữ liệu sang `/tmp` nên nhanh hơn ước tính). mAP50-95 test 0,3813, mAP50 0,5027 trên 41 lớp. Chi tiết ở NHAT_KY, mục 2026-11-24.
- **Đổi (2026-11-24, Dũng đồng ý):** vì mỗi seed chỉ mất khoảng 1,7 giờ, chạy thêm seed 1 và 2 để B0 có 3 seed như các thí nghiệm khác. Lý do ở NHAT_KY, mục 2026-11-24.

## Tuần 2 (26/11 – 02/12): huấn luyện và đánh giá

### 2.1. A5 rút gọn (ngày 1–2)

Xuất phát từ bản tốt nhất của A4. Lưới, quy tắc chọn và cách đo: [`nguoi_3_minh_quan_A3-A7.md`](nguoi_3_minh_quan_A3-A7.md), mục "A5".

| Yếu tố | Các mức | `train.py` |
|---|---|---|
| Optimizer và lr | SGD + momentum 0,9 lr 0,01; Adam lr 1e-3; Adam lr 5e-4; AdamW lr 1e-3 | `--opt`, `--lr`, `--momentum` đã có |
| Lịch lr (với optimizer thắng) | none, step, cosine, linear | `--sched` đã có |
| Làm thêm: khởi tạo, hàm kích hoạt | Như file của Minh Quân | Cần viết thêm code |

Khoảng 7 cấu hình × 3 seed, mỗi lần vài phút.

### 2.2. A7 ensemble và Thí nghiệm 5 (ngày 3)

- **A7:** lấy trung bình softmax của 3–5 mạng tốt nhất trên val. So với mạng đơn tốt nhất về macro-F1, số tham số, thời gian suy diễn.
- **Thí nghiệm 5:** train A2 trên `manifest.csv` (G) và `manifest_random.csv` (R), mỗi bản 3 seed. Cách đọc kết quả: [`nguoi_4_anh_duc_danh_gia_va_B1.md`](nguoi_4_anh_duc_danh_gia_va_B1.md), mục "Giai đoạn 1".
  - Kết quả G của Thí nghiệm 5 là 3 lần chạy A2 ở bước 1.1, không phải chạy lại.
  - Macro-F1 tính trên các lớp có ở cả hai test.
  - Chưa chấm test ở bước này. Test của G và của R cùng chấm ở bước 2.4. Thí nghiệm này không dùng để chọn cấu hình, nên vẫn giữ quy tắc chấm test một lần.

### 2.3. B0 (nếu bước 1.5 quyết định làm)

Chạy bằng **Save & Run All** trên Kaggle (`00_chung.md`, mục "Chạy trên Kaggle"). Chấm `split=test` một lần, ghi mAP50, mAP50-95 trên 41 lớp.

### 2.4. Chốt và chấm test (ngày 4)

- Chốt trên val: bản tốt nhất của mỗi bậc A0–A5, và A7.
- Chấm test một lần: `python -m src.evaluate --run <...> --split test --cham-test`.
- Báo cáo macro-F1 trên 41 lớp, kèm recall từng lớp.
- Chấm luôn test của Thí nghiệm 5: G (A2 trên `manifest.csv`) và R (A2 trên `manifest_random.csv`).

### 2.5. Confusion matrix và H2, H3 (ngày 5)

- **Thí nghiệm 3:** nhóm dễ nhầm của mục C6, đối chiếu với cỡ ảnh cắt (`box_w_px`). Chi tiết: [`nguoi_3_minh_quan_A3-A7.md`](nguoi_3_minh_quan_A3-A7.md), mục "Giai đoạn 3".
- **H3 (ảnh ngoài phân phối):**
  - Phần chính trả lời bằng lý thuyết.
  - Thêm một phép thử rẻ, không cần gán nhãn: cắt ngẫu nhiên vài trăm vùng nền không chứa box nào từ ảnh test, đưa qua mạng tốt nhất, đếm bao nhiêu vùng bị gán lớp với xác suất softmax cao (ví dụ > 0,9).
  - Kết quả cho thấy mạng phân loại luôn chọn một trong 52 lớp, kể cả khi ảnh không phải biển báo.
  - Đây là đề xuất mới, sẽ ghi vào sổ khi làm.
- **H2 (phân đoạn rồi phân loại, hay ngược lại):** trả lời bằng lý thuyết (giáo trình 6.3–6.4). Nếu có B0 thì so với hướng A ở mức ý tưởng hai giai đoạn và một giai đoạn. Ghi rõ là không có số liệu B1.

## Tuần 3 (03/12 – 09/12): báo cáo và dự phòng

- Ghép báo cáo từ `NHAT_KY.md` và `ket_qua/ket_qua.csv`. Cấu trúc gồm:
  - chương dữ liệu (mục A–D);
  - hướng A, từ A0 đến A7;
  - B0 nếu có;
  - các câu H1–H4.
- **Bảng tham số cho mọi mạng hướng A:** kích thước đầu ra, số tham số, receptive field từng lớp. Tính tay, kiểm bằng `src/summary.py`.
- Nếu tuần 1–2 bị trễ thì lấy thời gian ở đây. Thứ tự bỏ khi thiếu thời gian: B0 → phần làm thêm của A5 → A7.

## Câu hỏi báo cáo phải trả lời (gộp từ 4 file cũ)

Mỗi câu kèm số liệu. Câu nào chỉ trả lời bằng lý thuyết thì ghi rõ.

**Dữ liệu:**
- Vì sao phải chia theo nhóm frame mà không chia ngẫu nhiên từng ảnh? (A1, Thí nghiệm 5)
- Vì sao không lật ngang ảnh? (D1)
- Ảnh cắt nhỏ ảnh hưởng thế nào khi phóng lên 64×64? 71,4% ảnh cắt có cạnh ≤ 64 px.
- H1: thí nghiệm cần lưu ý gì về tập nhãn?

**Xây mạng (A0–A2, A4):**
- A0 và A1 có bao nhiêu tham số, lớp nào chiếm nhiều nhất, vì sao?
- A0 khác LeNet-5 gốc ở đâu?
- Hai lớp conv 3×3 so với một lớp conv 5×5: receptive field và số tham số?
- BatchNorm có bao nhiêu tham số? Lúc train và lúc suy diễn khác nhau thế nào? A2 có hội tụ nhanh hơn A1 không?
- Mạng sâu hơn có luôn tốt hơn không? Suy biến khác overfitting ở đâu? Kết nối tắt giải quyết thế nào?

**Huấn luyện (A3, A5, A7):**
- Overfitting nhận ra từ đường cong nào? Dropout và tăng cường dữ liệu giảm khoảng cách train và val bao nhiêu?
- Dropout lúc train và lúc suy diễn khác nhau thế nào?
- SGD + momentum và Adam khác nhau thế nào trên bài toán này? Lịch lr nào tốt nhất, vì sao?
- Ensemble tăng được bao nhiêu, đổi lại tốn gì?
- Transfer learning (chỉ lý thuyết, giáo trình 4.10): nếu được dùng thì kỳ vọng gì với dữ liệu ít như thế này?

**Đánh giá:**
- H2: phân đoạn trước rồi phân loại, hay ngược lại, thì sao?
- H3: gặp ảnh ngoài phân phối thì sao?
- H4: ảnh mới không giống ảnh train thì kết quả thế nào, có phải overfitting không? (Thí nghiệm 5)
- Biển nào hay bị nhầm với biển nào, có phải vì ảnh cắt nhỏ không? (Thí nghiệm 3)
