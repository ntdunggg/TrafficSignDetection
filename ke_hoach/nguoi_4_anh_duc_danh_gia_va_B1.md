# Người 4 (Anh Đức): đánh giá, ảnh mới, hệ hai giai đoạn B1, ghép báo cáo

Đọc trước:
- [`00_chung.md`](00_chung.md).
- [`NHAT_KY.md`](../NHAT_KY.md): mục A1 (rò rỉ dữ liệu), mục E phần Thí nghiệm 5 và 6, mục H (4 câu hỏi cho báo cáo).
- Giáo trình 6.3–6.4 (mạng hai giai đoạn và một giai đoạn).

Vai trò: kiểm tra mô hình có tốt thật không, trên ảnh mới và khi chia dữ liệu đúng cách. Ghép hệ hai giai đoạn. Cuối cùng ghép báo cáo của cả nhóm.

## Giai đoạn 0 (bắt đầu ngay): bộ ảnh ngoài phân phối cho Thí nghiệm 6

Việc này không cần GPU nhưng tốn nhiều ngày công, nên làm từ đầu.

**Thu thập 150–200 ảnh khác hẳn dữ liệu train.** Dữ liệu train quay ở TP.HCM bằng một camera, ảnh 960×540 hoặc 1920×1080.
- Chụp bằng điện thoại ở nơi khác: Hà Nội, các quận khác, đường liên tỉnh.
- Nhiều điều kiện: ban đêm, trời mưa, ngược sáng, góc nghiêng, biển bị che một phần.
- Thêm khoảng 30 ảnh **không có biển nào trong 52 lớp** nhưng có vật dễ nhầm: biển hiệu cửa hàng tròn màu đỏ, đèn giao thông, biển báo loại khác. Nhóm ảnh này dùng để đếm số lần báo nhầm.

**Gán nhãn:**
- Dùng LabelImg, định dạng YOLO, với `classes.txt` của Dũng (đúng thứ tự id 52 lớp).
- Gán **mọi** biển thuộc 52 lớp nhìn thấy được, kể cả biển xa. `archive` cũng gán cả biển rất xa (NHAT_KY, bảng Đính chính).
- Biển không thuộc 52 lớp thì **không gán**, nhưng đếm riêng: biển gì, bao nhiêu.
- Box ôm sát biển, không lấy cột.

**Cấu trúc:**
- `data/ood/images/`, `data/ood/labels/`. Thư mục `data/` đã git ignore.
- Thêm `data/ood/meta.csv` với các cột `stem,noi_chup,ngay_dem,thoi_tiet,ghi_chu`.

**Riêng tư:** ảnh có người và biển số xe, không đăng công khai.

**Xong khi:** đủ ảnh, đã tự xem lại nhãn một lượt, giao cho Sỹ Nguyên và Minh Quân trước giai đoạn 3.

## Giai đoạn 1: Thí nghiệm 5 trên mạng phân loại (trả lời câu H4)

**Câu hỏi:** nếu test giống train, điểm bị thổi phồng bao nhiêu?

| Bản | Train và val | Test |
|---|---|---|
| G | `manifest.csv` (chia theo nhóm frame, không rò rỉ) | test của G |
| R | `manifest_random.csv` (chia ngẫu nhiên theo ảnh như split gốc, Dũng dựng) | test của R |

- Dùng cùng mạng (A2 của Dũng) và cùng cấu hình train. Mỗi bản 3 seed.
- Tính macro-F1 trên **cùng tập lớp**: các lớp có ở cả hai test.
- **Đọc kết quả:** R cao hơn G bao nhiêu thì đó là phần mô hình "nhớ cảnh đã thấy" chứ không phải nhận biết biển tốt hơn.
- Kèm số liệu rò rỉ: tỉ lệ ảnh test có ảnh train gần trùng. Split gốc 30,7%, split nhóm 0% (NHAT_KY mục A1).
- Vẽ đường cong train và val của cả hai bản. Ở bản R, val cũng bị rò rỉ, nên đường cong trông "đẹp" mà không phát hiện được overfitting.

## Giai đoạn 2: script đánh giá cho B1

B1 là hệ hai giai đoạn:
1. YOLO 1 lớp (Sỹ Nguyên train, `single_cls=True`) tìm vùng có biển.
2. Cắt từng vùng **bằng đúng hàm `crop_box` của Dũng**.
3. Mạng phân loại tốt nhất của Minh Quân cho ra lớp.

- Lớp của mỗi box là lớp có xác suất lớn nhất.
- Điểm của box = độ tin cậy của YOLO × xác suất của lớp đó.

Viết `src/eval_b1.py`:
- Chạy YOLO 1 lớp trên ảnh test với `conf=0.001` (ngưỡng thấp, như khi tính mAP), cắt vùng, phân loại.
- Tính mAP50 và mAP50-95 **bằng cùng cách Ultralytics tính**: ghép box dự đoán với box thật theo IoU, rồi dùng `ultralytics.utils.metrics.ap_per_class`.

**Kiểm tra script trước khi tin:**
1. Cho dự đoán của B0 (Sỹ Nguyên) đi qua script của bạn.
2. Số ra phải khớp với số `yolo val split=test` của B0, sai lệch rất nhỏ.

Chưa khớp thì chưa dùng script cho B1.

## Giai đoạn 3

### Chấm B1 và so với B0 (trả lời câu H2)

- mAP50, mAP50-95 trên 41 lớp của test, so với B0.
- **Phân tích lỗi**, chia mỗi box thật thành 3 loại:
  - YOLO 1 lớp bỏ sót: không có box dự đoán nào có IoU ≥ 0,5.
  - Tìm đúng vị trí nhưng phân loại sai.
  - Đúng.

  Loại thứ nhất là "lỗi dồn tầng" của mạng hai giai đoạn: bước 1 sót thì bước 2 không cứu được.
- So recall theo cỡ box: cạnh dài < 32 px, 32–64 px, > 64 px.

### Chấm Thí nghiệm 6 (trả lời câu H3)

Chấm B0 và B1 trên bộ ảnh ngoài phân phối:
- mAP so với test thường.
- Recall theo từng điều kiện trong `meta.csv`: ngày, đêm, mưa.
- Số box báo nhầm trên 30 ảnh không có biển, kèm độ tin cậy của các box đó.

YOLO không có lớp "không biết", nên mọi vật giống biển đều có thể bị báo nhầm với độ tin cậy cao.

## Giai đoạn 4: ghép báo cáo

Gom phần của cả 4 người thành một báo cáo:
- Thống nhất ký hiệu (A0–A7, B0–B2a, Thí nghiệm 3, 5, 6).
- Mọi số liệu lấy từ `ket_qua/ket_qua.csv`.
- Mọi mAP của test ghi rõ "trên 41 lớp".
- Điền số liệu thật vào câu trả lời H1–H4 trong sổ.

Đề xuất bố cục:
1. Giới thiệu và dữ liệu.
2. Hướng A: mạng phân loại tự xây.
3. Hướng B: phát hiện đối tượng.
4. Đánh giá.
5. Trả lời câu hỏi.
6. Hạn chế.

## Phần báo cáo của bạn

1. **Đánh giá:** Thí nghiệm 5, B1 so với B0, Thí nghiệm 6.
2. **Câu hỏi và trả lời** (mỗi câu kèm số liệu):
   - **H2:** phân đoạn hoặc tìm vùng trước rồi phân loại, so với làm cùng lúc, thì sao? Lỗi dồn tầng chiếm bao nhiêu?
   - **H3:** gặp ảnh ngoài phân phối thì sao? Điểm giảm bao nhiêu, báo nhầm bao nhiêu?
   - **H4:** ảnh mới không giống train thì kết quả thế nào? Có phải overfitting không? Phân biệt overfitting với lệch phân phối bằng số liệu của Thí nghiệm 5 và 6.
   - Vì sao chia ngẫu nhiên từng ảnh làm điểm cao giả? (Thí nghiệm 5 và mục A1)
   - Vì sao tính mAP phải dùng ngưỡng độ tin cậy thấp?
