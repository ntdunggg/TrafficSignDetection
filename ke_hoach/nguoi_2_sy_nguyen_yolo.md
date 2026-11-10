# Người 2 (Sỹ Nguyên): YOLO train từ đầu (hướng B)

Đọc trước:
- [`00_chung.md`](00_chung.md).
- [`NHAT_KY.md`](../NHAT_KY.md): D1 và D2 (không lật ngang, `imgsz=960`), mục E phần "Chạy thử 1 epoch" (7 lưu ý khi train thật) và phần "Kế hoạch chính".
- Giáo trình `dl_lecturenotes` chương 6.3–6.4 (R-CNN, YOLO, SSD).

Vai trò: mọi việc liên quan đến YOLO. **Không dùng file `.pt` có sẵn** (cô không cho dùng mạng đã học trước). Tạo mạng bằng file `.yaml`.

Tham số cố định cho mọi lần chạy: `ultralytics==8.4.191`, `imgsz=960`, `fliplr=0`, cùng `batch`, ghi md5 thư mục nhãn (xem `00_chung.md`). Optimizer và lr chọn ở bước 0.2.

## Giai đoạn 0: chuẩn bị

### 0.1. Notebook YOLO mẫu trên Kaggle

Notebook gồm các bước:
1. Cài đúng `pip install ultralytics==8.4.191`.
2. Lấy dataset riêng tư mà Dũng chia sẻ.
3. Tính md5 thư mục nhãn.
4. Train.
5. Lưu `results.csv`, `args.yaml`, `weights/best.pt` vào output.

Thư mục input của Kaggle chỉ đọc, nên Ultralytics sẽ cảnh báo không ghi được file `.cache`. Cảnh báo này không làm sai kết quả, chỉ làm lần quét nhãn sau chậm hơn một chút. Muốn có cache thì chép dataset sang `/kaggle/working` trước.

Đo thời gian 1 epoch:

```bash
yolo detect train data=.../data.yaml model=yolov8s.yaml epochs=1 imgsz=960 batch=16 fliplr=0 seed=0 name=do_thoi_gian
```

Ghi lại GPU nào (T4, P100...), thời gian, bộ nhớ GPU. Nếu thiếu bộ nhớ thì giảm `batch`, và từ đó cả nhóm dùng chung một giá trị `batch`.

### 0.2. Chọn lr cho YOLO train từ đầu

Lý do (NHAT_KY, "Kế hoạch chính"): cấu hình tạm `AdamW lr0=0.000179` là giá trị Ultralytics tự chọn khi tinh chỉnh ngắn. Train từ đầu thường cần lr lớn hơn.

Chạy 3 mức, mỗi mức 30 epoch, seed 0:

| Tên | Tham số |
|---|---|
| `lr_adamw_179` | `optimizer=AdamW lr0=0.000179 momentum=0.9` |
| `lr_adamw_1e3` | `optimizer=AdamW lr0=0.001 momentum=0.9` |
| `lr_sgd_1e2` | `optimizer=SGD lr0=0.01 momentum=0.937` |

```bash
yolo detect train data=.../data.yaml model=yolov8s.yaml epochs=30 imgsz=960 batch=16 fliplr=0 seed=0 optimizer=AdamW lr0=0.001 momentum=0.9 name=lr_adamw_1e3
```

- Phải ghi cả `momentum`. Với AdamW, Ultralytics dùng `momentum` làm beta1, nên bỏ trống sẽ thành 0,937.
- **Chọn mức có mAP50-95 trên val cao nhất ở epoch cuối. Không chấm test.**
- Gửi kết quả cho Dũng ghi vào sổ, rồi báo cả nhóm.
- Hạn chế cần nêu trong báo cáo: lr được chọn từ lần chạy ngắn, 1 seed.

### 0.3. Chốt số epoch E

Train từ đầu cần nhiều epoch hơn tinh chỉnh. Nhìn đường cong của lần chạy 0.2: nếu mAP val còn tăng rõ ở epoch 30 thì E phải lớn, khoảng 150–300.

Cân với quota GPU. Hướng B cần khoảng 9 × E epoch (B0 3 seed, B2a 3 seed, YOLO 1 lớp 1–3 seed). Báo E cho cả nhóm.

## Giai đoạn 1: B0, mốc phát hiện 52 lớp

```bash
yolo detect train data=.../data.yaml model=yolov8s.yaml epochs=E imgsz=960 batch=16 fliplr=0 seed=0 <optimizer, lr đã chọn> name=b0_s0
```

- Chạy với seed 0, 1, 2.
- Sau khi chạy xong cả 3 seed, mới chấm test:

  ```bash
  yolo detect val model=.../b0_s0/weights/best.pt data=.../data.yaml split=test imgsz=960 name=b0_s0_test
  ```

- Lưu AP từng lớp và confusion matrix.
- Test chỉ có 41 lớp (NHAT_KY, F3). Báo cáo phải ghi rõ mAP tính trên 41 lớp.
- Giao `best.pt` của B0 cho Anh Đức (dùng ở Thí nghiệm 6 và để kiểm tra script đánh giá B1).

## Giai đoạn 2

### B2a: thêm đầu dò P2 cho biển nhỏ

```bash
yolo detect train data=.../data.yaml model=yolov8s-p2.yaml epochs=E ... name=b2a_s0
```

Ultralytics 8.4.191 tự hiểu `yolov8s-p2.yaml` là file `yolov8-p2.yaml` với cỡ s (đã thử).

So với B0 ở các điểm sau:
- mAP chung.
- Recall theo cỡ box: cạnh dài < 32 px, 32–64 px, > 64 px. Ultralytics không tính sẵn, nên viết script nhỏ hoặc nhờ Anh Đức dùng chung script đánh giá.
- Số tham số, GFLOPs, thời gian mỗi epoch, bộ nhớ GPU.

### YOLO 1 lớp cho B1

Cùng lệnh như B0 nhưng thêm `single_cls=True`: mọi box thành một lớp "biển báo".
- Chạy 1 seed. Còn quota thì chạy 3 seed.
- Giao `best.pt` cho Anh Đức.

## Giai đoạn 3

- **Phân tích confusion matrix của B0 (Thí nghiệm 3).** Tập trung các nhóm lớp dễ nhầm ở mục C6 của sổ: nhóm cấm rẽ và cấm quay đầu, các cặp trái/phải.
- **Chấm B0 trên bộ ảnh ngoài phân phối** của Anh Đức (Thí nghiệm 6).

## Giải thích kiến trúc YOLOv8 (phần lý thuyết, làm rải rác)

In cấu trúc mạng bằng `model.info(detailed=True)` hoặc đọc `ultralytics/cfg/models/v8/yolov8.yaml`. Lập bảng từng khối: đầu vào, đầu ra, stride, số tham số. Cần giải thích:
- **Backbone, neck, head.**
  - Các khối trong backbone: Conv, C2f, SPPF.
  - Neck kiểu FPN/PAN: Upsample và Concat.
  - Head `Detect` dự đoán ở 3 tỉ lệ: stride 8, 16, 32. Bản P2 thêm stride 4.
- **Từ YOLO v1–v3 (trong giáo trình) đến v8:** lưới, anchor (v2), dự đoán nhiều tỉ lệ (v3), không dùng anchor (v8), head tách riêng phần phân loại và phần định vị.
- **Hàm mất mát:** box loss (CIoU), cls loss (BCE), DFL. Đây chính là 3 cột trong log train.
- **NMS** làm gì và vì sao cần.

## Phần báo cáo của bạn

1. **Phương pháp phát hiện:** kiến trúc YOLOv8, bảng các khối, cách train từ đầu, cách chọn lr và E.
2. **Kết quả:** B0, B2a (kèm 3 seed), so sánh theo cỡ box, confusion matrix.
3. **Câu hỏi và trả lời** (mỗi câu kèm số liệu):
   - YOLO là mạng một giai đoạn: khác Faster R-CNN (hai giai đoạn) ở đâu?
   - Vì sao dự đoán ở nhiều tỉ lệ? Biển nhỏ nhất (cạnh dài khoảng 12–14 px, NHAT_KY mục B3) chiếm bao nhiêu ô ở stride 8 và ở stride 4?
   - Thêm P2 có giúp biển nhỏ không, đổi lại tốn thêm bao nhiêu tính toán?
   - Train từ đầu trên 2138 ảnh: có overfitting không? Đọc đường cong train loss và val loss.
   - Vì sao lr của lần train từ đầu khác lr Ultralytics tự chọn khi tinh chỉnh? (giáo trình 4.5, 4.10)
   - Ba thành phần loss trong log là gì?
