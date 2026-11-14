# Người 1 (Dũng): dữ liệu, khung code, mạng A0–A2

Đọc trước:
- [`00_chung.md`](00_chung.md).
- [`NHAT_KY.md`](../NHAT_KY.md): tóm tắt hiện trạng, mục A (cách chia), D1 (không lật ngang), mục E phần "Kế hoạch chính".

Vai trò: người giữ dữ liệu và nền móng code. Hai bạn Minh Quân và Anh Đức chờ phần của bạn để bắt đầu, nên **giai đoạn 0 là việc gấp nhất**.

## Giai đoạn 0: chuẩn bị (gấp)

### 0.1. Đóng gói dataset và chia sẻ

Đóng gói dạng không nén (ảnh JPEG đã nén sẵn, nén thêm chỉ được 0,2%, xem mục D3), bỏ các file `.cache` do Ultralytics tạo:

```bash
cd data && zip -0 -r vn_signs_yolo.zip vn_signs_yolo -x "*.cache"
```

- Tải lên Kaggle bằng **New Dataset**, để **Private**, rồi thêm 3 bạn còn lại làm collaborator.
  - File thực tế đã gói ngày 2026-11-14 là `data/vn_signs_kaggle.zip`, gồm cả `vn_signs_crops/`, `classes.txt`, `crops_info.json` (NHAT_KY, lịch sử 2026-11-14), không chỉ `vn_signs_yolo/` như lệnh trên.
  - ✅ Đã tải lên ngày 2026-11-14: [`dngnguyntrng/vn-sign`](https://www.kaggle.com/datasets/dngnguyntrng/vn-sign), Private. Đường dẫn trong notebook: `/kaggle/input/datasets/dngnguyntrng/vn-sign/` (đã kiểm bằng `find`, ghi vào mục "Chạy trên Kaggle" của `00_chung.md`).
  - Còn lại: thêm 3 bạn làm collaborator; chạy thử `src.train --epochs 1` trên Kaggle; nhắn nhóm.
- Xuất file `classes.txt`: 52 dòng theo đúng thứ tự id trong `data.yaml`. Anh Đức cần file này để gán nhãn bằng LabelImg.
- Tạo `ket_qua/ket_qua.csv` với dòng tiêu đề như trong `00_chung.md`.

**Xong khi:** cả nhóm mở được dataset trên Kaggle, giải nén ra đủ 3191 ảnh, có `classes.txt` và `ket_qua/ket_qua.csv`.

### 0.2. Dựng bộ ảnh cắt cho hướng A

Viết `src/data/make_crops.py`. Script đọc `data/vn_signs_yolo` và ghi ra `data/vn_signs_crops/`. Thư mục `data/` đã được git ignore.

**Cách cắt** (Anh Đức dùng lại đúng hàm này cho B1, nên viết thành hàm riêng `crop_box(img, box)`):
- Cắt hình vuông có tâm là tâm box, cạnh = 1,2 × max(w, h). Như vậy mỗi phía có lề 10%, giống GTSRB (NHAT_KY mục A1). Cắt vuông để biển không bị méo khi đổi kích thước.
- Phần vượt ra ngoài ảnh thì đệm bằng cách lặp lại pixel ở mép.
- Lưu PNG ở **kích thước gốc**, chưa đổi về 64×64. Việc đổi kích thước làm lúc nạp dữ liệu, để sau này thử được 32×32 hoặc 48×48 mà không phải cắt lại.

**File `manifest.csv`**, mỗi vùng cắt một dòng, gồm các cột:

`path,split,class_id,stem,line,group,box_w_px,box_h_px`

- `group` lấy từ `du_lieu/splits/groups.csv`.
- `line` là số thứ tự dòng trong file nhãn, khớp với khóa `stem#line` mà catalog đã dùng.

Không dùng `torchvision.datasets.ImageFolder`. Lý do: val chỉ có 35 lớp và test chỉ có 41 lớp, nên mỗi tập sẽ tự đánh số lớp khác nhau. Dùng `manifest.csv` với `class_id` cố định.

**Kiểm tra (ghi kết quả vào log):**
- Số vùng cắt đúng bằng số box: train 5621, val 1078, test 1630.
- Không `stem` nào nằm ở hai tập.
- Bảng số vùng cắt theo lớp và theo tập. Phải khớp `class_counts.csv`, trừ 1193#2 đã chuyển từ lớp 28 sang lớp 30.
- Tạo một trang ghép 10 ảnh cắt ngẫu nhiên cho mỗi lớp, rồi tự nhìn để chắc chắn ảnh cắt đúng biển. 39 ảnh 1319–1357 đã được xoay lại, nên xem kỹ các lớp 12, 14, 31.

### 0.3. Khung code PyTorch chung

Cả nhóm dùng chung khung này, nên phải viết rõ ràng và chạy được. Cấu trúc đề xuất:

```
src/
  data/make_crops.py     # 0.2
  data/dataset.py        # CropDataset(manifest, split, size=64, augment=None)
  models/                # mỗi mạng một file: a0_lenet.py, a1_vgg.py, a2_vgg_bn.py ...
  train.py               # vòng train: seed, device (cuda/mps/cpu), log mỗi epoch ra CSV,
                         # lưu trọng số có val macro-F1 tốt nhất
  evaluate.py            # accuracy, macro-F1 trên các lớp có mặt, recall từng lớp,
                         # confusion matrix (ảnh và CSV)
  summary.py             # in từng lớp: kích thước đầu ra, số tham số (dùng forward hook)
```

- `train.py` nhận cấu hình qua tham số dòng lệnh, ví dụ `--model a2 --opt adam --lr 1e-3 --epochs 30 --seed 0`, để Minh Quân chạy các biến thể mà không phải sửa code.
- Chuẩn hóa ảnh theo mean và std **tính trên tập train** (giáo trình 4.2). Lưu hai giá trị này vào file để mọi người dùng chung.
- Cố định seed cho `random`, `numpy`, `torch`.

**Xong khi:** Minh Quân và Anh Đức tự chạy được `python -m src.train --model a0 --epochs 1` trên máy của mình.

## Giai đoạn 1: mạng A0, A1, A2

Cấu hình train giữ cố định cho A0–A4, đến khi Minh Quân làm A5:
- Optimizer Adam, lr 1e-3 (mặc định trong slide bài 5), batch 128, 30 epoch, hàm mất mát cross-entropy.
- Chưa tăng cường dữ liệu (để dành cho A3).
- Chọn epoch tốt nhất theo val macro-F1.
- Mỗi mạng chạy 3 seed.

| Bậc | Kiến trúc (đầu vào 3×64×64) | Câu hỏi phải trả lời |
|---|---|---|
| A0 | Kiểu LeNet-5: conv 5×5 6 kênh → ReLU → max pool 2 → conv 5×5 16 kênh → ReLU → max pool 2 → FC 120 → FC 84 → FC 52 | Kích thước đầu ra từng lớp bằng `(N+2P−F)/S+1`, số tham số từng lớp. Khác LeNet-5 gốc ở đâu (ReLU, max pool thay cho tanh, avg pool) |
| A1 | Kiểu VGG: 3 khối [conv 3×3 → ReLU → conv 3×3 → ReLU → max pool 2], số kênh 32, 64, 128 → FC 256 → FC 52. Mọi conv dùng padding 1 | Vì sao chồng hai lớp 3×3 thay cho một lớp 5×5 (cùng receptive field, ít tham số hơn, nhiều lớp phi tuyến hơn). Phần lớn tham số nằm ở lớp FC đầu tiên |
| A1b (làm thêm) | Như A1 nhưng thay flatten và FC 256 bằng global average pooling → FC 52 | Bỏ FC lớn thì giảm bao nhiêu tham số và mất bao nhiêu độ chính xác (ý tưởng của GoogLeNet, ResNet) |
| A2 | A1 thêm BatchNorm sau mỗi conv, trước ReLU. Conv đặt `bias=False` | BN có bao nhiêu tham số học được (γ, β) và bao nhiêu thống kê không học (running mean, var). Lúc train và lúc suy diễn khác nhau thế nào. Có hội tụ nhanh hơn không (so đường cong với A1) |

**Đáp số để tự kiểm tra** (bạn tự tính lại tay rồi so):
- A0: kích thước không gian 64 → 60 → 30 → 26 → 13. Sau flatten còn 16×13×13 = 2704. Tổng **342.056** tham số; riêng FC 2704→120 đã chiếm 324.600.
- A1: kích thước không gian 64 → 32 → 16 → 8. Sau flatten còn 128×8×8 = 8192. Các conv có 287.008 tham số, FC 8192→256 có 2.097.408, FC 256→52 có 13.364. Tổng **2.397.780**.

**Receptive field:** tính bằng `r_ra = r_vào + (k − 1) × j_vào` và `j_ra = j_vào × s`, bắt đầu từ r = 1, j = 1. Ghi giá trị ở lớp cuối của mỗi mạng.

Bàn giao cho Minh Quân: code A2 và kết quả 3 seed, vì Minh Quân bắt đầu A3 từ A2.

### Bản ảnh cắt chia ngẫu nhiên cho Thí nghiệm 5 (giao Anh Đức)

Viết `manifest_random.csv`, cùng cột với `manifest.csv` nhưng chia theo split gốc của archive:
- Test: các ảnh trong `/Users/tdu/Downloads/archive/split_dataset/test_files.txt`.
- Train và val: các ảnh trong `train_files.txt`. Lấy ngẫu nhiên 15% số ảnh làm val (seed 0, chia theo ảnh). Cố ý giữ kiểu chia ngẫu nhiên này để thấy rò rỉ.
- Bỏ 25 ảnh có nhãn rỗng (danh sách ở `du_lieu/splits/excluded_empty.txt`).

Kiểm tra: tỉ lệ vùng cắt ở test có ảnh gốc liền số với một ảnh train. Split gốc có 612/639 ảnh test như vậy (NHAT_KY mục A1).

## Giai đoạn 2–3

- Lập **bảng tham số cho mọi mạng của hướng A**, kể cả mạng của Minh Quân: kích thước đầu ra, số tham số, receptive field từng lớp. Tính tay và kiểm bằng `src/summary.py`. Đây là dạng câu hỏi hay gặp trong đề thi.
- Nếu nhóm còn thời gian cho Thí nghiệm 1, 2, 4 thì dựng các bản dataset biến thể. Cách làm và bằng chứng có sẵn ở mục E của sổ.
- Giữ sổ `NHAT_KY.md`: mỗi thay đổi kế hoạch hoặc kết quả quan trọng ghi một dòng vào lịch sử. Định kỳ (ví dụ mỗi tuần) đọc các mục "Chuyển vào NHAT_KY.md: có" trong `nhat_ky/` và chép sang, ghi nguồn.

## Phần báo cáo của bạn

1. **Chương dữ liệu:** nguồn dữ liệu, các lỗi tìm được và cách sửa (mục A–D của sổ), chia theo nhóm frame và cơ sở tài liệu (A1, A1b), câu H1 về tập nhãn.
2. **A0–A2:** kiến trúc, bảng tham số, đường cong loss và macro-F1 của train và val, so sánh 3 mạng.
3. **Câu hỏi và trả lời** (mỗi câu kèm số liệu):
   - Vì sao phải chia theo nhóm frame mà không chia ngẫu nhiên từng ảnh?
   - Vì sao không lật ngang ảnh?
   - A0 và A1 có bao nhiêu tham số, lớp nào chiếm nhiều nhất, vì sao?
   - Hai lớp conv 3×3 so với một lớp conv 5×5: receptive field, số tham số?
   - BatchNorm có bao nhiêu tham số? A2 có hội tụ nhanh hơn A1 không?
   - Ảnh cắt nhỏ (43,8% box có cạnh dài ≤ 32 px) ảnh hưởng thế nào khi phóng lên 64×64?
