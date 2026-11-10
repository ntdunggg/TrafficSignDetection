# Kế hoạch chung của nhóm

Đề tài môn Học sâu (IT4653, HUST): nhận diện biển báo giao thông Việt Nam. Cô chấm nặng nhất hai phần:
1. **Tự xây dựng mạng.**
2. **Trả lời được các câu hỏi đi kèm trong quá trình làm.** Mỗi lựa chọn thiết kế phải giải thích được bằng kiến thức của môn: kích thước đầu ra, số tham số, BatchNorm, overfitting, optimizer...

Mọi quyết định, bằng chứng và lịch sử thay đổi nằm trong [`NHAT_KY.md`](../NHAT_KY.md), là sổ chung của nhóm, Dũng giữ. Ba bạn còn lại ghi chi tiết vào sổ riêng trong [`nhat_ky/`](../nhat_ky/), Dũng chép phần quan trọng sang sổ chung. Cách ghi và luật cho AI agent ở [`AGENTS.md`](../AGENTS.md). Kế hoạch đầy đủ ở mục E, phần "Kế hoạch chính: xây dựng mạng". Các file trong thư mục này chỉ tách kế hoạch đó thành việc cụ thể cho từng người.

| File | Người | Phần việc |
|---|---|---|
| [`nguoi_1_dung_du_lieu_va_A0-A2.md`](nguoi_1_dung_du_lieu_va_A0-A2.md) | Người 1 (Dũng) | Dữ liệu, bộ ảnh cắt, khung code PyTorch, mạng A0–A2, bảng tham số |
| [`nguoi_2_sy_nguyen_yolo.md`](nguoi_2_sy_nguyen_yolo.md) | Người 2 (Sỹ Nguyên) | YOLO train từ đầu: chọn lr, B0, B2a, YOLO 1 lớp cho B1, giải thích kiến trúc |
| [`nguoi_3_minh_quan_A3-A7.md`](nguoi_3_minh_quan_A3-A7.md) | Người 3 (Minh Quân) | Mạng A3–A5, A7 (A5 thay cho Thí nghiệm 7), confusion matrix của mạng phân loại |
| [`nguoi_4_anh_duc_danh_gia_va_B1.md`](nguoi_4_anh_duc_danh_gia_va_B1.md) | Người 4 (Anh Đức) | Ảnh ngoài phân phối, Thí nghiệm 5, hệ hai giai đoạn B1, ghép báo cáo |

Phân công chốt ngày 2026-11-10. Chưa có MSSV, hạn nộp và tài khoản GPU. Khi nhóm chốt thì điền vào bảng trên, README và sổ.

## Hai hướng

- **Hướng A (chính):** tự xây CNN phân loại 52 loại biển trên ảnh cắt 64×64, bằng PyTorch. Xây theo bậc A0 → A7, mỗi bậc gắn với một bài giảng.
- **Hướng B (bổ sung):** phát hiện biển trên cả khung hình bằng YOLOv8 **train từ đầu** (B0). Thêm hệ hai giai đoạn: YOLO 1 lớp tìm vùng, mạng của hướng A phân loại vùng (B1). Thêm bản YOLO có đầu dò P2 cho biển nhỏ (B2a).

**Cô không cho dùng mạng đã học trước.** Không dùng file `.pt` có sẵn, không dùng trọng số ImageNet hay COCO. Mọi mạng đều train từ đầu.

## Các giai đoạn và việc bàn giao

| Giai đoạn | Người 1 (Dũng) | Người 2 (Sỹ Nguyên) | Người 3 (Minh Quân) | Người 4 (Anh Đức) |
|---|---|---|---|---|
| 0. Chuẩn bị | Đóng gói dataset, dựng bộ ảnh cắt, khung code chung | Notebook YOLO, đo thời gian epoch, chạy chọn lr | Đọc code chung, chuẩn bị A3–A5 | Bắt đầu thu thập và gán nhãn ảnh ngoài phân phối |
| 1 | A0, A1, A2. Dựng bản ảnh cắt chia ngẫu nhiên cho Thí nghiệm 5 | B0 (3 seed) | A3, A4 (bắt đầu từ A2) | Thí nghiệm 5 trên mạng phân loại |
| 2 | Bảng tham số cho mọi mạng A | B2a. YOLO 1 lớp cho B1 | A5, A7 | Viết script đánh giá B1, kiểm tra bằng B0 |
| 3 | Viết chương dữ liệu | Confusion matrix B0, chấm ảnh ngoài phân phối | Chấm test các mạng đã chốt, confusion matrix | Ghép B1, chấm Thí nghiệm 6 |
| 4 | Báo cáo phần mình | Báo cáo phần mình | Báo cáo phần mình | Ghép toàn bộ báo cáo |

**Bàn giao** (ai không nhận được đúng hạn thì báo ngay, không tự làm lại):

| Từ | Cho | Thứ gì | Hết giai đoạn |
|---|---|---|---|
| Dũng | Cả nhóm | Dataset trên Kaggle (riêng tư), `classes.txt` | 0 |
| Dũng | Minh Quân, Anh Đức | Bộ ảnh cắt `data/vn_signs_crops/` và file `manifest.csv`, khung code `src/` | 0 |
| Dũng | Minh Quân | Mạng A2 và kết quả của nó | 1 |
| Dũng | Anh Đức | `manifest_random.csv` (chia ngẫu nhiên cho Thí nghiệm 5) | 1 |
| Sỹ Nguyên | Cả nhóm | Mức lr đã chọn cho YOLO, số epoch E | 0 |
| Sỹ Nguyên | Anh Đức | Trọng số B0 và YOLO 1 lớp (`best.pt`) | 2 |
| Minh Quân | Anh Đức | Mạng phân loại tốt nhất (code, trọng số, cách tiền xử lý) | 2 |
| Anh Đức | Sỹ Nguyên, Minh Quân | Bộ ảnh ngoài phân phối đã gán nhãn | 3 |

## Quy tắc chung

1. **Không dùng trọng số đã học trước.** YOLO tạo bằng `model=yolov8s.yaml` (hoặc `yolov8s-p2.yaml`). Mạng PyTorch khởi tạo ngẫu nhiên.
2. **Không lật ngang ảnh.** Nhiều lớp chỉ khác nhau ở chiều trái/phải (NHAT_KY mục D1). YOLO đặt `fliplr=0`. Mạng phân loại không dùng `RandomHorizontalFlip`.
3. **Không sửa `data/vn_signs_yolo`.** Cần bản biến thể thì nhờ Dũng dựng thư mục riêng.
4. **Chọn trên val, test chỉ chấm một lần** cho cấu hình đã chốt. Mỗi cấu hình chạy 3 seed (0, 1, 2), báo cáo trung bình ± độ lệch chuẩn.
5. **Ghi phiên bản tập nhãn** mỗi lần chạy, bằng mã md5 của thư mục `labels` (bỏ qua file `.cache`):

   ```python
   import hashlib, pathlib
   def label_md5(labels_dir):
       h, root = hashlib.md5(), pathlib.Path(labels_dir)
       for p in sorted(root.rglob("*.txt")):
           h.update(p.relative_to(root).as_posix().encode()); h.update(p.read_bytes())
       return h.hexdigest()
   ```

6. **Ghi mỗi lần chạy vào `ket_qua/ket_qua.csv`**, mỗi lần một dòng, gồm các cột:

   `ngay,nguoi,thi_nghiem,ban,seed,md5_nhan,so_tham_so,epoch,thoi_gian_phut,val_chinh,test_chinh,ghi_chu`

   `val_chinh` và `test_chinh` là mAP50-95 với YOLO, macro-F1 với mạng phân loại. Không chấm test thì để trống.
7. **Lưu đủ để chạy lại:** code (commit), file cấu hình, `results.csv` hoặc log theo epoch, trọng số tốt nhất. Trọng số không commit vào git (để trên Kaggle hoặc Drive của nhóm).
8. **Mỗi người giữ một mục "Câu hỏi và trả lời" trong phần báo cáo của mình.** Mỗi câu trả lời kèm số liệu hoặc hình làm bằng chứng. Câu hỏi gợi ý có trong file của từng người. Dạng câu hỏi lấy từ đề thi trong `/Users/tdu/2026.1/học sâu/`.
9. **Riêng tư:** ảnh có người và biển số xe. Không đăng dataset, ảnh cắt hay ảnh tự chụp lên chỗ công khai. Dataset trên Kaggle để chế độ riêng tư.
10. **Ghi chép:** mỗi việc, thay đổi, quyết định ghi vào sổ của mình (`nhat_ky/<tên>.md`, Dũng ghi thẳng `NHAT_KY.md`) theo mẫu trong `AGENTS.md`. Thay đổi kế hoạch thì báo cả nhóm, đánh dấu "Chuyển vào NHAT_KY.md: có" để Dũng chép sang sổ chung. Không sửa sổ của người khác.
11. **Dùng AI agent:** agent phải đọc `AGENTS.md` (Claude Code đọc qua `CLAUDE.md`). Agent hỏi trước khi đổi kế hoạch, không tự dùng trọng số có sẵn hay lật ảnh.

## Môi trường

- YOLO: `pip install ultralytics==8.4.191`. Train trên Kaggle hoặc Colab (GPU), mỗi người dùng tài khoản riêng để có thêm quota.
- Mạng phân loại: PyTorch và torchvision. Ảnh 64×64 nhỏ nên chạy được trên máy cá nhân (Mac M2 dùng `device="mps"`) hoặc Colab.
- Trên MPS, một số phép tính không tất định: cùng seed vẫn có thể ra số hơi khác. Đây là thêm một lý do để chạy 3 seed.
