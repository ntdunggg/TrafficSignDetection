# Nhận diện biển báo giao thông Việt Nam

Bài tập lớn môn **Học sâu (IT4653)**, Đại học Bách khoa Hà Nội. Nhóm 4 người.

Nhóm **tự xây dựng và train từ đầu** mọi mạng. Không dùng mạng đã học trước: không file `.pt` có sẵn, không trọng số ImageNet hay COCO.

## Nhóm

| Người | Họ tên | MSSV | Phần việc |
|---|---|---|---|
| Người 1 | Dũng | | [Dữ liệu, khung code, mạng A0–A2](ke_hoach/nguoi_1_dung_du_lieu_va_A0-A2.md) |
| Người 2 | Sỹ Nguyên | | [YOLO train từ đầu (hướng B)](ke_hoach/nguoi_2_sy_nguyen_yolo.md) |
| Người 3 | Minh Quân | | [Mạng A3–A5, A7](ke_hoach/nguoi_3_minh_quan_A3-A7.md) |
| Người 4 | Anh Đức | | [Đánh giá, ảnh ngoài phân phối, hệ hai giai đoạn B1, ghép báo cáo](ke_hoach/nguoi_4_anh_duc_danh_gia_va_B1.md) |

## Bài toán và hai hướng làm

- **Hướng A (chính): mạng phân loại tự xây.** CNN viết bằng PyTorch, phân loại 52 loại biển trên ảnh cắt 64×64. Xây theo bậc A0 → A7, mỗi bậc gắn với một bài giảng:
  - A0: kiểu LeNet.
  - A1: kiểu VGG.
  - A2: thêm BatchNorm.
  - A3: dropout và tăng cường dữ liệu.
  - A4: khối residual.
  - A5: khởi tạo, hàm kích hoạt, optimizer, lịch lr.
  - A7: ensemble.
- **Hướng B (bổ sung): phát hiện biển trên cả khung hình.**
  - B0: YOLOv8 tạo từ file `.yaml` và train từ đầu.
  - B1: hệ hai giai đoạn. YOLO 1 lớp tìm vùng có biển, mạng của hướng A phân loại vùng đó.
  - B2a: YOLO có thêm đầu dò P2 cho biển nhỏ.

Kế hoạch chi tiết và lý do của từng lựa chọn nằm ở [`NHAT_KY.md`](NHAT_KY.md), mục E. Việc của từng người nằm trong [`ke_hoach/`](ke_hoach/00_chung.md).

## Dữ liệu

- **Nguồn:** bộ `archive`, định dạng YOLO, 52 lớp, ảnh camera hành trình ở TP.HCM.
- **Đã sửa trên bản copy:**
  - Loại 25 ảnh có nhãn rỗng.
  - Sửa box trùng và box vượt khung.
  - Xoay lại 39 ảnh.
  - Đặt tên lớp tiếng Việt.

  Bản gốc `archive` giữ nguyên.
- **Quy mô sau khi sửa:** 3191 ảnh, 8329 box.
- **Chia theo nhóm frame**, để ảnh liền nhau của cùng một đoạn video không rơi vào hai tập khác nhau. Split gốc của `archive` có 30,7% ảnh test gần trùng ảnh train.

| Tập | Ảnh | Box | Số lớp có mặt |
|---|---:|---:|---:|
| train | 2138 | 5621 | 52 |
| val | 474 | 1078 | 35 |
| test | 579 | 1630 | 41 |

Mọi mAP và macro-F1 trên test chỉ tính trên 41 lớp có mặt (sổ, mục F3).

**Ảnh không có trong git.** Ảnh có người và biển số xe, nên không đăng công khai. Nhóm chia sẻ dataset qua Kaggle ở chế độ riêng tư.

## Cấu trúc thư mục

```
README.md            # file này
AGENTS.md            # luật cho AI agent: đọc gì, cấm gì, ghi chép thế nào
CLAUDE.md            # một dòng trỏ tới AGENTS.md, để Claude Code đọc
NHAT_KY.md           # sổ chung (Dũng giữ): quyết định, bằng chứng, kế hoạch, thí nghiệm, câu hỏi báo cáo
nhat_ky/             # sổ riêng của Sỹ Nguyên, Minh Quân, Anh Đức; phần quan trọng được chép sang NHAT_KY.md
ke_hoach/            # kế hoạch chung và việc của từng người
du_lieu/
  scripts/           # script dựng dataset từ archive, chia tập, kiểm tra, đo, catalog
  splits/            # danh sách ảnh train/val/test, groups.csv (nhóm frame), class_counts.csv
  log/               # log khi dựng và kiểm tra dataset, kết quả xem catalog
src/                 # code PyTorch của hướng A: make_crops, dataset, models/, train, evaluate, summary
ket_qua/             # ket_qua.csv: mỗi lần chạy một dòng
data/                # không có trong git: dataset làm việc, ảnh cắt, kết quả train, bằng chứng ảnh
catalog/             # không có trong git: trang HTML để xem và sửa nhãn
```

## Dựng lại dataset

Các script chỉ đọc `archive`, ghi ra thư mục mới, và dừng nếu thư mục đích đã có:

```bash
python3 -I du_lieu/scripts/build_dataset.py <archive_dir> du_lieu/splits data/vn_signs_yolo
python3 -I du_lieu/scripts/verify_dataset.py <archive_dir> data/vn_signs_yolo
```

Sau đó chạy `apply_catalog.py` để áp các sửa sau đợt xem catalog ngày 2026-11-09 (xoay ảnh, xoá 3 box sai, đổi tên lớp 26). Bước cuối là chuyển box 1193#2 từ lớp 28 sang lớp 30. Lệnh và chi tiết từng bước ghi ở các mục A–C của sổ.

## Quy tắc chính

Đầy đủ ở [`ke_hoach/00_chung.md`](ke_hoach/00_chung.md).

1. **Train từ đầu.** YOLO tạo bằng `model=yolov8s.yaml`, mạng PyTorch khởi tạo ngẫu nhiên.
2. **Không lật ngang ảnh**, vì nhiều lớp chỉ khác nhau ở chiều trái/phải.
3. **Chọn trên val, chấm test một lần.** Mỗi cấu hình chạy 3 seed.
4. **Mọi việc và thay đổi đều ghi lại:** vào sổ riêng trong `nhat_ky/`, phần quan trọng chép sang `NHAT_KY.md`. Mẫu ghi ở `AGENTS.md`.
5. **Dùng AI agent thì agent phải theo [`AGENTS.md`](AGENTS.md).** Claude Code tự đọc qua `CLAUDE.md`. Công cụ khác chưa tự đọc thì bảo agent đọc file này đầu mỗi phiên.

## Nguồn gốc repo

Repo được tạo từ đồ án [thuanvipghe/Traffic-sign-VietNam-recognition](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition) (ĐH Tôn Đức Thắng, 2023–2024). Đồ án đó dùng bộ 46 lớp khác và các model đã train sẵn.

Nhóm không dùng dữ liệu, model hay code của đồ án đó. Toàn bộ nội dung cũ đã xoá khỏi repo ngày 2026-11-10, nhưng vẫn giữ làm bản lưu ở nhánh [`de-tai-cu`](https://github.com/ntdunggg/TrafficSignDetection/tree/de-tai-cu) (commit `7dde2b3`). Xem bằng lệnh `git switch de-tai-cu`.
