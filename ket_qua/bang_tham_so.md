# Bảng tham số các mạng hướng A (tính tay)

File này tính tay, cho từng lớp của mỗi mạng: kích thước đầu ra, số tham số, receptive field. Mọi con số đã kiểm lại bằng `python -m src.summary --model <tên>` (ngày 2026-11-19, khớp hết). Riêng receptive field thì `src.summary` không in, chỉ tính tay.

Dùng cho phần "bảng tham số" của báo cáo (`ke_hoach/dung_lam_mot_minh.md`, tuần 3), và để trả lời khi cô hỏi.

## 0. Số liệu đầu vào lấy từ đâu

| Số | Lấy từ | Vì sao |
|---|---|---|
| 3 kênh | Ảnh màu RGB | Màu sắc phân biệt được biển cấm (đỏ), biển hiệu lệnh (xanh), biển nguy hiểm (vàng) |
| 64×64 | `size=64` trong `src/data/dataset.py` | Mọi ảnh cắt được đổi về 64×64 lúc nạp. 78,1% box có cạnh dài ≤ 64 px (NHAT_KY mục E) |
| 52 lớp | `du_lieu/classes.txt` | Số loại biển, cũng là số đầu ra của lớp cuối |
| F, P, S, số kênh của từng lớp | Dòng khai báo trong `src/models/a0_lenet.py`, `a1_vgg.py` | Tham số không ghi thì lấy mặc định của PyTorch: `Conv2d` có S = 1, P = 0; `MaxPool2d(k)` có stride = k |

Không cần dữ liệu train. Kích thước và số tham số chỉ phụ thuộc vào kiến trúc, không phụ thuộc ảnh nào được đưa vào.

## 1. Ba công thức

**Kích thước đầu ra:**

    N_ra = (N_vào + 2P − F) / S + 1

N là cạnh của đầu vào, F là cạnh bộ lọc, P là phần đệm, S là bước trượt. Bộ lọc rộng F trượt trên ảnh rộng N + 2P, mỗi lần dịch S ô, và số chỗ đặt được bộ lọc chính là cạnh của đầu ra. Số kênh ra bằng số bộ lọc. Pooling làm riêng trên từng kênh, nên giữ nguyên số kênh.

**Số tham số:**

    Conv:    (F · F · C_vào + 1) · C_ra
    Linear:  (số_vào + 1) · số_ra

- Mỗi bộ lọc conv phủ F×F điểm trên **mọi** kênh vào, nên có F·F·C_vào trọng số, cộng thêm 1 bias. Lớp có C_ra bộ lọc như vậy.
- Mỗi nơ-ron Linear nối với mọi số vào, cộng thêm 1 bias.
- ReLU, MaxPool, Flatten có **0 tham số**. Chúng là phép tính cố định (cắt số âm, lấy max, duỗi phẳng), không có gì để học.

**Receptive field** (r: một điểm ở bản đồ đặc trưng nhìn thấy vùng r×r của ảnh gốc; j: hai điểm liền nhau trên bản đồ đặc trưng cách nhau j pixel trên ảnh gốc):

    r_ra = r_vào + (k − 1) · j_vào
    j_ra = j_vào · s

Bắt đầu từ ảnh gốc với r = 1, j = 1. Ở đây k là cỡ cửa sổ, s là stride. Padding không làm đổi r và j. ReLU không đổi r và j. Lớp Linear nối với mọi điểm, nên nhìn thấy toàn ảnh.

## 2. A0: kiểu LeNet-5

Code: `src/models/a0_lenet.py`. Cấu trúc: conv 5×5 (6 kênh) → ReLU → max pool 2 → conv 5×5 (16 kênh) → ReLU → max pool 2 → FC 120 → FC 84 → FC 52.

| # | Lớp | Tính kích thước | Đầu ra | Tính tham số | Tham số | r | j |
|---|---|---|---|---|---:|---:|---:|
| 0 | Đầu vào | | 3×64×64 | | | 1 | 1 |
| 1 | Conv 5×5, 3→6, S 1, P 0 | (64 + 0 − 5)/1 + 1 = 60 | 6×60×60 | (5·5·3 + 1)·6 = 76·6 | 456 | 1 + 4·1 = 5 | 1 |
| 2 | ReLU | giữ nguyên | 6×60×60 | | 0 | 5 | 1 |
| 3 | MaxPool 2, S 2 | (60 − 2)/2 + 1 = 30 | 6×30×30 | | 0 | 5 + 1·1 = 6 | 2 |
| 4 | Conv 5×5, 6→16 | (30 − 5)/1 + 1 = 26 | 16×26×26 | (5·5·6 + 1)·16 = 151·16 | 2.416 | 6 + 4·2 = 14 | 2 |
| 5 | ReLU | | 16×26×26 | | 0 | 14 | 2 |
| 6 | MaxPool 2, S 2 | (26 − 2)/2 + 1 = 13 | 16×13×13 | | 0 | 14 + 1·2 = **16** | 4 |
| 7 | Flatten | 16·13·13 | 2704 | | 0 | | |
| 8 | Linear 2704→120 | | 120 | (2704 + 1)·120 = 2705·120 | 324.600 | toàn ảnh | |
| 9 | ReLU | | 120 | | 0 | | |
| 10 | Linear 120→84 | | 84 | (120 + 1)·84 = 121·84 | 10.164 | | |
| 11 | ReLU | | 84 | | 0 | | |
| 12 | Linear 84→52 | | 52 | (84 + 1)·52 = 85·52 | 4.420 | | |
| | **Tổng** | | | 456 + 2.416 + 324.600 + 10.164 + 4.420 | **342.056** | | |

**Nhận xét:**
- **Phần conv chỉ có 2.872 tham số (0,8%).** Riêng lớp Linear 2704→120 đã có 324.600 (94,9%). Lý do: mỗi bộ lọc conv được **dùng chung** ở mọi vị trí trên ảnh (chia sẻ trọng số), nên số tham số không phụ thuộc kích thước ảnh. Lớp FC thì có một trọng số riêng cho từng cặp (số vào, số ra), tức 2704 × 120.
- **Receptive field cuối phần conv là 16×16,** chỉ một phần tư cạnh của ảnh 64×64. Mỗi đặc trưng ở lớp 6 chỉ nhìn được một mảng của tấm biển. Phải đến lớp FC mới ghép thông tin của cả ảnh.
- **A0 khác LeNet-5 gốc** (LeCun và cộng sự, 1998, "Gradient-Based Learning Applied to Document Recognition"):
  - LeNet-5 nhận ảnh xám 32×32 (1 kênh). A0 nhận ảnh màu 64×64 (3 kênh).
  - LeNet-5 dùng tanh. A0 dùng ReLU: không bão hòa ở vùng dương, nên gradient không bị triệt tiêu ở vùng đó.
  - LeNet-5 dùng lớp lấy mẫu kiểu trung bình, có hệ số và bias học được. A0 dùng max pool, không có tham số.
  - Lớp conv thứ hai của LeNet-5 chỉ nối mỗi bản đồ ra với một phần các bản đồ vào, theo bảng nối. A0 nối đầy đủ, nên có (5·5·6 + 1)·16 = 2.416 tham số.
  - Đầu ra của LeNet-5 là 10 đơn vị RBF. A0 dùng Linear ra 52 logit, rồi softmax nằm trong `CrossEntropyLoss`.
  - Kích thước giữ đúng tinh thần LeNet-5: 6 và 16 kênh, các lớp FC 120 và 84.

## 3. A1: kiểu VGG

Code: `src/models/a1_vgg.py`. Cấu trúc: 3 khối [conv 3×3 → ReLU → conv 3×3 → ReLU → max pool 2] với số kênh 32, 64, 128, rồi FC 256 → FC 52. Mọi conv có P = 1, S = 1.

Với conv 3×3, P = 1, S = 1 thì cạnh giữ nguyên: (N + 2 − 3)/1 + 1 = N. Chỉ max pool làm cạnh giảm một nửa. Các dòng ReLU không đổi gì, nên không ghi vào bảng.

| # | Lớp | Đầu ra | Tính tham số | Tham số | r | j |
|---|---|---|---|---:|---:|---:|
| 0 | Đầu vào | 3×64×64 | | | 1 | 1 |
| 1 | Conv 3×3, 3→32 | 32×64×64 | (3·3·3 + 1)·32 = 28·32 | 896 | 1 + 2·1 = 3 | 1 |
| 2 | Conv 3×3, 32→32 | 32×64×64 | (3·3·32 + 1)·32 = 289·32 | 9.248 | 3 + 2·1 = 5 | 1 |
| 3 | MaxPool 2 | 32×32×32 | | 0 | 5 + 1·1 = 6 | 2 |
| 4 | Conv 3×3, 32→64 | 64×32×32 | (3·3·32 + 1)·64 = 289·64 | 18.496 | 6 + 2·2 = 10 | 2 |
| 5 | Conv 3×3, 64→64 | 64×32×32 | (3·3·64 + 1)·64 = 577·64 | 36.928 | 10 + 2·2 = 14 | 2 |
| 6 | MaxPool 2 | 64×16×16 | | 0 | 14 + 1·2 = 16 | 4 |
| 7 | Conv 3×3, 64→128 | 128×16×16 | (3·3·64 + 1)·128 = 577·128 | 73.856 | 16 + 2·4 = 24 | 4 |
| 8 | Conv 3×3, 128→128 | 128×16×16 | (3·3·128 + 1)·128 = 1153·128 | 147.584 | 24 + 2·4 = 32 | 4 |
| 9 | MaxPool 2 | 128×8×8 | | 0 | 32 + 1·4 = **36** | 8 |
| 10 | Flatten | 8192 = 128·8·8 | | 0 | | |
| 11 | Linear 8192→256 | 256 | (8192 + 1)·256 = 8193·256 | 2.097.408 | toàn ảnh | |
| 12 | Linear 256→52 | 52 | (256 + 1)·52 = 257·52 | 13.364 | | |
| | **Tổng** | | 287.008 (6 conv) + 2.097.408 + 13.364 | **2.397.780** | | |

**Nhận xét:**
- **A1 có gấp khoảng 7 lần tham số của A0** (2.397.780 so với 342.056). Lớp Linear 8192→256 vẫn chiếm phần lớn: 2.097.408, tức 87,5%. Sáu lớp conv chỉ có 287.008 (12,0%).
- **Vì sao lớp FC này lớn:** sau 3 lần pool vẫn còn 8×8 vị trí × 128 kênh = 8192 số, và mỗi số nối với cả 256 nơ-ron.
- **Receptive field cuối phần conv là 36×36** (A0 là 16×16), hơn nửa cạnh ảnh. Lý do: sâu hơn (6 conv so với 2), và mỗi lần pool làm j gấp đôi, nên các conv sau có bước nhảy lớn và mở rộng r nhanh hơn.
- **Nếu bỏ flatten và FC 256, thay bằng global average pooling** (A1b, phần làm thêm): mỗi kênh lấy trung bình 8×8 thành 1 số, còn 128 số, rồi Linear 128→52 có (128 + 1)·52 = 6.708 tham số. Tổng còn 287.008 + 6.708 = 293.716, giảm 8,2 lần. Số này mới tính tay, chưa kiểm bằng code.

## 4. Hai conv 3×3 so với một conv 5×5

Câu này cô hay hỏi về VGG. So trên khối đầu của A1 (32 kênh vào, 32 kênh ra, cùng kích thước đầu ra):

| | Receptive field | Tham số | Số lần phi tuyến (ReLU) |
|---|---:|---:|---:|
| Hai conv 3×3, 32→32 | 1 + 2 + 2 = 5 | 2·(3·3·32 + 1)·32 = 2·9.248 = **18.496** | 2 |
| Một conv 5×5, 32→32 | 1 + 4 = 5 | (5·5·32 + 1)·32 = **25.632** | 1 |

- **Receptive field như nhau:** cùng nhìn vùng 5×5.
- **Ít trọng số hơn:** không tính bias thì là 18C² so với 25C², tức ít hơn 28%.
- **Thêm một lần ReLU,** nên hàm biểu diễn được phức tạp hơn.
- **Tổng quát:** ba conv 3×3 có receptive field 7, bằng một conv 7×7, với 27C² so với 49C² trọng số. Đây là lập luận của VGG (Simonyan & Zisserman, 2014).

## 5. Kiểm bằng code

```bash
python -m src.summary --model a0    # tổng 342,056
python -m src.summary --model a1    # tổng 2,397,780
```

Hai lệnh đã chạy ngày 2026-11-19. Kích thước đầu ra và số tham số từng lớp khớp với hai bảng trên.
