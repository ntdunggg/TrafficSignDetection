# Nhật ký dự án: nhận diện biển báo giao thông Việt Nam

Sổ chung của nhóm, Dũng giữ. Từ 2026-11-19 nhóm chỉ còn Dũng, kế hoạch ở `ke_hoach/dung_lam_mot_minh.md`. Sổ riêng cũ của từng người ở `nhat_ky/` chỉ còn làm lịch sử. Cách ghi ở `AGENTS.md`. Trước 2026-11-10 file này tên là `DATASET_REVIEW.md`. Mục A–D rà soát dataset `archive`, mục E là kế hoạch và thí nghiệm, mục F–H là câu hỏi.

Đây là bản ghi duy nhất cho từng ghi chú, quyết định và thay đổi, dùng để viết báo cáo cùng sản phẩm.

- Mỗi mục gồm: **Ghi chú của bạn**, **Trả lời / bằng chứng**, **Quyết định**, **Thay đổi**, **Trạng thái**.
- Mức độ: 🔴 nặng, 🟠 vừa, 🟡 nhẹ. Trạng thái: ✅ xong, 📝 đã ghi để làm sau, ⏳ chờ bạn.

## Tóm tắt hiện trạng (cập nhật 2026-11-10)

| Thứ | Vị trí | Ghi chú |
|---|---|---|
| Dữ liệu gốc | `/Users/tdu/Downloads/archive` | Giữ nguyên, 6437 file khớp md5 (kiểm lại 2026-11-09) |
| Dataset làm việc | `data/vn_signs_yolo/` | 3191 ảnh, 8329 box, 52 lớp, tên tiếng Việt; git ignore |
| Catalog xem và sửa | `catalog/catalog.html` | 7,2 MB; git ignore. Dựng trước đợt sửa 2026-11-09 nên vẫn hiện ảnh chưa xoay và 3 box đã xoá. Không dựng lại (bạn quyết định 2026-11-09) |
| Kết quả catalog | `du_lieu/log/catalog_ket_qua_2026-11-09.txt` | 683 box, dạng `khóa trạng_thái` (d đúng, k không rõ, s sai) |
| Sao lưu trước đợt sửa | `data/_build/backup_truoc_catalog_2026-11-09/` | Nhãn, ảnh, data.yaml trước khi sửa |
| Script dựng dữ liệu, file chia tập, log | `du_lieu/scripts/`, `du_lieu/splits/`, `du_lieu/log/` | Có trong git, không chứa ảnh. Trước 2026-11-10 nằm ở `data/_build/` |
| Bằng chứng ảnh, `meta.npz`, md5 của `archive`, bản trích bài báo | `data/_build/` | git ignore |
| Kết quả chạy Ultralytics | `data/runs/` | git ignore. `sanity_1epoch`, `sanity_test` (chạy thử, xem mục E) |

| Tập | Ảnh | Tỉ lệ ảnh | Box |
|---|---:|---:|---:|
| train | 2138 | 67,0% | 5621 |
| val | 474 | 14,9% | 1078 |
| test | 579 | 18,1% | 1630 |

## Lịch sử thay đổi

**2026-10-30**
- Kiểm tra `archive`, lập bản rà soát đầu tiên.

**2026-11-02**
- Tôi tự tạo `tools/`, `analysis/`, `splits/` khi bạn mới hỏi. Bạn yêu cầu bỏ, đã chuyển vào Thùng rác (`traffic_sign_tu_tao_20261102`). Từ đây phải nêu thắc mắc trước, chưa làm ngay.
- Thêm `.gitignore` (`data/`, `catalog/`) trước khi tạo dữ liệu dẫn xuất.
- Chia lại train/val/test theo nhóm frame (A1, A2). Loại 25 ảnh nhãn rỗng (A3). Dựng cấu trúc chuẩn Ultralytics (A4). Sửa nhãn trên bản copy (B1, B2). Đặt tên tiếng Việt (C2).
- Tạo catalog (B4, C3, C4, C5). Đo bằng chứng cho D2, D3.
- Chốt C1: chưa gộp dataset, vì không có bộ 46 lớp và không có link bản gán nhãn của editor.

**2026-11-10**
- Viết kế hoạch làm việc cho từng người ở `ke_hoach/` (1 file chung, 4 file riêng). Chốt cách cắt ảnh cho hướng A (vuông, cạnh 1,2 × max(w, h), dùng manifest). Số tham số A0 (342.056) và A1 (2.397.780) ghi trong file của người 1 đã kiểm bằng PyTorch.
- Cô trả lời: không được dùng mạng đã học trước. Bỏ A6 và B2b, mọi mạng train từ đầu. Thêm lần chạy ngắn chọn lr cho YOLO train từ đầu ở giai đoạn 0 (bạn đồng ý): AdamW 0.000179, AdamW 0.001, SGD 0.01, mỗi mức 30 epoch, chọn theo mAP50-95 val. Tên người, hạn nộp, tài khoản GPU: nhóm nói sau.
- Bạn cho biết cô coi trọng xây dựng mạng và các câu hỏi đi kèm. Đọc tài liệu môn ở `/Users/tdu/2026.1/học sâu/`. Viết lại kế hoạch ở đầu mục E: hướng A (tự xây CNN phân loại, bậc A0–A7) làm chính, hướng B (YOLO mốc, hai giai đoạn, đầu P2, train từ đầu) bổ sung. Thí nghiệm 7 chuyển thành A5, Thí nghiệm 5 làm trên mạng phân loại trước, Thí nghiệm 1, 2, 4 thành phần làm thêm. Phân công 4 người theo hướng mới.
- Dọn repo cho rõ đây là đề tài tự xây, tránh nhầm với đồ án cũ (bạn chọn cả 3 phương án đề xuất):
  - Viết lại `README.md`. README cũ là đồ án ĐH Tôn Đức Thắng 2023–2024 (bộ 46 lớp, model train sẵn).
  - Xoá khỏi repo `YOLOV8/` (3 file `.pt`), `Faster R-CNN MobileNetV3/` (`.pth` và code), `Picture/`, `traffic_labels.jpg`, `Traffic_Sign_VN_Colab_Kaggle.ipynb`. Lý do: cô không cho dùng mạng đã học trước, giữ trọng số và notebook chạy chúng dễ gây hiểu nhầm. Giữ làm bản lưu ở nhánh `de-tai-cu` (commit `7dde2b3`, đẩy lên GitHub). Không viết lại lịch sử git.
  - Chuyển script, file chia tập và log từ `data/_build/` (git ignore) sang `du_lieu/scripts/`, `du_lieu/splits/`, `du_lieu/log/` để có trong git. Lý do: bạn cùng nhóm cần `groups.csv` và cần dựng lại được dataset. Ảnh bằng chứng, `meta.npz`, md5 của `archive`, bản trích bài báo và bản sao lưu vẫn ở `data/_build/`. Các script nhận đường dẫn qua tham số, nên chuyển chỗ không phải sửa code.
  - Đổi tên sổ `DATASET_REVIEW.md` thành `NHAT_KY.md`, vì sổ giờ ghi cả kế hoạch, thí nghiệm và câu hỏi báo cáo. Sửa mọi đường dẫn trong sổ và `ke_hoach/`.
- Chốt phân công (bạn đồng ý): Người 1 Dũng, Người 2 Sỹ Nguyên, Người 3 Minh Quân, Người 4 Anh Đức. Lý do xếp: Dũng đã nắm dữ liệu, cách chia và sổ, nên làm giai đoạn 0 (việc gấp nhất) nhanh nhất. Ba vai còn lại xếp theo thứ tự tên bạn đưa, vì chưa biết thế mạnh từng người; nếu cần đổi thì vai Người 3 nên cho người chắc lý thuyết môn nhất, vai Người 2 cho người có nhiều quota GPU. Đổi tên 4 file kế hoạch theo tên người (ví dụ `nguoi_1_dung_du_lieu_va_A0-A2.md`), trong file gọi bằng tên thay cho số. Điền tên vào README. MSSV, hạn nộp, tài khoản GPU: chưa có.
- Thêm luật cho AI agent và đổi cách ghi chép (bạn chọn cách a). Lý do: cả nhóm dùng AI agent, nhưng luật "hỏi trước khi làm", "ghi mọi thay đổi" mới chỉ nằm trong trí nhớ agent trên máy Dũng; agent của người khác không thấy và có thể tự thêm trọng số có sẵn hoặc lật ảnh. Nếu 4 agent cùng sửa `NHAT_KY.md` thì dễ xung đột khi gộp nhánh.
  - `AGENTS.md`: đọc gì đầu phiên, việc cấm, cách làm, mẫu ghi chép, quy tắc git. `CLAUDE.md` chỉ có dòng `@AGENTS.md` để Claude Code đọc; Codex, Cursor, Copilot đọc thẳng `AGENTS.md`.
  - Sổ riêng `nhat_ky/sy_nguyen.md`, `nhat_ky/minh_quan.md`, `nhat_ky/anh_duc.md`. Mỗi người chỉ sửa sổ của mình, mục mới ở trên. Mục ghi "Chuyển vào NHAT_KY.md: có" thì Dũng chép sang sổ này, ghi nguồn. Dũng ghi thẳng vào sổ này.
  - Sửa quy tắc 10, thêm quy tắc 11 trong `ke_hoach/00_chung.md`; sửa chỗ "gửi Dũng ghi vào sổ" trong file của Sỹ Nguyên và Anh Đức.

**2026-11-14** (giai đoạn 0 của Dũng, nhánh `dung/giai-doan-0`)
- **Bộ ảnh cắt (0.2):** `src/data/make_crops.py` cắt mỗi box thành PNG ở `data/vn_signs_crops/` (git ignore) kèm `manifest.csv`. Cắt theo hàm `crop_box`: hình vuông, tâm là tâm box, cạnh = round(1,2 × cạnh dài box), đệm bằng pixel mép. `line` đếm từ 0, khớp khóa `stem#line` của catalog (1193#2 là dòng thứ ba). Mất 40 giây.
- **Kiểm tra** (`src/data/check_crops.py`, log ở `du_lieu/log/check_crops_2026-11-14.txt`): đạt cả 4 mục.
  - Số ảnh cắt: train 5621, val 1078, test 1630.
  - 3191 stem và 169 nhóm frame, không cái nào nằm ở hai tập. `groups.csv` có 171 nhóm; nhóm 113 và 160 chỉ gồm ảnh nhãn rỗng đã loại.
  - Số theo lớp khớp `class_counts.csv` sau 5 sửa nhãn đã ghi: lớp 14 −1 (box trùng 1965) và lớp 34 −1 (box trùng 2095) ở B1, lớp 16 −3 (catalog), 1193#2 chuyển 28 → 30. Lần chạy đầu báo lệch 3 lớp vì script chưa tính các sửa này; đã đối chiếu `build_log.txt`, `apply_catalog_log.txt` rồi sửa script.
  - Ảnh vuông, cạnh đúng. Cạnh ảnh cắt: nhỏ nhất 8 px, trung vị 43, p75 70, lớn nhất 570; 71,4% có cạnh ≤ 64 px (sẽ bị phóng to khi đưa về 64×64).
  - Đã xem ảnh ghép 10 ảnh/lớp của cả 52 lớp: mỗi hàng đúng một loại biển, các ảnh đã xoay (lớp 14 ảnh 1329, lớp 31 ảnh 1324–1328) đứng thẳng. Ảnh ghép ở `data/vn_signs_crops/_xem/`.
- **Mean/std của train** ở 64×64 (giáo trình 4.2), lưu `du_lieu/crops_info.json` cùng md5 nhãn `80a393b981121f57e7080674c2750722`: mean (0,468; 0,437; 0,462), std (0,207; 0,199; 0,208).
- **Khung code (0.3):** `src/data/dataset.py` (đọc manifest, đổi về 64×64 một lần, giữ trong RAM), `src/data/augment.py` (chỗ để Minh Quân thêm tăng cường, chỉ có `none`), `src/models/` (A0), `src/train.py`, `src/evaluate.py`, `src/summary.py`, `src/utils.py`.
  - `train.py`: chọn optimizer (adam, adamw, sgd), lịch lr (none, step, cosine, linear), tăng cường, seed; ghi `config.json`, `log.csv`, `best.pt` theo val macro-F1; có `--nguoi` thì thêm một dòng vào `ket_qua/ket_qua.csv`.
  - `evaluate.py`: accuracy, macro-F1 trên các lớp có mặt, recall và F1 từng lớp, confusion matrix (CSV, PNG). Chấm test phải thêm `--cham-test`.
- **Chạy thử A0** trên M2 (MPS): `summary.py` ra đúng 342.056 tham số, khớp tính tay. 1 epoch khoảng 1–2 giây. Cùng seed chạy 2 lần trên MPS ra cùng val macro-F1 0,2540 (35 lớp); trên CPU ra 0,2506. Đây chỉ là chạy thử, không ghi vào `ket_qua.csv`.
- **Gói cho Kaggle:** `data/vn_signs_kaggle.zip` (822 MB, không nén thêm) gồm `vn_signs_yolo/` (bỏ `.cache`), `vn_signs_crops/` (bỏ `_xem/`), `classes.txt`, `crops_info.json`. Đã kiểm số file: 3191 ảnh, 3191 nhãn, 8329 PNG. Dũng đã tải lên Kaggle cùng ngày (Private).
- **Cách chạy trên Kaggle** ghi ở `ke_hoach/00_chung.md`, mục "Chạy trên Kaggle": YOLO trỏ `data=` vào `vn_signs_yolo/data.yaml` trong input; mạng phân loại trỏ `--manifest`, `--info` vào input và `--out` vào `/kaggle/working`. Trước đó chỉ kế hoạch của Sỹ Nguyên nhắc Kaggle (đường dẫn để trống `.../data.yaml`), chưa chỗ nào hướng dẫn chạy mạng phân loại trên Kaggle. Dataset đã tải lên: [`dngnguyntrng/vn-sign`](https://www.kaggle.com/datasets/dngnguyntrng/vn-sign), Private; đường dẫn trong notebook là `/kaggle/input/datasets/dngnguyntrng/vn-sign/` (đã kiểm bằng `find` trong notebook thật: thấy `crops_info.json`, `vn_signs_crops/manifest.csv`, `vn_signs_yolo/data.yaml`). Đường dẫn đoán ban đầu `/kaggle/input/vn-sign/` sai: Kaggle thêm tầng `datasets/<chủ>/`.
- `du_lieu/classes.txt`: 52 dòng dạng `mã tên`, dòng thứ i là lớp id i, cho LabelImg của Anh Đức. `ket_qua/ket_qua.csv`: chỉ có dòng tiêu đề.
- Sửa `.gitignore`: `data/` thành `/data/` vì mẫu cũ bỏ qua cả `src/data/`; thêm `*.pt`, `*.pth`, `*.ckpt`, `runs/`, `__pycache__/`.

**2026-11-19** (đổi kế hoạch, nhánh `dung/ke-hoach-mot-minh`)
- **Nhóm chỉ còn Dũng.** Sỹ Nguyên, Minh Quân, Anh Đức không làm nữa. Hạn nộp còn khoảng 2–3 tuần, trong khoảng 03/12 đến 10/12/2026.
- Viết kế hoạch mới [`ke_hoach/dung_lam_mot_minh.md`](ke_hoach/dung_lam_mot_minh.md), thay cho phân công 4 người. Lịch 3 tuần: tuần 1 xây mạng A0–A4, tuần 2 huấn luyện và đánh giá, tuần 3 viết báo cáo và dự phòng.
- **Giữ:** A0–A4, A5 rút gọn (optimizer, lịch lr), A7, Thí nghiệm 5, Thí nghiệm 3.
- **Làm thêm:**
  - B0, 1 seed, chỉ làm nếu đo 1 epoch trên Kaggle thấy kịp;
  - phần khởi tạo và hàm kích hoạt của A5.
- **Bỏ:** B1, B2a, chọn lr cho YOLO, Thí nghiệm 6, Thí nghiệm 1, 2, 4.
- **Lý do cắt:**
  - Cô chấm nặng phần tự xây mạng, mà YOLO tạo từ file `.yaml` có sẵn.
  - YOLO tốn 551 giây mỗi epoch trên M2 (mục E, "Chạy thử 1 epoch"), trong khi A0 chỉ 1–2 giây.
- **H2, H3** chuyển sang trả lời bằng lý thuyết. H3 có thêm một phép thử rẻ, đề xuất mới: đưa vùng nền không chứa biển qua mạng, đếm số vùng bị gán lớp với độ tin cậy cao.
- **Quyết định của Dũng:** làm theo phạm vi rút gọn này, không hỏi lại cô.
- File `ke_hoach/nguoi_2..4` và `nhat_ky/` giữ lại làm lịch sử, không xoá. Kế hoạch mới trỏ tới các file đó để lấy mô tả chi tiết của A3–A5, A7 và Thí nghiệm 5.
- PR #1 và #3 (giai đoạn 0) đã gộp vào `main` ngày 2026-11-14.
- **A0, 3 seed** (nhánh `dung/a0-a2`, bước 1.1 của kế hoạch mới):
  - Cấu hình: Adam lr 1e-3, batch 128, 30 epoch, không tăng cường, MPS.
  - `src.summary` ra 342.056 tham số. FC 2704→120 chiếm 324.600, tức 94,9%.
  - Mỗi lần chạy mất 0,5 phút. Ba dòng đã ghi vào `ket_qua.csv`.

  | Seed | Epoch tốt nhất | Val macro-F1 (35 lớp) | Val acc | Train acc ở epoch đó |
  |---|---:|---:|---:|---:|
  | 0 | 25 | 0,6400 | 0,8024 | 0,9986 |
  | 1 | 30 | 0,5781 | 0,7699 | 0,9984 |
  | 2 | 29 | 0,6079 | 0,7690 | 0,9988 |
  | **TB ± ĐLC** | | **0,6087 ± 0,0310** | **0,7805 ± 0,0190** | |

  - **Overfitting rõ:**
    - Train acc khoảng 0,999, val acc khoảng 0,78.
    - Val loss nhỏ nhất ở epoch 6–7 (1,22–1,68), rồi tăng tới 1,66–2,42 ở epoch 30.
    - Dùng làm số liệu cho A3.
  - **Phát hiện:** val loss tăng từ epoch 7, nhưng val macro-F1 vẫn tăng tới epoch 25–30.
    - Cách hiểu: mạng tự tin quá mức trên các mẫu sai, nên loss tăng, trong khi số mẫu đúng vẫn tăng.
    - Đây mới là cách hiểu, chưa kiểm.
    - Hệ quả: chọn epoch theo val loss hay theo val macro-F1 sẽ cho mạng khác nhau. Kế hoạch chọn theo macro-F1.
  - Độ lệch chuẩn giữa các seed là 0,031. Chênh lệch giữa các mạng nhỏ hơn mức này thì coi là chưa phân biệt được.
  - Chưa tính tay bảng kích thước và tham số. Receptive field cũng chưa tính, Dũng tự làm.
- **Bảng tham số tính tay A0, A1** ([`ket_qua/bang_tham_so.md`](ket_qua/bang_tham_so.md), Dũng nhờ Claude viết):
  - Kích thước đầu ra, số tham số, receptive field từng lớp. Kích thước và số tham số khớp với `src.summary`.
  - Receptive field cuối phần conv: A0 là 16, A1 là 36.
  - So hai conv 3×3 với một conv 5×5 (32→32): cùng receptive field 5, số tham số 18.496 so với 25.632.
  - A1b (GAP thay cho FC 256) tính tay ra 293.716 tham số, chưa kiểm bằng code.
- **A1 (kiểu VGG), 3 seed:**
  - Code `src/models/a1_vgg.py`, cùng cấu hình nền với A0.
  - 2.397.780 tham số, khớp đáp số trong kế hoạch.
  - 5 giây mỗi epoch, 2,5 phút mỗi lần chạy trên MPS.

  | Seed | Epoch tốt nhất | Val macro-F1 (35 lớp) | Val acc | Train acc |
  |---|---:|---:|---:|---:|
  | 0 | 20 | 0,5975 | 0,7811 | 0,9963 |
  | 1 | 26 | 0,5926 | 0,7774 | 0,9975 |
  | 2 | 24 | 0,5496 | 0,7551 | 0,9984 |
  | **TB ± ĐLC** | | **0,5799 ± 0,0263** | **0,7712 ± 0,0140** | |

  - **A1 không tốt hơn A0:**
    - Macro-F1 thấp hơn 0,029. Chênh lệch này nhỏ hơn độ lệch chuẩn giữa các seed (0,026–0,031), nên chưa phân biệt được hai mạng.
    - A1 có gấp 7 lần tham số và chạy chậm gấp 5 lần.
  - **Overfitting nặng hơn:**
    - Khoảng cách train acc − val acc: A1 là 0,226, A0 là 0,218.
    - Val loss nhỏ nhất của A1 ở epoch 2–4 với seed 1, 2 (epoch 20 với seed 0). Ở epoch 30 val loss là 2,13–2,93, của A0 là 1,66–2,42.
    - Hình so sánh: `data/runs_a/hinh/a0_a1.png` (git ignore). Vẽ lại bằng `python -m src.so_sanh data/runs_a data/runs_a/hinh/a0_a1.png a0 a1`.
  - **Cách hiểu (chưa kiểm):** 5621 ảnh train là ít so với 2,4 triệu tham số. Nếu không có điều chuẩn, mạng sâu hơn chỉ học thuộc train nhanh hơn.
    - Đúng với điều giáo trình 4.7 nói về overfitting khi mô hình lớn mà dữ liệu ít.
    - A2 (BatchNorm) và A3 (dropout, tăng cường dữ liệu) sẽ kiểm tra cách hiểu này.
  - **Câu báo cáo:** "Mạng sâu hơn, nhiều tham số hơn có luôn tốt hơn không?" Ở đây là không, khi chưa có điều chuẩn.

**2026-11-20** (nhánh `dung/a0-a2`)
- **A2 (A1 + BatchNorm), 3 seed:**
  - Code `src/models/a2_vgg_bn.py`: BN sau mỗi conv, trước ReLU; conv `bias=False`; FC giữ nguyên. Chỉ đổi một yếu tố so với A1.
  - 2.398.228 tham số, hơn A1 448: bỏ 448 bias, thêm 896 γ và β. Có 902 buffer: 896 running mean và var, 6 bộ đếm.
  - Tính tay ở `ket_qua/bang_tham_so.md` mục 4, khớp `src.summary`. Mục đó có cả phần BN lúc train và lúc suy diễn khác nhau thế nào.

  | Seed | Epoch tốt nhất | Val macro-F1 (35 lớp) | Val acc | Train acc |
  |---|---:|---:|---:|---:|
  | 0 | 20 | 0,5932 | 0,7672 | 0,9988 |
  | 1 | 24 | 0,5998 | 0,7662 | 0,9995 |
  | 2 | 19 | 0,6202 | 0,7820 | 0,9980 |
  | **TB ± ĐLC** | | **0,6044 ± 0,0141** | **0,7718 ± 0,0088** | |

  - **So với A1** (0,5799 ± 0,0263):
    - A2 cao hơn 0,0245. Mức này xấp xỉ độ lệch chuẩn của A1, nên chưa đủ để nói A2 tốt hơn hẳn.
    - Độ lệch chuẩn giảm gần một nửa (0,0141 so với 0,0263): kết quả ổn định hơn giữa các seed.
    - Ngang A0 (0,6087 ± 0,0310).
  - **Hội tụ nhanh hơn ở các epoch đầu** (trung bình 3 seed):

    | Epoch | 1 | 2 | 3 | 5 |
    |---|---:|---:|---:|---:|
    | Train loss A1 | 2,965 | 1,324 | 0,578 | 0,188 |
    | Train loss A2 | 2,399 | 0,758 | 0,350 | 0,125 |
    | Val F1 A1 | 0,184 | 0,372 | 0,420 | 0,477 |
    | Val F1 A2 | 0,306 | 0,438 | 0,480 | 0,511 |

    Epoch đầu tiên đạt val F1 ≥ 0,5: A2 là 5, 4, 5; A1 là 5, 12, 6. Khớp với điều giáo trình 4.4 nói về BN: chuẩn hóa đầu vào từng lớp giúp train nhanh và ổn định hơn.
  - **Val loss thấp hơn rõ:** ở epoch 30, A2 là 1,31–1,91, A1 là 2,13–2,93. Mạng bớt tự tin quá mức trên mẫu sai.
  - **Overfitting vẫn còn:**
    - Train acc khoảng 0,999, khoảng cách train acc − val acc là 0,227, gần bằng A1.
    - BN không phải cách chữa overfitting chính. Phần này để A3 (dropout, tăng cường dữ liệu).
  - **Thời gian:** 11–15 giây mỗi epoch, A1 là 5 giây, mỗi lần chạy 5–7 phút. Chậm hơn khá nhiều so với số phép tính BN thêm vào. Có thể do BN trên MPS chậm hoặc máy đang bận, chưa kiểm.
  - Hình `data/runs_a/hinh/a0_a1_a2.png` (git ignore).
- **Nền cho A3:** theo kế hoạch, A3 xuất phát từ A2. A3-0 chính là 3 lần chạy A2 ở trên, không chạy lại.
- **A3: chống overfitting (nhánh `dung/a3`, tách từ `dung/a0-a2`).** Dũng yêu cầu ghi lại cả quá trình làm.
  - **Bốn bản,** mỗi bản chỉ đổi một yếu tố so với A3-0:

    | Bản | Chạy bằng | Đổi gì so với A2 |
    |---|---|---|
    | A3-0 | (là A2) | |
    | A3-d | `--model a3d` | Dropout p = 0,5 trước FC cuối |
    | A3-a | `--model a2 --augment a3` | Tăng cường dữ liệu |
    | A3-da | `--model a3d --augment a3` | Cả hai |
  - **Code:**
    - `src/models/a3_dropout.py`: mạng A3Dropout, dùng lại `vgg_bn_block` của A2. Dropout không có tham số, nên số tham số vẫn là 2.398.228 (đã kiểm bằng `src.summary`).
    - `src/data/augment.py` thêm bản `"a3"`:
      - `RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))` và `ColorJitter(brightness=0.3, contrast=0.3)`, đúng như kế hoạch A3.
      - Không lật. Không đổi hue, vì màu mang nghĩa của biển.
      - Góc lộ ra sau khi xoay hoặc dịch tô đen (mặc định của torchvision).
  - **Sửa công cụ:**
    - `src/train.py`: tên lần chạy mặc định thêm `_aug<tên>` khi có tăng cường. Nếu không thì A3-a trùng tên với A2, và `train.py` sẽ từ chối chạy. Tên các lần chạy cũ không đổi.
    - `src/so_sanh.py`: nhận tiền tố tên lần chạy, để so được các bản có tăng cường.
  - **Kiểm tra trước khi train:**
    - Hình `data/runs_a/hinh/a3_tang_cuong.png` (git ignore): 8 ảnh train, mỗi ảnh 5 bản tăng cường.
    - Không ảnh nào bị lật: mũi tên lớp 6 giữ hướng. Biển vẫn nằm trong khung. Giá trị vẫn trong 0..1.
    - Tăng cường tốn 0,30 ms mỗi ảnh, tức khoảng 1,7 giây mỗi epoch.
  - **Lưu ý khi đọc kết quả:**
    - Cột `train_acc` trong `log.csv` đo trên ảnh **đã tăng cường** và với dropout **đang bật**. Vì vậy ở A3-d, A3-a, A3-da nó thấp hơn "độ chính xác thật" trên train.
    - Để so khoảng cách train và val công bằng giữa 4 bản, chấm lại train bằng `python -m src.evaluate --run <...> --split train`. Lệnh này dùng `model.eval()`, không tăng cường.
  - Chạy 9 lần train (3 bản × 3 seed) ngày 2026-11-20, chạy nền trên MPS, xong cả 9. 9 dòng `A3-d`, `A3-a`, `A3-da` trong `ket_qua.csv`, md5 nhãn `80a393b9...` như A0–A2.
  - **Kết quả trên val** (35 lớp, chọn epoch theo val macro-F1, trung bình ± độ lệch chuẩn mẫu qua 3 seed):

    | Bản | Val macro-F1 | Val acc | Epoch tốt nhất (3 seed) | Train acc sạch | Khoảng cách acc sạch | Khoảng cách acc theo log |
    |---|---|---|---|---|---|---|
    | A3-0 (A2) | 0,6044 ± 0,0141 | 0,7718 ± 0,0088 | 20, 24, 19 | 0,9997 | 0,228 | 0,227 |
    | A3-d | 0,6513 ± 0,0075 | 0,7860 ± 0,0301 | 29, 26, 23 | 0,9986 | 0,213 | 0,195 |
    | A3-a | **0,6825 ± 0,0360** | 0,7894 ± 0,0131 | 17, 27, 25 | 0,9945 | 0,205 | 0,186 |
    | A3-da | 0,6429 ± 0,0385 | 0,7934 ± 0,0212 | 29, 29, 28 | 0,9779 | 0,185 | 0,095 |

    - "Train acc sạch": chấm lại checkpoint tốt nhất trên train bằng `python -m src.evaluate --run <...> --split train` (eval, không tăng cường). Kết quả ở `<run>/danh_gia_train/`.
    - "Khoảng cách": train acc trừ val acc, trung bình 3 seed.
    - Số liệu từng seed: `python -m src.so_sanh data/runs_a data/runs_a/hinh/a3.png a2 a3d a2_adam_lr0.001_auga3 a3d_adam_lr0.001_auga3`. Hình `data/runs_a/hinh/a3.png` (git ignore) có val F1, train loss, val loss của 4 bản.
  - **Val loss** (trung bình 3 seed):

    | Bản | Val loss nhỏ nhất | Val loss ở epoch 30 | Train macro-F1 sạch |
    |---|---|---|---|
    | A3-0 (A2) | 1,206 | 1,571 | 0,9998 |
    | A3-d | 1,014 | 1,509 | 0,9991 |
    | A3-a | 1,091 | 1,432 | 0,9911 |
    | A3-da | 0,835 | 1,229 | 0,9125 |

  - **Đọc kết quả:**
    - Cả ba bản đều hơn A3-0 trên val macro-F1. Dropout +0,047, tăng cường +0,078. Mức tăng lớn hơn độ lệch chuẩn của A3-0 (0,014).
    - **Khoảng cách train và val giảm ít hơn số log cho thấy.** Theo log, A3-da giảm từ 0,227 xuống 0,095. Đo sạch thì chỉ từ 0,228 xuống 0,185. Phần chênh là do `train_acc` trong log đo lúc dropout đang bật và ảnh đã tăng cường. Đây là lý do phải chấm lại train ở chế độ eval.
    - Cả 4 bản vẫn gần như thuộc lòng train (train acc sạch ≥ 0,978). Khoảng cách còn khoảng 0,19–0,23, tức overfitting giảm nhưng chưa hết.
    - **A3-da chưa học xong ở 30 epoch.** Epoch tốt nhất là 28–29 ở cả 3 seed. Train loss ở epoch 30 còn 0,25–0,40, các bản khác 0,004–0,125. Train macro-F1 sạch chỉ 0,91. Val loss còn đang giảm (hình `a3.png`). Ghép hai cách chống overfitting làm mạng học chậm hơn. Với 30 epoch cố định, đây là thiếu khớp (underfitting), chứ chưa chứng minh được là ghép cả hai thì tệ hơn.
    - A3-da có val acc cao nhất (0,793) nhưng macro-F1 thấp hơn A3-a. Train macro-F1 (0,91) thấp hơn hẳn train acc (0,98), nên các lớp ít ảnh là phần chưa học xong. Đây là suy luận từ hai con số trên, chưa kiểm bằng recall từng lớp.
    - Val loss: cả 3 bản đều có val loss nhỏ nhất thấp hơn A3-0, và ở epoch 30 tăng lại ít hơn. A3-0 có val loss nhỏ nhất ở epoch 4–6 (2/3 seed), sau đó tăng trong khi val F1 vẫn lên. Đó là dấu hiệu mạng ngày càng tự tin sai trên ảnh val.
    - Thời gian: 12,8–18,8 giây mỗi epoch, không khác rõ giữa các bản. Tăng cường tốn khoảng 1,7 giây mỗi epoch, nhỏ hơn dao động của MPS.
  - **Chọn bản nền cho A4: A3-a (Dũng quyết định 2026-11-20).** A4 dùng tăng cường `a3`, không dropout.
    - **Phân vân:**
      - A3-a có trung bình cao nhất (0,6825). Nhưng nó chỉ hơn A3-d 0,031, nhỏ hơn độ lệch chuẩn của chính nó (0,036).
      - Theo quy tắc chọn đã đặt trước (`ke_hoach/nguoi_3_minh_quan_A3-A7.md`, mục A5), hai bản này coi là hoà, và chọn bản có độ lệch chuẩn nhỏ hơn: A3-d (0,0075).
      - Quy tắc đó viết cho A5. Kế hoạch A4 chỉ ghi "bản tốt nhất của A3".
    - **Lý do chọn A3-a:**
      1. **A3-a hơn đều ở từng seed, không chỉ ở trung bình.** A3-a: 0,6558 / 0,6683 / 0,7234. A3-d: 0,6504 / 0,6442 / 0,6592. Seed kém nhất của A3-a gần bằng seed tốt nhất của A3-d. Ghép từng cặp seed (9 cặp), A3-a thắng 8. Độ lệch chuẩn lớn chủ yếu do seed 2 cao vượt, không phải có seed tệ.
      2. **Tăng cường gắn với dữ liệu, dùng y nguyên cho mọi mạng.** Dropout của A3-d gắn với FC 8192→256 của A2 (87,5% tham số). A4 thay thân mạng và có thể cả đầu phân loại, nên dropout ở đó không còn cùng vai trò.
      3. **Không chọn A3-da:** chưa học xong ở 30 epoch (xem trên), mà A4 vẫn giữ 30 epoch.
    - **Hạn chế phải ghi trong báo cáo:** kiểm định t Welch giữa A3-a và A3-d cho t ≈ 1,47, p ≈ 0,27 (3 seed mỗi bản). Chưa đủ bằng chứng thống kê rằng A3-a hơn A3-d. Chọn A3-a là theo hai lý do trên, không phải theo kiểm định.
- **A4: kết nối tắt (vẫn trên nhánh `dung/a3`, chưa commit A3).** Kế hoạch: `ke_hoach/nguoi_3_minh_quan_A3-A7.md`, mục A4.
  - **Câu cần trả lời:** mạng trơn sâu hơn có tệ hơn cả trên **train** không? Đó là suy biến (degradation), khác overfitting. Kết nối tắt có sửa được không?
  - **Ba mạng** (`src/models/a4_resnet.py`), cùng khung:
    - Stem conv 3×3 3→32. Bốn tầng, số kênh 32, 64, 128, 256. Tầng 2–4 giảm kích thước bằng conv stride 2, không dùng max pool (đúng kế hoạch). Kích thước 64 → 32 → 16 → 8.
    - Đầu phân loại: global average pooling → FC 256→52. Không dùng FC 8192→256 như A2, để tham số nằm ở phần conv, là phần A4 muốn so.

    | Bản | Tên trong code | Phần thân mỗi tầng | Số conv 3×3 | Tham số | Buffer |
    |---|---|---|---:|---:|---:|
    | A4-plain-nông | `a4pn` | 1 conv + BN + ReLU | 5 | 411.540 | 1.029 |
    | A4-plain-sâu | `a4ps` | 2 conv + BN + ReLU | 9 | 1.195.860 | 1.993 |
    | A4-res-sâu | `a4rs` | 1 basic block: 2 conv, cộng nhánh tắt | 9 | 1.239.764 | 2.892 |

    - plain-nông và plain-sâu chỉ khác độ sâu. plain-sâu và res-sâu chỉ khác kết nối tắt.
    - Nhánh tắt: giữ nguyên x ở tầng 1. Ở tầng 2–4 (đổi số kênh và stride) dùng conv 1×1 + BN, đúng kế hoạch (He 2016 gọi là phương án B). Nhánh này thêm 43.904 tham số, tức 3,7% so với plain-sâu. Đây là khác biệt thứ hai, nhỏ, phải ghi khi so.
    - Số tham số đã kiểm bằng `python -m src.summary --model a4pn|a4ps|a4rs`.
  - **Cấu hình huấn luyện:** như A3-a. Adam lr 1e-3, batch 128, 30 epoch, `--augment a3`, không dropout, khởi tạo mặc định của PyTorch.
  - **Cách đo suy biến:** so train loss theo epoch của 3 bản. Cả 3 dùng cùng tăng cường và không có dropout, nên `train_loss` trong log so được với nhau. Cuối cùng chấm lại train ở chế độ eval (`--split train`).
  - **Dự đoán ghi trước khi chạy:** ở độ sâu 9 conv có BN, có thể **chưa thấy suy biến**. Bài ResNet (He và cộng sự 2016, arXiv 1512.03385) thấy suy biến ở mạng trơn sâu hơn nhiều: Hình 1 so 20 với 56 lớp trên CIFAR-10, Bảng 2 so 18 với 34 lớp trên ImageNet. Nếu đúng như dự đoán thì đó cũng là một câu trả lời, nhưng sẽ cần hỏi Dũng có thêm bản sâu hơn không.
  - Chạy thử 1 epoch `a4ps`: 25,4 giây mỗi epoch trên MPS, gấp khoảng 2 lần A2. Tính tay số phép nhân-cộng mỗi ảnh: plain-sâu khoảng 249 triệu, A2 khoảng 157 triệu (`ket_qua/bang_tham_so.md`, mục 4c).
  - Chạy 9 lần train (3 bản × 3 seed), bắt đầu 2026-11-20, xong 2026-11-21, xen kẽ theo seed. 9 dòng `A4-plain-nong`, `A4-plain-sau`, `A4-res-sau` trong `ket_qua.csv`. Mỗi lần: plain-nông 5–7 phút, plain-sâu 9–10 phút, res-sâu 10–12 phút.
  - **Kết quả trên val** (35 lớp, trung bình ± độ lệch chuẩn mẫu qua 3 seed). A3-a viết lại để so:

    | Bản | Tham số | Val macro-F1 | Val acc | Val loss nhỏ nhất | Train acc sạch | Khoảng cách acc sạch |
    |---|---:|---|---|---|---|---|
    | A3-a (A2 + tăng cường) | 2.398.228 | 0,6825 ± 0,0360 | 0,7894 ± 0,0131 | 1,091 | 0,9945 | 0,205 |
    | A4-plain-nông | 411.540 | 0,5175 ± 0,0182 | 0,6837 ± 0,0056 | 1,379 | 0,9539 | 0,270 |
    | A4-plain-sâu | 1.195.860 | 0,6605 ± 0,0244 | 0,8377 ± 0,0146 | 0,610 | 0,9963 | 0,159 |
    | A4-res-sâu | 1.239.764 | 0,6517 ± 0,0139 | 0,8135 ± 0,0112 | 0,798 | 0,9927 | 0,179 |

    - Từng seed (val macro-F1): plain-nông 0,5024 / 0,5378 / 0,5122; plain-sâu 0,6653 / 0,6340 / 0,6821; res-sâu 0,6556 / 0,6362 / 0,6632.
    - "Train acc sạch": chấm lại checkpoint tốt nhất bằng `python -m src.evaluate --split train`.
    - Lệnh: `python -m src.so_sanh data/runs_a data/runs_a/hinh/a4.png a2_adam_lr0.001_auga3 a4pn_adam_lr0.001_auga3 a4ps_adam_lr0.001_auga3 a4rs_adam_lr0.001_auga3`. Hình `data/runs_a/hinh/a4.png` (git ignore).
  - **Train loss trong log theo epoch** (trung bình 3 seed; cùng tăng cường, không dropout, nên so được với nhau):

    | Bản | e1 | e2 | e3 | e5 | e10 | e20 | e30 |
    |---|---|---|---|---|---|---|---|
    | A3-a | 3,130 | 1,568 | 1,034 | 0,547 | 0,224 | 0,094 | 0,063 |
    | A4-plain-nông | 2,511 | 1,740 | 1,425 | 1,072 | 0,635 | 0,305 | 0,174 |
    | A4-plain-sâu | 2,399 | 1,525 | 1,157 | 0,727 | 0,278 | 0,053 | 0,029 |
    | A4-res-sâu | 2,183 | 1,351 | 1,042 | 0,682 | 0,301 | 0,072 | 0,029 |

  - **Trả lời câu hỏi A4:**
    - **Không thấy suy biến ở độ sâu này. Dự đoán ghi trước khi chạy là đúng.** Mạng trơn sâu (9 conv) có train loss **thấp hơn** mạng trơn nông (5 conv) ở cả 30 epoch (trung bình 3 seed, đã kiểm từng epoch): 0,029 so với 0,174 ở e30. Train acc sạch 0,996 so với 0,954. Suy biến là mạng sâu hơn mà train tệ hơn, ở đây ngược lại.
    - Vì không có suy biến nên kết nối tắt không có gì để sửa. Ở e30, train loss của res-sâu và plain-sâu bằng nhau (0,029).
    - Kết nối tắt chỉ giúp **hội tụ nhanh hơn ở vài epoch đầu**: train loss e1–e3 của res-sâu là 2,18 / 1,35 / 1,04, so với 2,40 / 1,53 / 1,16 của plain-sâu. Từ e10 thì hai bản ngang nhau.
    - Trên val, plain-sâu và res-sâu hoà: chênh 0,009, nhỏ hơn độ lệch chuẩn của cả hai.
    - **Plain-nông thiếu khớp:** train acc sạch chỉ 0,954, train loss e30 còn 0,174, val F1 thấp nhất (0,52). Nhưng plain-nông khác plain-sâu ở cả độ sâu, số tham số (411 nghìn so với 1,2 triệu) và receptive field (19 so với 49, `bang_tham_so.md` mục 4c). Không tách được yếu tố nào gây ra.
  - **So với A3-a:**
    - Val macro-F1 của plain-sâu và res-sâu thấp hơn A3-a 0,02–0,03, nhỏ hơn độ lệch chuẩn của A3-a (0,036), nên coi là hoà.
    - Nhưng hai bản A4 sâu có val acc cao hơn (0,84 và 0,81 so với 0,79), val loss nhỏ nhất thấp hơn hẳn (0,61 và 0,80 so với 1,09), khoảng cách train và val nhỏ hơn (0,16–0,18 so với 0,21), mà chỉ dùng một nửa số tham số.
    - Val acc cao hơn mà macro-F1 không cao hơn: có thể các lớp nhiều ảnh được lợi, lớp ít ảnh thì không. Chưa kiểm bằng recall từng lớp.
    - So A4 với A3-a không phải đổi một yếu tố: khác cả thân mạng (stride 2 thay max pool, thêm tầng 256 kênh) và đầu phân loại (global average pooling thay FC 8192→256). Chỉ dùng để chọn bản nền, không dùng để kết luận yếu tố nào tốt.
  - **Hạn chế:** chỉ thử đến 9 conv, nên chưa trả lời được "mạng trơn rất sâu có suy biến không" trên dữ liệu này. Muốn thấy thì cần mạng sâu hơn nhiều (bài ResNet: 20 với 56 lớp). Chưa chạy, chờ Dũng quyết định.
  - **Chọn bản nền cho A5: A4-res-sâu (`a4rs`, tăng cường `a3`), Dũng quyết định 2026-11-21.**
    - Theo quy tắc chọn đặt trước (`ke_hoach/nguoi_3_minh_quan_A3-A7.md`, mục A5): trung bình cao nhất, nhưng nếu chênh lệch nhỏ hơn độ lệch chuẩn thì coi là hoà và chọn bản dao động ít hơn.
    - A3-a, plain-sâu, res-sâu hoà nhau trên val macro-F1. Res-sâu có độ lệch chuẩn nhỏ nhất (0,0139, so với 0,0244 và 0,0360).
    - Thêm vào đó, so với A3-a, res-sâu có val acc cao hơn (0,814 so với 0,789), val loss nhỏ nhất thấp hơn (0,80 so với 1,09) và chỉ dùng một nửa số tham số.
    - Phương án khác đã cân nhắc: plain-sâu có trung bình F1, val acc, val loss nhỉnh hơn res-sâu một chút, nhưng dao động giữa các seed lớn hơn.

**2026-11-21** (nhánh `dung/manifest-random`, tách từ `dung/a3`)
- **Dựng `manifest_random.csv` cho Thí nghiệm 5 (bản R).** Kế hoạch: `ke_hoach/dung_lam_mot_minh.md` mục 1.4, cách làm ở `ke_hoach/nguoi_1_dung_du_lieu_va_A0-A2.md`, mục "Bản ảnh cắt chia ngẫu nhiên cho Thí nghiệm 5".
  - **Script:** `src/data/make_manifest_random.py`. Chạy `python -m src.data.make_manifest_random`.
    - Test là ảnh trong `archive/split_dataset/test_files.txt`.
    - Train và val lấy từ `train_files.txt`: xáo ngẫu nhiên theo ảnh (seed 0), 15% làm val.
    - Chỉ đọc `archive` và `manifest.csv`, không sửa.
  - **Không cắt lại ảnh.** Dùng đúng các ảnh cắt của `data/vn_signs_crops/`, chỉ đổi cột `split`.
    - Nhãn là bản đã sửa, cùng md5 `80a393b9...` với bản G. Bản R và bản G chỉ khác nhau ở cách chia tập.
    - Vì cột `path` giữ nguyên, `manifest_random.csv` nằm cạnh `manifest.csv` trong `data/vn_signs_crops/` (git ignore).
  - **Chuẩn hoá:** mean/std tính lại trên train của bản R, ghi ở `du_lieu/crops_info_random.json` (có trong git), theo quy tắc chuẩn hoá bằng mean/std của train. Train trên bản R thì thêm `--manifest data/vn_signs_crops/manifest_random.csv --info du_lieu/crops_info_random.json`.
  - **Phát hiện:** split gốc của archive đã không chứa ảnh nào trong 25 ảnh nhãn rỗng. `train_files` (2552) + `test_files` (639) = 3191 ảnh, khớp đúng 3191 ảnh của dataset. Vì vậy bước "bỏ 25 ảnh nhãn rỗng" của kế hoạch không bỏ ảnh nào.
  - **Kết quả:**

    | Tập | Ảnh (R) | Ảnh cắt (R) | Số lớp (R) | Ảnh cắt (G, để so) |
    |---|---:|---:|---:|---:|
    | train | 2169 | 5690 | 52 | 5621 |
    | val | 383 | 995 | 50 | 1078 |
    | test | 639 | 1644 | 52 | 1630 |

    - Val của R thiếu 2 lớp, là 44 và 47; test của R có đủ 52 lớp.
    - 41 lớp có ở test của G đều có ở test của R. Khi so R với G trên test, tính macro-F1 trên 41 lớp chung này để cùng tập nhãn (như mô tả Thí nghiệm 5 ở mục E).
  - **Kiểm tra rò rỉ** ("có ảnh train liền số" = ảnh số n có ảnh n − 1 hoặc n + 1 nằm trong train):

    | | R | G |
    |---|---|---|
    | Ảnh test có ảnh train liền số | 569/639 (89,0%) | 23/579 (4,0%) |
    | Ảnh val có ảnh train liền số | 336/383 (87,7%) | 46/474 (9,7%) |
    | Ảnh cắt test có ảnh gốc liền số với một ảnh train | 1481/1644 (90,1%) | |
    | Nhóm frame bị tách ra nhiều tập | 122/169 | 0 |

    - Kế hoạch yêu cầu khớp con số 612/639 ở mục A1. Đo lại test với **toàn bộ** `train_files` cho đúng 612/639. Con số 569 thấp hơn vì 15% ảnh của `train_files` đã chuyển sang val.
    - Bản G vẫn có 4% ảnh test liền số với train: đó là ảnh ở ranh giới giữa hai nhóm frame liền nhau.
  - Đã đọc thử cả 3 tập qua `CropDataset`: 5690 / 995 / 1644 ảnh, tensor 3×64×64.
  - Chưa chạy Thí nghiệm 5. Theo kế hoạch, việc này ở tuần 2, dùng cấu hình đã chốt.

**2026-11-22** (nhánh `dung/b0-thoi-gian`, tách từ `dung/manifest-random`)
- **Chuẩn bị đo thời gian B0 trên Kaggle.** Kế hoạch: `ke_hoach/dung_lam_mot_minh.md`, mục 1.5. Quy tắc quyết định đặt trước: 100 epoch dưới khoảng 10 giờ thì làm B0 ở tuần 2, không thì bỏ.
  - **Script:** `src/yolo/do_thoi_gian_b0.py`, tự chứa, dán vào một ô notebook Kaggle. Script làm các việc sau:
    - Kiểm phiên bản `ultralytics==8.4.191`, có GPU, và md5 nhãn khớp `80a393b9...`.
    - Train 1 epoch với cấu hình của B0 thật.
    - Đo riêng thời gian train và thời gian val + lưu checkpoint của epoch, bằng callback của Ultralytics.
    - Ghi loại GPU, bộ nhớ đỉnh, ước tính giờ cho 100 epoch và quyết định theo quy tắc trên vào `/kaggle/working/b0_thoi_gian.json`.
  - **Cấu hình:**
    - `yolov8s.yaml`, `pretrained=False`, `imgsz=960`, `fliplr=0`.
    - `optimizer=SGD lr0=0.01 momentum=0.937`, ghi cố định như kế hoạch.
    - `batch=16`, là mặc định của Ultralytics. Vì `nbs=64`, gradient cộng dồn qua 4 batch mới cập nhật một lần.
  - **Hai chỗ khác mặc định, có lý do:**
    1. **`close_mosaic=0` trong lần đo.** Mặc định `close_mosaic=10` tắt mosaic ở 10 epoch cuối. Chạy 1 epoch thì mosaic bị tắt luôn, epoch đo được sẽ nhẹ hơn epoch thật (lưu ý 2 của "Chạy thử 1 epoch", mục E). B0 thật vẫn giữ `close_mosaic=10`.
    2. **Không tải `yolo26n.pt` để kiểm tra AMP.**
       - Đọc mã nguồn Ultralytics 8.4.191 (`utils/checks.py`, hàm `check_amp`): trên GPU CUDA, trước khi train nó tải `yolo26n.pt` (trọng số đã học trước) để so kết quả fp16 với fp32. Log ghi "not used for training". Trên M2 (MPS) không gặp vì hàm trả về False ngay.
       - Để giữ đúng quy tắc không tải checkpoint (AGENTS.md mục 2), script thay `check_amp` bằng hàm trả về True. AMP vẫn bật và không tải file `.pt` nào.
       - Khi không có mạng, Ultralytics cũng bỏ qua bước kiểm tra này và giữ `amp=True`.
       - Rủi ro là bỏ mất bước phát hiện GPU lỗi fp16. Nếu loss ra NaN thì chạy lại với `AMP = False`.
       - B0 thật cũng phải làm như vậy.
  - **Chạy thử trên M2** (bản sửa tạm trong scratchpad: `device=mps`, `imgsz=320`, `fraction=0.01`):
    - Chạy hết, callback đo được thời gian.
    - Log xác nhận `optimizer: SGD(lr=0.01, momentum=0.937)`, `fliplr=0`, `close_mosaic=0`, `pretrained=False`.
    - Không có file `.pt` nào được tải về; chỉ có `last.pt`, `best.pt` do chính lần chạy tạo ra.
    - Số đo trên M2 không dùng được cho quyết định.
  - **Sự cố khi chạy thử:** lần chạy thử với `fraction=0.01` đã ghi đè `data/vn_signs_yolo/labels/train.cache` bằng cache của 1% ảnh. File `.cache` không phải nhãn, md5 nhãn bỏ qua nó (đã kiểm: md5 vẫn `80a393b9...`). Ultralytics tự dựng lại cache khi danh sách ảnh khác. Dù vậy, đây là ghi vào thư mục dữ liệu dùng chung, phải tránh: lần sau chạy thử thì chép dữ liệu ra chỗ khác.
  - **Đã đo trên Kaggle (2026-11-22).** Dũng chạy bằng tài khoản khác, chép output về.

    | Thứ | Giá trị |
    |---|---|
    | GPU | Tesla T4 (notebook có 2 card, script dùng 1), 15,6 GB; 4 CPU |
    | Phần mềm | Python 3.13.16, torch 2.11.0+cu130, Ultralytics 8.4.191 |
    | md5 nhãn | `80a393b9...`, khớp |
    | Mạng | YOLOv8s từ `yolov8s.yaml`, 11.155.724 tham số, 28,8 GFLOPs (`nc=80` đổi thành 52) |
    | Train 1 epoch | 168,5 giây (134 lượt × khoảng 1,3 giây, batch 16) |
    | Val + lưu checkpoint 1 epoch | 45,7 giây |
    | **Mỗi epoch** | **214,2 giây** |
    | Cả lần chạy (đọc dữ liệu, 1 epoch, chấm `best.pt` lần cuối) | 391,2 giây |
    | Bộ nhớ GPU đỉnh | 8,16 GB, batch 16 vừa |
    | **Ước tính 100 epoch** | **5,95 giờ** |

  - **Quyết định theo quy tắc đặt trước: làm B0 ở tuần 2.** 5,95 giờ dưới mức 10 giờ, còn dư khoảng 6 giờ so với giới hạn 12 giờ của một phiên.
  - **Kiểm tra quy tắc:** log ghi `optimizer: SGD(lr=0.01, momentum=0.937)`, `fliplr=0`, `pretrained=False`, `close_mosaic=0`. Không tải file `.pt` nào. Hai file được tải là phông chữ để vẽ hình (`Arial.Unicode.ttf`, `Arial.ttf`), không phải trọng số.
  - **Vì sao 5,95 giờ có thể là ước tính hơi cao:**
    - Val ở epoch đầu chậm (2,7 giây mỗi lượt, 15 lượt). Chấm lại `best.pt` cũng trên 474 ảnh đó chỉ mất 5,6 giây. Có thể do mạng chưa học nên ra rất nhiều box, bước NMS tốn thời gian. Đây là suy luận, chưa đo ở epoch sau.
    - Ultralytics cảnh báo đọc ảnh chậm từ `/kaggle/input` (1,9 MB/s) và chỉ dùng 2 worker đọc ảnh (máy có 4 CPU). Chép dữ liệu sang `/kaggle/working` trước khi train có thể nhanh hơn. Chưa thử.
  - **Không ghi vào `ket_qua.csv`:** đây là lần đo thời gian, không phải thí nghiệm. Giống quy ước của `train.py`: chạy thử thì không ghi.
  - **Cho B0 thật ở tuần 2:**
    - Giữ cấu hình trên với `epochs=100`, `close_mosaic=10` (mặc định), 1 seed.
    - Vẫn phải thay `check_amp` để không tải `yolo26n.pt`.
    - Lưu `best.pt` và `results.csv` ra khỏi `/kaggle/working` trước khi phiên kết thúc.
    - Cần 1 phiên khoảng 6 giờ trong quota GPU 30 giờ mỗi tuần của Kaggle.
- **Đổi dataset Kaggle `dngnguyntrng/vn-sign` sang công khai (Dũng quyết định 2026-11-22).**
  - Bối cảnh: tài khoản `dngnguyntrng` chưa xác minh số điện thoại nên không bật được Internet và GPU. Dũng chạy bằng tài khoản khác, và đổi dataset sang công khai để tài khoản đó dùng được.
  - Trái với quy tắc cũ "dataset trên Kaggle để riêng tư" (`ke_hoach/00_chung.md` quy tắc 9), vì ảnh có người và biển số xe. Tôi đã nêu cách khác: giữ riêng tư và thêm tài khoản mới làm collaborator.
  - Lý do của Dũng: ảnh lấy từ một dataset vốn đã công khai, nên để công khai không sao.
  - Dũng yêu cầu đổi luật. Đã sửa quy tắc 9 và đoạn "Chạy trên Kaggle" trong `ke_hoach/00_chung.md`, `README.md`, và `AGENTS.md` mục 2. Vẫn giữ: không đưa ảnh vào git, ảnh tự chụp không đăng công khai.

**2026-11-09**
- Nhận kết quả catalog: đã xem đủ 683/683 box (599 Đúng, 80 Không rõ, 4 Sai). Chi tiết ở mục G.
- Xoay lại 39 ảnh 1319–1357, xoá 3 box sai lớp 16, đổi tên lớp 26 (bản copy). Box giảm từ 8332 còn 8329.
- Chuyển box 1193#2 từ lớp 28 sang lớp 30 (bạn xác nhận). Lớp 28 không còn ở val và test, số lớp không đánh giá được tăng từ 10 lên 11.
- Thêm Thí nghiệm 4 vào kế hoạch (mục E): xoá cả 11 lớp không đánh giá được, so sánh giữ ảnh và bỏ ảnh.
- Thêm mục A1b: cách xáo trộn khi nguồn là video, tên phương pháp, và đo độ ổn định của cách chia với 5 seed (thêm chế độ `seeds` vào `make_split.py`, không thay split đang dùng).
- Ghi 4 câu hỏi cần trả lời trong báo cáo (mục H). Thêm quy tắc ghi phiên bản tập nhãn, Thí nghiệm 5 (ảnh không giống train) và 6 (ảnh ngoài phân phối).
- Dọn sổ trước khi train: C3, C4 chuyển ✅; C5 thành ✅ catalog, 📝 Thí nghiệm 2. Không còn mục nào chờ bạn quyết định, trừ 3 câu ở mục F.
- Catalog chưa dựng lại. Lý do: ảnh 2088 có 3 box, xoá box #1 làm box #2 thành #1. Nếu dựng lại mà vẫn dùng khóa lưu cũ (`vn_signs_catalog_v1`), đánh dấu "Sai" cũ của 2088#1 sẽ dính vào box khác. Kết quả xem đã lưu đủ trong `catalog_ket_qua_2026-11-09.txt`. **Quyết định của bạn:** không dựng lại; catalog giữ là bản trước đợt sửa.
- Bạn trả lời F1 (đề tài môn Deep Learning ở HUST) và F2 (không có video gốc). Tìm thêm 6 nguồn làm cơ sở cho cách chia, ghi ở A1. Trả lời vì sao 11 lớp không có ở val/test (F3), F3 còn chờ quyết định.
- Sắp xếp lại thí nghiệm và chia việc cho nhóm 4 người (đầu mục E): Thí nghiệm 7 làm trước, dùng cấu hình thắng làm bản nền, Thí nghiệm 3 chỉ còn phân tích confusion matrix, Thí nghiệm 6 bắt đầu ngay. Tên người, hạn nộp và tài khoản GPU chưa biết.
- Cấu hình tạm cho mọi lần train trước khi có kết quả Thí nghiệm 7: `optimizer=AdamW lr0=0.000179 momentum=0.9` (bạn chọn). Sửa lưu ý 1 của phần chạy thử: gộp S.505a chỉ làm lr đổi 3,4%, trước đó ghi quá mức.
- Chốt F3: giữ đủ 52 lớp, ghi rõ trong báo cáo rằng mAP test chỉ tính trên 41 lớp (bạn quyết định).
- Repo gốc không có link tải bộ 46 lớp. Bạn quyết định không xin dữ liệu, không dùng model có sẵn của repo; dữ liệu duy nhất là `archive` (C1).
- Bạn tự chạy thử Ultralytics trên máy (M2): train 1 epoch rồi chấm test. Ultralytics đọc đủ 3191 ảnh, 8329 box, 0 lỗi. Kết quả và các lưu ý cho lần train thật ở mục E, phần "Chạy thử 1 epoch". Thêm một dòng đính chính về GPU.
- Bạn đồng ý cố định `optimizer` và `lr0` cho mọi thí nghiệm. Thêm Thí nghiệm 7 (mục E) để chọn giá trị cố định, dựa trên lý thuyết môn IT4653 (slide bài 5, giáo trình mục 4.5, 4.6). Đối chiếu mã nguồn: `momentum` là beta1 của AdamW, và `auto` vẫn là AdamW tới 294 epoch với dataset này.

## Đính chính

Các điều tôi từng nói sai hoặc suy luận chưa đúng. Không dùng các nhận định cũ này trong báo cáo.

| Từng nói | Thực tế | Bằng chứng |
|---|---|---|
| W.246a là tránh sang phải, W.246c là tránh hai bên, nên hai lớp có thể gán sai tên (C3) | Ngược lại: W.246a là tránh hai bên, W.246c là tránh sang phải. Hai mã có vẻ đúng | [W.246a](https://hoatieu.vn/bieu-mau/bien-bao-chu-y-chuong-ngai-vat-w246a-168207), [W.246c](https://hoatieu.vn/bieu-mau/bien-bao-chu-y-chuong-ngai-vat-tranh-sang-ben-phai-w246c-168213), `data/_build/bang_chung_w246.jpg` |
| Dùng hardlink thì ảnh gốc sẽ bị đổi nếu Ultralytics ghi đè JPEG lỗi (A4) | Sai với bản hiện tại: Ultralytics ghi file tạm rồi `os.replace`, nên hardlink không bị đổi. Dữ liệu cũng có 0 JPEG lỗi | [`ultralytics/data/utils.py`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/data/utils.py), `verify_log.txt` |
| Biển xa không có nhãn có thể do quy tắc 0–20 m của README đề tài cũ (A3) | Không áp dụng: `archive` có gán cả biển rất xa (0562, 2668) và biển bị cắt ở mép (1318). README đó có vẻ mô tả bộ 46 lớp khác | 24 ảnh mẫu ở mục B4 |
| Máy này không có GPU và chưa cài Ultralytics, phải train trên Colab/Kaggle (A4 và trả lời ngày 2026-11-09) | Sai: MacBook M2 có GPU dùng được qua MPS, `/opt/miniconda3` đã có Ultralytics 8.4.191 và PyTorch 2.14.1. Đã chạy thử được trên máy. Train nhiều lần vẫn nên dùng Colab/Kaggle vì chậm (xem mục E) | Log bạn chạy ngày 2026-11-09, `data/runs/sanity_1epoch/args.yaml` |

---

## A. Chia tập dữ liệu

### 🔴 A1. Rò rỉ dữ liệu giữa train và test ✅
**Ghi chú của bạn:** Hỏi có bài báo nào về cách chia dữ liệu và cách sửa. Sau đó: chọn bài có lời giải rõ ràng (ok), dùng cách tôi thấy tốt nhất.

**Vấn đề:** Ảnh đánh số liền nhau là frame từ video. Split gốc chia ngẫu nhiên từng ảnh, nên 612/639 ảnh test có ảnh train số liền kề.

**Tài liệu chọn:**
- [Find the Leak, Fix the Split](https://arxiv.org/html/2511.13944v2) (Glazner và cộng sự, preprint arXiv, đăng lần đầu 11/2025): trích đặc trưng frame, giảm chiều bằng PaCMAP, gom cụm bằng HDBSCAN, gán nguyên cụm vào một tập. Bài không đo ảnh hưởng lên model và không nêu chi tiết cách gán cụm.
- [Roberts et al. 2017, Ecography](https://pred.uni-regensburg.de/id/eprint/1455/): cơ sở lý thuyết về chia theo khối khi dữ liệu phụ thuộc nhau.
- [Ultralytics Academy](https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/split-the-dataset-correctly): khuyến nghị chia theo cảnh hoặc video.

Mới đọc tóm tắt hoặc trang HTML qua công cụ, cần đọc toàn văn trước khi trích dẫn.

**Cơ sở bổ sung (tìm thêm 2026-11-09, vì không có video gốc):** Bạn yêu cầu tìm nguồn chứng minh cách chia có cơ sở.

| Nguồn | Đã đọc | Nội dung chính | Liên hệ với dataset này |
|---|---|---|---|
| Stallkamp và cộng sự, *The German Traffic Sign Recognition Benchmark*, IJCNN 2011 ([PDF](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf)) | Toàn văn (bản trích chữ lưu ở `data/_build/nguon_gtsrb_ijcnn2011.txt`) | Mỗi biển thật được quay thành một chuỗi ảnh gọi là *track*. Train/test chia "theo lớp và theo track", để giữ phân bố lớp và để mọi ảnh của một biển nằm cùng một tập. Bài cũng nói frame liền nhau của biển đi qua chậm rất giống nhau, không làm dữ liệu đa dạng hơn. Lớp có dưới 9 track bị loại khỏi dataset | Cùng bài toán biển báo. Nhóm frame của mình đóng vai trò track. Cân lớp khi chia giống "class-wise". Việc loại lớp ít track là tiền lệ cho câu F3 |
| Figueiredo và Mendes, IEEE Access 2024, DOI [10.1109/ACCESS.2024.3383047](https://doi.org/10.1109/ACCESS.2024.3383047) | Chỉ tóm tắt qua kết quả tìm kiếm | Dataset phát hiện đối tượng dựng từ frame video mà chia ngẫu nhiên thì bị rò rỉ. Họ gom cụm ảnh (đặc trưng CLIP, t-SNE, DBSCAN/OPTICS/Agglomerative), gán nguyên cụm vào train/val/test, thử bằng YOLOv8 | Gần nhất với dataset này: phát hiện đối tượng, nguồn video, YOLOv8 |
| Barz và Denzler, *Do We Train on Test Data? Purging CIFAR of Near-Duplicates*, J. Imaging 2020 ([arXiv](https://arxiv.org/abs/1902.00423)) | Tóm tắt | 3,3% ảnh test CIFAR-10 và 10% của CIFAR-100 có ảnh gần trùng trong train. Thay chúng bằng ảnh mới thì độ chính xác giảm 9–14% (tương đối) | Bằng chứng định lượng rằng ảnh gần trùng làm điểm test bị thổi phồng. Split gốc của mình có 30,7% ảnh test gần trùng train, cao hơn nhiều |
| scikit-learn, [Cross-validation iterators for grouped data](https://scikit-learn.org/stable/modules/cross_validation.html) | Đã đọc trang | Giả định độc lập (i.i.d.) bị phá khi dữ liệu có nhóm mẫu phụ thuộc nhau. Mẫu ở tập đánh giá phải đến từ nhóm không có mặt trong train. `StratifiedGroupKFold` vừa giữ nhóm vừa cố giữ phân bố lớp | Định nghĩa chuẩn của group split và stratified group split |
| Kapoor và Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science*, Patterns 2023 ([arXiv](https://arxiv.org/abs/2207.07048)) | Tóm tắt | Rò rỉ dữ liệu gây lỗi tái lập ở ít nhất 294 bài thuộc 17 ngành. Bài phân loại 8 kiểu rò rỉ | Rò rỉ là lỗi phổ biến, nên ghi rõ cách chống trong báo cáo |
| Babu và cộng sự, *Improving Image Data Leakage Detection in Automotive Software*, arXiv 2410.23312 (2024) | Tóm tắt | Phương pháp phát hiện rò rỉ ảnh cho dữ liệu xe tự lái, tìm ra rò rỉ chưa ai biết trong KITTI | Dữ liệu quay từ xe cũng gặp đúng vấn đề này, kể cả bộ nổi tiếng |

**Cách làm của mình khớp với tài liệu ở đâu:**
- Chia theo nhóm ảnh phụ thuộc, không chia từng ảnh: GTSRB (track), Figueiredo 2024 và Glazner (cụm), scikit-learn (group).
- Cân phân bố lớp khi chia: GTSRB ("class-wise"), `StratifiedGroupKFold`.
- Kiểm chứng bằng cách đo ảnh gần trùng giữa các tập: cùng tinh thần với Barz và Denzler. Kết quả: 30,7% xuống 0%.

**Khác với tài liệu, và hạn chế:**
- Không có video gốc nên không có track thật như GTSRB. Nhóm được suy từ độ giống ảnh.
- Đặc trưng đơn giản (ảnh thu nhỏ xám) thay vì CLIP hoặc DINO. Nếu cùng một biển được quay hai lần ở góc khác nhau, độ giống có thể dưới 0,95 nên không bị gộp, và vẫn còn một ít rò rỉ mà phép đo không thấy.
- Chỉ GTSRB và trang scikit-learn được đọc toàn văn. Các nguồn còn lại mới đọc tóm tắt, cần đọc toàn văn trước khi trích số liệu vào báo cáo.

**Cách làm:** Cùng ý tưởng "giữ nguyên cụm" của bài báo, nhưng dùng đặc trưng đơn giản hơn.
1. Tính tương quan giữa ảnh thu nhỏ xám 64×36 của mọi cặp ảnh.
2. Cắt nhóm khi hai ảnh liền số có tương quan dưới `cut`, hoặc khi đổi kích thước ảnh.
3. Gộp hai nhóm nếu có cặp ảnh tương quan trên `merge` (cảnh gần trùng).
4. Gán nguyên nhóm vào một tập bằng simulated annealing, ưu tiên mọi lớp có ở train, lớp có từ 2 nhóm trở lên có mặt ở val và test, rồi đến tỉ lệ ảnh và box gần 70/15/15.

**Lý do không dùng DINO + HDBSCAN:** Dữ liệu có sẵn thứ tự frame, cách đơn giản không cần cài thư viện nặng và đạt mục tiêu đo được bên dưới.

**Bằng chứng so sánh** (`du_lieu/log/split_compare.txt`, phạt thiếu lớp 200/300):

| Cấu hình | Số nhóm | Ảnh test giống train > 0,95 | Lớp thiếu ở val hoặc test |
|---|---:|---:|---:|
| Split gốc | không có | 30,7% | 0 |
| cut 0,3, merge 0,95 | 86 | 0,0% | 22 |
| cut 0,4, merge 0,95 | 129 | 0,0% | 19 |
| cut 0,5, merge 0,95 | 171 | 0,0% | 17 |
| cut 0,5, merge 0,97 | 192 | 0,4% (val 1,6%) | 16 |

**Quyết định:** `cut=0.5, merge=0.95, seed=0`, phạt thiếu lớp 800/1200. Đây là cấu hình giữ 0% ảnh gần trùng mà thiếu ít lớp nhất. Tăng phạt giúp giảm số lớp thiếu ở test từ 12 xuống 10.

**Kết quả cuối** (`du_lieu/splits/split_report.txt`):

| | Độ giống lớn nhất với train: median | p90 | > 0,95 |
|---|---:|---:|---:|
| Test split gốc | 0,921 | 0,974 | 30,7% |
| Val mới | 0,677 | 0,860 | 0,0% |
| Test mới | 0,730 | 0,812 | 0,0% |

**Cái giá phải trả:**
- 10 lớp chỉ có ở train, **không đánh giá được**: 15 (W.203c), 21 (W.203b), 24 (P.129), 27 (W.225), 29 (P.104), 42 (R.301e), 44 (W.233), 47 (P.124b1), 48 (W.210), 51 (W.246c). Từ 2026-11-09 thêm lớp 28 (S.505a xe tải và công) sau khi chuyển box 1193#2 (xem C5), thành 11 lớp.
- 7 lớp nữa thiếu ở val nhưng có ở test: 11, 20, 28, 33, 38, 43, 50.
- Một số lớp có rất ít box: test có lớp 28 (1), 14, 25, 34 (2–3); val có lớp 9, 26, 36, 37 (1). AP của các lớp này dao động mạnh.
- `mAP` trên test mới chỉ tính trên 41 lớp (42 trước khi chuyển box 1193#2), nên không so trực tiếp được với split gốc (52 lớp) hay kết quả 0,984 trong README của đề tài cũ.

**Giới hạn:** Nhóm được suy ra từ độ giống ảnh, không có thông tin video thật. Một đoạn đường đi qua hai lần với góc khác có thể không bị phát hiện. Cần báo cáo rõ giới hạn này.

### A1b. Cách xáo trộn khi nguồn là video ✅ (ghi 2026-11-09)
**Ghi chú của bạn:** Hỏi đã ghi lại cách chia khi nguồn liên tục chưa, xáo trộn thế nào và vận dụng phương pháp gì.

**1. Xáo trộn ở mức nào**
- Với nguồn liên tục (frame cắt từ video), chỉ được xáo trộn ở mức **nhóm frame**, không xáo trộn từng ảnh. Hai frame liền nhau gần như giống hệt (tương quan trung vị 0,9). Xáo trộn từng ảnh sẽ đưa hai frame gần trùng vào hai tập khác nhau, đúng lỗi của split gốc.
- Xáo trộn **trong** tập train khi huấn luyện (Ultralytics đổi thứ tự ảnh mỗi epoch) thì không sao. Việc đó chỉ đổi thứ tự học, không đưa ảnh qua lại giữa các tập.

**2. Phương pháp vận dụng**
- **Group split / block cross-validation** ([Roberts et al. 2017](https://pred.uni-regensburg.de/id/eprint/1455/)): các mẫu phụ thuộc nhau được gom thành một nhóm, và cả nhóm nằm trọn trong một tập.
- **Stratified group split**: vừa giữ nguyên nhóm, vừa cân tỉ lệ lớp giữa các tập. Đây là việc thuật toán ở A1 làm.
- Scikit-learn có sẵn `StratifiedGroupKFold`, nhưng chỉ dành cho mỗi mẫu một nhãn. Ở đây mỗi ảnh có nhiều box thuộc nhiều lớp, nên tự viết hàm tối ưu (simulated annealing) trên số box của từng lớp.

**3. Yếu tố ngẫu nhiên**
- Thuật toán khởi tạo ngẫu nhiên 20 lần, mỗi lần chạy 8000 bước simulated annealing, rồi giữ phương án tốt nhất. Mọi lựa chọn ngẫu nhiên lấy từ một bộ sinh số có `seed=0`.
- Chạy lại với seed 0 cho ra đúng split đang dùng (2138/474/579 ảnh), nên **kết quả lặp lại được**.

**4. Đổi seed thì kết quả thay đổi bao nhiêu** (đo 2026-11-09, `du_lieu/log/split_seeds.txt`; chỉ đo, không thay split đang dùng; số box tính trên nhãn `archive` trước khi sửa):

| Seed | Ảnh cùng tập với seed 0 | Train / val / test (ảnh) | Ảnh val, test giống train > 0,95 | Lớp thiếu ở val hoặc test | Lớp không có ở cả val và test |
|---:|---:|---|---:|---:|---:|
| 0 | 100% | 2138 / 474 / 579 | 0% | 17 | 10 |
| 1 | 64,7% | 2161 / 527 / 503 | 0% | 16 | 10 |
| 2 | 74,4% | 2126 / 512 / 553 | 0% | 17 | 10 |
| 3 | 58,8% | 2161 / 536 / 494 | 0% | 16 | 10 |
| 4 | 65,2% | 2175 / 523 / 493 | 0% | 16 | 10 |

**Đọc kết quả:**
- Seed nào cũng giữ được 0% ảnh gần trùng, và tỉ lệ các tập ổn định quanh 67/16/17.
- 10 lớp không đánh giá được **giống hệt nhau ở mọi seed** (15, 21, 24, 27, 29, 42, 44, 47, 48, 51). Vậy nguyên nhân nằm ở dữ liệu chứ không ở seed: mỗi lớp này chỉ xuất hiện trong 1 nhóm frame, nên không thể chia ra nhiều tập. Muốn đánh giá được thì phải thu thêm dữ liệu từ đoạn đường khác. Lớp 28 thành không đánh giá được sau khi sửa box 1193#2, không do seed.
- Chỉ 59–74% ảnh giữ nguyên tập khi đổi seed. Vậy **ảnh nào vào test phụ thuộc nhiều vào seed**, và điểm AP của từng lớp có thể thay đổi theo cách chia.

**Quyết định:**
- Giữ split seed 0 cố định cho mọi thí nghiệm, để các bản được so trên cùng tập test.
- Khi báo cáo, nêu rằng điểm test phụ thuộc vào một cách chia cụ thể. Nếu có thời gian: train cấu hình chính trên 2–3 split với seed khác nhau để đo độ dao động do cách chia (khác với độ dao động do seed huấn luyện).

### 🟠 A2. Tập validation ✅
**Ghi chú của bạn:** Val riêng là gì, tại sao cần, cách chọn. Sau đó: ok.

**Trả lời:** Val dùng trong lúc huấn luyện để chọn checkpoint và chỉnh siêu tham số. Test chỉ dùng một lần cuối để báo cáo. Dùng test để chọn model làm điểm test không còn khách quan.

**Cách chọn:** Lấy nguyên cả nhóm frame (không lấy ngẫu nhiên từng ảnh), khoảng 15% số ảnh, tỉ lệ box từng lớp gần tỉ lệ chung.

**Thay đổi:** `images/val`, `labels/val` có 474 ảnh, tạo cùng bước A1.

### 🟡 A3. 25 ảnh nhãn rỗng ✅
**Ghi chú của bạn:** Xoá đi, nhưng copy một bản khác và giữ nguyên `archive`.

**Bằng chứng:** Đã xem cả 25 ảnh. Đa số không có biển rõ, nhưng một số có biển nhỏ chưa gán nhãn (0187; 1514, 1515; 1703–1708; 2031, 2051; 2999). Vì vậy không dùng làm ảnh nền. Cũng không hợp làm val vì val cần nhãn để tính mAP.

**Thay đổi:** Không có trong `data/vn_signs_yolo`, danh sách ở `du_lieu/splits/excluded_empty.txt`: 0060, 0061, 0187, 0544, 1514, 1515, 1703–1708, 2015–2017, 2031, 2051, 2723, 2766, 2793–2797, 2999. `archive` vẫn còn đủ.

### 🟡 A4. Cấu trúc chuẩn Ultralytics ✅
**Ghi chú của bạn:** Làm nếu đảm bảo an toàn, đặt trong folder nhưng git ignore trước.

**Thay đổi:**
- `.gitignore` tạo trước, đã kiểm tra bằng `git check-ignore`.
- `data/vn_signs_yolo/{images,labels}/{train,val,test}` và `data.yaml`.
- Ảnh là bản clone APFS (`cp -c`), không dùng chung inode với ảnh gốc, tốn thêm gần như 0 dung lượng. Nhãn là file mới.
- `data.yaml` không ghi `path`: Ultralytics lấy thư mục chứa file yaml làm gốc, nên mang sang Colab/Kaggle vẫn chạy được.

**Kiểm tra** (`du_lieu/log/verify_log.txt`):
- 3191 cặp ảnh/nhãn, không ảnh nào nằm ở hai tập. Bằng đúng `archive` trừ 25 ảnh rỗng.
- 8332 box (8334 trừ 2 dòng B1), 0 box lỗi biên.
- 0 ảnh khác nội dung so với gốc, 0 ảnh dùng chung inode, 0 JPEG lỗi.
- `archive` khớp md5 trước và sau: 6437 file (`archive_md5_truoc.txt`).

**Kiểm tra bằng Ultralytics (2026-11-09):** Đã chạy thật trên máy (trước đó tôi ghi nhầm là máy chưa cài, xem Đính chính). Ultralytics đọc đúng cấu trúc, `data.yaml` không có `path` vẫn chạy. Chi tiết ở mục E, phần "Chạy thử 1 epoch".

---

## B. Chất lượng nhãn

### 🔴 B1. Hai box chồng khít khác lớp ✅
**Ghi chú của bạn:** Nhìn lại thì hai biển giống nhau, chỉ khác vị trí hai loại xe trên biển, và đều là biển 23.

**Quyết định:** Giữ lớp 23 (`P.111`, Cấm xe hai và ba bánh). Xoá dòng lớp 14 (Cấm xe tải) ở ảnh 1965 và lớp 34 (Cấm xe sơ-mi rơ-moóc) ở ảnh 2095.

**Thay đổi (chỉ trên bản copy):**
- `1965`: xoá `14 0.384806 0.272862 0.062949 0.11281`
- `2095`: xoá `34 0.457935 0.261132 0.039278 0.086`

**Ghi cho báo cáo:** Lớp 23 có biến thể đổi vị trí hình xe trên biển.

### 🟡 B2. Box vượt khung ảnh ✅
**Ghi chú của bạn:** Sửa đi.

**Trả lời:** Đúng là biển bị khung hình cắt ở mép phải, box tràn ra ngoài 1,9% chiều rộng ảnh.

**Thay đổi (chỉ trên bản copy):** Ảnh `0248`, lớp 49: `0.992188 0.345370 0.053125 0.101852` thành `0.982813 0.345370 0.034375 0.101852`. Box dừng đúng ở mép ảnh.

### 🟠 B3. Box nhỏ 📝
**Ghi chú của bạn:** Ok, ghi chú lại để bám theo sau.

**Số liệu** (`du_lieu/log/measure_log.txt`):
- Box "nhỏ" theo COCO (diện tích dưới 32×32 px trên ảnh gốc): 3894/8332 (46,7%), trong 1764 ảnh.
- Box có cạnh ngắn dưới 8 px ở imgsz 640: 811, trong 483 ảnh.
- Tỉ lệ box nhỏ cao nhất: lớp 51 (5/5), 20 (41/52), 18 (162/214), 2 (341/470), 3 (322/454).

**Kế hoạch:** Xem Thí nghiệm 1 ở mục E.

### 🟡 B4. Biển bị sót nhãn ✅ catalog, 📝 sót nhãn
**Ghi chú của bạn:** Ok với catalog. Bạn gửi một bài trao đổi về dataset này báo lỗi: có chỗ 4–5 biển nhưng chỉ gán 1, và có biển mờ đến mức phóng to vẫn không nhận ra.

**Tự kiểm tra:** Xem 24 ảnh ngẫu nhiên có vẽ box (0009, 0128, 0295, 0512, 0562, 0734, 0802, 1129, 1318, 1433, 1552, 2108, 2199, 2396, 2531, 2548, 2668, 2728, 2892, 2923, 3036, 3042, 3048, 3062).
- Box nhìn chung ôm đúng biển, kể cả biển rất nhỏ.
- Nhiều biển không có nhãn là biển chỉ dẫn xanh hoặc biển chữ đỏ, các loại này không thuộc 52 lớp nên không tính là sót.
- Không thấy trường hợp "4–5 biển chỉ gán 1" trong mẫu này. Hai ảnh đáng xem lại: 0512, 2199.

**Giới hạn:** Mẫu 24 ảnh quá nhỏ và xem ở độ phân giải thấp, **không ước lượng được tỉ lệ sót**. Bài trao đổi cho thấy đã có người gặp lỗi này, chỉ chưa rõ mức độ.

**Ảnh hưởng nếu sót:** Model phát hiện đúng biển chưa gán nhãn sẽ bị tính là dương tính giả, làm giảm precision và mAP đo được.

**Kế hoạch:** Sau lần train đầu, lọc các dự đoán độ tin cậy cao mà không có nhãn để tìm ảnh sót.

**Thay đổi:** Catalog ở `catalog/catalog.html`, tab "Tất cả biển" có tên, mã, số box từng tập và tối đa 6 ảnh cắt mỗi lớp.

---

## C. Lớp

### 🔴 C1. Mất cân bằng và gộp dataset ✅ chưa gộp
**Ghi chú của bạn:** Hỏi gộp với dataset hiện có thì sao. Không có sẵn bộ 46 lớp của repo. Có ghi "đã có 1 editor làm trước phần này" kèm bài trao đổi ở B4.

**Dataset tìm được:**

| Dataset | Quy mô | Nhận xét |
|---|---|---|
| Bộ của đề tài cũ (README cũ) | 46 lớp, 11.577 ảnh, GoPro, TP.HCM | Gần nhất, nhưng không có link và bạn không có |
| [Zalo Traffic Sign 2020](https://datasetninja.com/zalo-traffic-sign) | 4.500 ảnh, Google Street View | Kiểu ảnh khác hẳn |
| [Vietnamese Traffic Signs](https://datasetninja.com/vietnamese-traffic-signs) | 1.170 ảnh | Không rõ giấy phép |
| [VN_traffic_sign_2](https://universe.roboflow.com/trafficsignvietnamese/vn_traffic_sign_2) | 111 ảnh, CC BY 4.0 | Quá ít |

Chưa tải và chưa kiểm tra chất lượng các bộ trên.

**Đánh giá:** Trong 5 lớp hiếm nhất của `archive`, 4 lớp (W.233, W.246c, P.124b1, W.202b) không có trong bảng 46 biển của đề tài cũ, lớp còn lại (W.203b) chỉ khớp chung với mã `203`. So bảng đó với 52 lớp: 33 lớp có mã tương ứng, trong đó 7 lớp bị gộp chung (201, 203, 205).

**Quyết định:** Chưa gộp. Bộ gần nhất không có sẵn, và các bộ còn lại phải ánh xạ lớp, kiểm tra trùng ảnh và giấy phép.

**Bản gán nhãn của editor:** Bài trao đổi chỉ cho thấy một người đề nghị gán bổ sung, chưa cho thấy đã có bản hoàn chỉnh. Bạn không có link, nên không dùng bản đó (2026-11-02). Bài trao đổi được giữ làm bằng chứng cho B4.

**Hệ quả:** Lớp hiếm vẫn giữ nguyên số mẫu hiện có. Các lớp chỉ có ở train (xem A1) vẫn chưa đánh giá được.

**Cập nhật 2026-11-09:** Repo gốc ([thuanvipghe/Traffic-sign-VietNam-recognition](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition)) không có thư mục dữ liệu, không có bản phát hành, không có link tải bộ 46 lớp. Repo chỉ còn model đã train (`YOLOV8/*.pt`, Faster R-CNN), `results.csv` và ảnh minh hoạ. Từ 2026-11-10 các file này đã xoá khỏi repo của nhóm, còn ở nhánh `de-tai-cu`.

**Quyết định của bạn (2026-11-09):**
- Không xin bộ 46 lớp từ tác giả.
- Không dùng model có sẵn trong repo làm model của đề tài, vì đề tài phải tự xây và tự train model. Model và `results.csv` cũ chỉ dùng làm tài liệu tham khảo (xem H4).
- Dữ liệu duy nhất của đề tài là `archive`, sau khi làm sạch thành `data/vn_signs_yolo`.

### 🟠 C2. Tên lớp ✅
**Ghi chú của bạn:** Dùng tên tiếng Việt.

**Thay đổi:** `names` trong `data.yaml` lấy từ `classes_vie.txt`. Sửa tên tiếp trong catalog.

**Lệch giữa file Anh và Việt (nhận định theo hiểu biết, chưa đối chiếu nguồn):** id 1, 22, 37.

### 🟠 C3. W.246a và W.246c ✅
**Ghi chú của bạn:** Check 0961–0964 có phải box nhỏ không. Nếu gán sai thì đổi tên hay gán lại. Cho vào catalog để vừa nhìn vừa sửa.

**0961–0964:** Có. Box lớp 26 chỉ 13–15 × 11–12 px trên ảnh 960×540. Lớp 26 có 41/80 box nhỏ, lớp 51 có 5/5. Các box lớn hơn mới thấy rõ biển (1120, 1221, 1006).

**Đối chiếu với ảnh** (`data/_build/bang_chung_w246.jpg`):
- Lớp 26 (W.246a): mũi tên rẽ ra hai bên, khớp "vòng tránh hai bên". Mã đúng, nhưng tên "Chướng ngoại vật phía trước" chung chung và có lỗi chính tả.
- Lớp 51 (W.246c): mũi tên một hướng, ảnh rất mờ. Có vẻ khớp "tránh sang bên phải" và tên trong file.

Nguồn là trang tra cứu [hoatieu.vn](https://hoatieu.vn/bieu-mau/bien-bao-chu-y-chuong-ngai-vat-w246a-168207), chưa đối chiếu văn bản QCVN gốc. Mới xem 8 box lớn, chưa xem hết.

**Trả lời "đổi tên hay gán lại":** Không cần gán lại. Id chỉ là số, tên chỉ là chú thích. Chỉ cần gán lại những box lẫn biển khác.

**Kết quả catalog (2026-11-09):** Lớp 26: 56 Đúng, 24 Không rõ (chủ yếu 0961–0981, box rất nhỏ). Lớp 51: 3 Đúng, 2 Không rõ. Không box nào bị đánh Sai, nên không cần gán lại.

**Thay đổi:** Đổi tên lớp 26 trong `data.yaml` thành "Chú ý chướng ngại vật, vòng tránh hai bên" (bạn đồng ý). `classes_vie.txt` trong `archive` giữ nguyên.

### 🟠 C4. Mã không chuẩn và dấu `*` ✅
**Ghi chú của bạn:**
- `*` có thể để phân biệt cùng loại biển nhưng khác thông số, như 40–50–60 km/h.
- id 31 có biển đó thật.
- Mã kiểu cũ có ảnh hưởng gì. Cho vào catalog.

**Trả lời:** Mã kiểu cũ như `B.8a` không ảnh hưởng đến huấn luyện vì YOLO chỉ dùng số id. Nó chỉ ảnh hưởng khi đối chiếu quy chuẩn trong báo cáo hoặc khi gộp dataset.

**Nhận định về `*`:** Giả thuyết của bạn hợp lý với P.127 và P.117 (biển có thông số). Với `P.124a*`, `W.239b*`, `P.106a*Xe tải` thì cần xem ảnh.

**Kết quả catalog (2026-11-09), xem đủ 408 box:**

| Lớp | Đúng | Không rõ | Sai | Ghi chú của bạn |
|---|---:|---:|---:|---|
| 31 Camera | 38 | 0 | 0 | 1319–1328 bị ngược chiều |
| 34 B.8a | 78 | 3 | 0 | 1901, 1924, 1931 không chỉ cấm mỗi xe sơ-mi rơ-moóc |
| 14 P.106a* | 71 | 13 | 0 | 1329–1331 bị ngược |
| 16 P.117* | 44 | 26 | 3 | 0551–0593 là 2,0 m; 0595–0603 là 2,5 m; 0797–0814 là 2,2 m |
| 17 P.124a* | 103 | 2 | 0 | |
| 43 W.239b* | 17 | 10 | 0 | |

**Kết luận về dấu `*`:** Ít nhất với lớp 16, dấu `*` gom các biển cùng loại nhưng khác thông số (2,0 / 2,5 / 2,2 m), đúng như bạn đoán.

**Quyết định (2026-11-09):**
- Lớp 16 giữ chung một lớp, không tách theo chiều cao. Lý do: tách ra thì 2,5 m chỉ còn 9 box. Lớp tốc độ P.127 thì khác vì dataset đã tách sẵn và mỗi mức có hàng trăm box.
- Lớp 34: giữ 3 box 1901, 1924, 1931 ở lớp 34, chỉ ghi chú.
- Xoá 3 box lớp 16 bị đánh Sai (2088#1, 2278#1, 2279#1). Khi xem lại: 2088 là biển tròn bị khung hình cắt mất một nửa, 2278 và 2279 là biển tròn có vạch bị tấm biển xanh che một phần, không giống biển giới hạn chiều cao. Biển vẫn còn trong ảnh nhưng không có nhãn, ảnh hưởng nhỏ vì chỉ 3 box.

**Phát hiện khi kiểm tra ghi chú "bị ngược":** Ảnh không bị lật mà **bị xoay 90°** cả khung, và đoạn bị xoay rộng hơn ghi chú: 39 ảnh **1319–1357**. Đây chính là 39 ảnh dọc 540×960 (không có thẻ xoay EXIF, nên YOLO cũng đọc nằm ngang). Cả 39 đều ở test, gồm lớp 31 (10 box), 14 (3 box) và 12 (26 box, lớp này không có trong catalog).

**Thay đổi:** Xoay trái 90° cả 39 ảnh (lưu lại JPEG chất lượng 95) và đổi nhãn theo công thức `x' = y, y' = 1 − x, w' = h, h' = w`. Đã vẽ lại box lên 8 ảnh để kiểm tra: box khớp biển. Mọi ảnh giờ là 960×540 hoặc 1920×1080.

**Kiểm tra rò rỉ sau khi xoay:** Độ giống lớn nhất của 39 ảnh với train là 0,761 (median 0,656), với val là 0,684. Không ảnh nào vượt 0,95.

### 🟠 C5. Biển phụ S.505a ✅ catalog, 📝 Thí nghiệm 2
**Ghi chú của bạn:** Ghi lại để thử hai bản như B3. Gộp một lớp có đảm bảo tốt hơn và lọc 100% không. Thêm phần trong catalog để bạn lọc giúp.

**Trả lời:** Không đảm bảo, phải đo. Không thể lọc 100%: detector luôn có sót và nhầm. Gộp cũng làm mất thông tin loại xe.

**Thay đổi:** Tab C5 trong catalog có 190 box (25, 28, 30), mỗi ảnh cắt cho thấy cả biển chính phía trên.

**Kết quả catalog (2026-11-09), xem đủ 190 box:** Lớp 25: 25 Đúng. Lớp 28: 26 Đúng, 1 Sai. Lớp 30: 138 Đúng.

**Ghi chú của bạn:**
- Lớp 30 đúng nhưng các biển có điều kiện khác nhau (chữ ghi kèm).
- 1193 là biển "chỉ xe tải, trừ xe phục vụ trong khu CNC", nên box 1193#2 phải chuyển từ lớp 28 sang lớp 30.

**Quyết định:** Lớp 30 giữ nguyên, chỉ ghi chú cho báo cáo.

**Thay đổi (2026-11-09):** Box 1193#2 chuyển từ lớp 28 sang lớp 30 (`28 0.507812 0.415741 0.065625 0.172222` thành `30 ...`, tọa độ giữ nguyên). Nhãn cũ sao lưu trong `backup_truoc_catalog_2026-11-09/labels/test/1193.txt`.

**Hệ quả:** Đây là box lớp 28 duy nhất trong test. Giờ lớp 28 chỉ còn 26 box ở train, không đánh giá được. Lớp 30: train 68, val 19, test 52. Ưu tiên nhãn đúng hơn là giữ lớp 28 trong test. Kế hoạch ở Thí nghiệm 2.

### 🟡 C6. Nhóm lớp dễ nhầm 📝
**Ghi chú của bạn:** Giải thích vì sao dễ nhầm, ghi lại để thử hai bản đã lọc và chưa lọc.

**Trả lời:**
- Các nhóm cấm rẽ, cấm quay đầu, cấm xe cùng khung tròn đỏ, chỉ khác biểu tượng nhỏ bên trong.
- Các cặp trái/phải (6/50, 15/21, 1/37) gần như là ảnh phản chiếu của nhau.
- Ở imgsz 640, trung vị cạnh ngắn của box chỉ khoảng 18 px, nên chi tiết phân biệt gần như mất.

**Kế hoạch:** Thí nghiệm 3.

---

## D. Yêu cầu khi huấn luyện

### 🔴 D1. Không lật ngang 📝
**Ghi chú của bạn:** Đồng ý, note lại cho báo cáo.

**Ghi cho báo cáo:** Đặt `fliplr=0.0` (mặc định của Ultralytics là 0,5). Nhiều lớp phân biệt bằng hướng trái/phải (cấm rẽ trái và phải, ngoặt trái và phải, thu hẹp trái và phải). Lật ảnh mà giữ nguyên nhãn sẽ tạo mẫu sai.

### 🟠 D2. Độ phân giải 📝
**Ghi chú của bạn:** Làm theo cách tôi thấy ổn, nhưng phải có bằng chứng và ghi chú.

**Bằng chứng** (`measure_log.txt`, tính từ 8332 box):

| imgsz | Trung vị cạnh ngắn box | Cạnh ngắn < 8 px | < 16 px | Số pixel so với 640 |
|---:|---:|---:|---:|---:|
| 640 | 17,7 px | 9,7% | 44,6% | 1× |
| 960 | 26,5 px | 0,9% | 19,5% | 2,25× |
| 1280 | 35,3 px | 0,0% | 9,7% | 4× |

Bước lưới nhỏ nhất của YOLOv8 là 8 px, nên box dưới 8 px nhỏ hơn cả một ô lưới.

**Quyết định:** Dùng `imgsz=960` làm mặc định. So với 640, tỉ lệ box dưới 8 px giảm từ 9,7% xuống 0,9%, chi phí xử lý tăng 2,25 lần thay vì 4 lần như 1280. Thử 1280 nếu GPU đủ. Chưa đo tốc độ hoặc mAP thật.

### 🟡 D3. Dung lượng 📝
**Ghi chú của bạn:** Làm theo cách tôi thấy ổn, nhưng phải có bằng chứng và ghi chú.

**Bằng chứng** (`measure_log.txt`):
- Tổng 788 MB, trong đó 805 ảnh 1920×1080 chiếm 385 MB.
- Resize ảnh lớn về 1280: JPEG q95 còn khoảng 615 MB (giảm 22%), q90 còn khoảng 549 MB (giảm 30%). Ước tính từ mẫu 100 ảnh.
- Nén zip deflate trên JPEG chỉ còn 99,8%, gần như không giảm.

**Quyết định:** Không resize, vì giảm 22–30% không đáng đổi lấy việc nén lại ảnh khi biển vốn đã nhỏ, và làm mất khả năng thử imgsz cao hơn. Đóng gói zip không nén (`-0`) để giữ nguyên ảnh và nhanh hơn.

**Cách làm:**
1. Trên máy: tạo zip.
2. Tải lên Google Drive hoặc tạo Kaggle Dataset một lần.
3. Trên Colab: copy zip từ Drive vào `/content` rồi giải nén, không đọc trực tiếp từ Drive khi train.
4. Dùng `cache=disk` hoặc `cache=ram` nếu đủ bộ nhớ.

```bash
cd data && zip -0 -r vn_signs_yolo.zip vn_signs_yolo
```

Chưa tạo file zip vì dữ liệu có thể đổi sau khi bạn sửa trong catalog. Bước 3 và 4 là kinh nghiệm chung, chưa đo trên Colab.

---

## E. Kế hoạch thí nghiệm

**Lưu ý về tập nhãn (thêm 2026-11-09, xem câu H1):** Mỗi lần chạy phải ghi lại phiên bản tập nhãn: số lớp, danh sách tên, các lớp có mặt trong test, và mã md5 của thư mục `labels`. Hai kết quả chỉ so được khi chấm trên cùng tập nhãn test. Mọi thay đổi nhãn (gộp lớp, xoá lớp, sửa box) tạo ra một phiên bản mới.

**Quy tắc chung:** Cùng split, cùng siêu tham số (`imgsz=960`, `fliplr=0`), mỗi bản chạy 3 seed để biết độ dao động. Đánh giá trên **cùng một tập test**, báo cáo mAP50, mAP50-95, AP từng lớp và confusion matrix. Bản nền là `data/vn_signs_yolo` sau khi áp dụng kết quả catalog.

| # | Nguồn | So sánh | Đo thêm | Lưu ý |
|---|---|---|---|---|
| 1 | B3 | Có box nhỏ và không có box nhỏ | AP theo cỡ box | Nếu chỉ xoá nhãn mà giữ ảnh, model học coi biển thật là nền. Nên so cả phương án bỏ nguyên ảnh. Định nghĩa COCO loại gần nửa số box nên cần ngưỡng hẹp hơn, ví dụ dưới 8 px ở imgsz train |
| 2 | C5 | 3 lớp S.505a và gộp 1 lớp | Recall biển phụ, nhầm với biển chính | Muốn so công bằng, gộp kết quả của bản 3 lớp về 1 lớp rồi mới tính |
| 3 | C6 | ~~Dataset chưa lọc và đã lọc~~ Từ 2026-11-09: chỉ phân tích confusion matrix nhóm dễ nhầm, không train lại | Confusion matrix nhóm dễ nhầm | Lý do bỏ phần train lại: chỉ 6/2138 ảnh train khác nhãn giữa hai bản (7 dòng), chênh lệch sẽ chìm trong dao động giữa seed; test của hai bản khác nhau (39 ảnh xoay) nên không chấm chung được (đo 2026-11-09) |

### Kế hoạch chính: xây dựng mạng (2026-11-10)

**Vì sao đổi:** Bạn cho biết cô coi trọng nhất phần **xây dựng mạng** và **trả lời các câu hỏi đi kèm trong quá trình**. Kế hoạch ngày 2026-11-09 (bản sao lưu ở scratchpad, tên `DATASET_REVIEW_truoc_huongAB.md`) gần như chỉ chạy YOLO có sẵn rồi đổi dữ liệu hoặc optimizer, nhóm không tự xây phần mạng nào. Đối chiếu với tài liệu môn ở `/Users/tdu/2026.1/học sâu/` (slide bài 3, 4, 5, 8; giáo trình `dl_lecturenotes` chương 3, 4, 6; các đề thi). Đề thi hỏi nhiều về tính số tham số và kích thước đầu ra của lớp tích chập, tham số BatchNorm, receptive field, khối ResNet, overfitting và cách chữa, focal loss.

**Quyết định của bạn (2026-11-10):** Hướng A làm chính, thêm hướng B.

#### Hướng A (chính): tự xây CNN phân loại 52 loại biển bằng PyTorch

**Dữ liệu:** Cắt vùng biển từ `data/vn_signs_yolo`, giữ nguyên cách chia train/val/test (không rò rỉ, mục A1). Train 5621, val 1078, test 1630 ảnh cắt. Thêm lề 10% quanh box như GTSRB (A1, phần "Cơ sở bổ sung"). Cụ thể (ghi 2026-11-10): cắt hình vuông có tâm là tâm box, cạnh 1,2 × max(w, h), đệm bằng pixel mép khi vượt ảnh, để biển không bị méo. Lưu PNG ở kích thước gốc, đổi về 64×64 lúc nạp dữ liệu. Dùng `manifest.csv` có `class_id` cố định thay cho `ImageFolder`, vì val chỉ có 35 lớp và test 41 lớp, `ImageFolder` sẽ đánh số lớp khác nhau giữa các tập.

**Kích thước đầu vào 64×64.** Bằng chứng (đo 2026-11-10 trên ảnh gốc): cạnh dài của box có trung vị 36 px, p25 là 24 px, p75 là 58 px, p95 là 122 px. 78,1% box có cạnh dài ≤ 64 px, nên 64×64 giữ được gần hết chi tiết. Các box nhỏ (43,8% box ≤ 32 px) sẽ được phóng to, không thêm thông tin; đây là giới hạn của dữ liệu (B3).

**Quy tắc chung của hướng A:**
- Không lật ngang (D1). Tăng cường dữ liệu chỉ dùng xoay nhỏ, dịch, co giãn, đổi độ sáng.
- Mỗi bậc chạy 3 seed. Chọn trên val, test chỉ chấm một lần ở cuối cho các mạng đã chốt.
- Đo: accuracy và macro-F1 trên các lớp có mặt (val 35 lớp, test 41 lớp), confusion matrix, số tham số, thời gian train. Ghi đường cong loss và accuracy của train và val để thấy overfitting.
- Với mỗi mạng, báo cáo có **bảng từng lớp**: kích thước đầu ra (tính tay bằng `(N+2P−F)/stride+1`), số tham số (tính tay, rồi kiểm lại bằng code), receptive field. Đây đúng dạng câu hỏi trong đề thi.

| Bậc | Thay đổi so với bậc trước | Câu hỏi của môn trả lời được | Bài giảng |
|---|---|---|---|
| A0 | Mạng kiểu LeNet: 2 conv 5×5, pooling, 2 FC | Mạng tích chập cơ bản đạt bao nhiêu? Đếm tham số, kích thước đầu ra | Chương 3, LeNet-5 |
| A1 | Các khối conv 3×3 kiểu VGG, sâu hơn | Vì sao chồng nhiều 3×3 thay cho một lớp lớn? Ít tham số hơn mà receptive field như nhau | Slide VGG |
| A2 | Thêm BatchNorm | BN có bao nhiêu tham số, khác nhau thế nào giữa lúc train và lúc suy diễn, có hội tụ nhanh hơn không | 4.4 |
| A3 | Dropout và tăng cường dữ liệu | Overfitting là gì, giảm được bao nhiêu khoảng cách train và val | 4.7, 4.8 |
| A4 | Khối residual, so mạng sâu có và không có kết nối tắt | Mạng sâu bị suy biến không? Học phần dư giúp gì | Slide ResNet |
| A5 | Khởi tạo (He, Xavier), hàm kích hoạt (ReLU, LeakyReLU, Swish), optimizer và lịch lr | Thay cho Thí nghiệm 7 trên YOLO, xem ghi chú dưới | 4.1, 4.3, 4.5, 4.6 |
| ~~A6~~ | ~~ResNet-18 học trước trên ImageNet, tinh chỉnh~~ **Bỏ (2026-11-10):** cô không cho dùng mạng đã học trước | Transfer learning chỉ trả lời bằng lý thuyết | 4.10 |
| A7 | Ensemble vài mạng tốt nhất | Kết hợp mô hình tăng được bao nhiêu, đổi lại tốn gì | 4.9 |

**Thí nghiệm 7 chuyển sang A5.** Trên YOLO, Thí nghiệm 7 tốn khoảng 750–1050 epoch GPU. Trên mạng phân loại 64×64, mỗi lần train chỉ mất vài phút (chưa đo, ước lượng), nên chạy đủ 5 cấu hình × 3 seed và so được cả lịch giảm lr. Lưới cấu hình và quy tắc chọn giữ như Thí nghiệm 7 (lr của Adam 1e-3, 5e-4; SGD + momentum; lịch tuyến tính, cosin, theo bậc). YOLO giữ cấu hình tạm `optimizer=AdamW lr0=0.000179 momentum=0.9`, không chạy Thí nghiệm 7 trên YOLO nữa.

**Thí nghiệm 5 (rò rỉ) làm trên mạng phân loại trước.** Train cùng một mạng trên ảnh cắt chia ngẫu nhiên theo ảnh (split gốc) và chia theo nhóm (split hiện tại). Rẻ, mà vẫn trả lời câu H4. Bản trên YOLO thành phần làm thêm.

#### Hướng B (bổ sung): phát hiện đối tượng

| Bản | Làm gì | Câu hỏi của môn trả lời được | Bài giảng |
|---|---|---|---|
| B0 | YOLOv8s **train từ đầu** (`model=yolov8s.yaml`, không dùng trọng số COCO), 52 lớp, 3 seed. Đây là mốc so sánh và là bản G của Thí nghiệm 5 | YOLO hoạt động thế nào: lưới, dự đoán ở nhiều tỉ lệ, NMS | 6.4 |
| B1 | Hai giai đoạn: YOLO 1 lớp "biển báo" (`single_cls=True`, train từ đầu) chỉ đề xuất vùng, mạng tốt nhất của hướng A phân loại vùng. So mAP với B0 trên cùng test | Hai giai đoạn và một giai đoạn khác nhau thế nào, lỗi có dồn tầng không. Trả lời câu H2 | 6.3, 6.4 |
| B2a | Thêm đầu dò P2 (`yolov8-p2.yaml` có sẵn trong Ultralytics 8.4.191, cỡ s, train từ đầu) cho biển nhỏ | Vì sao bản đồ đặc trưng độ phân giải cao giúp phát hiện vật nhỏ (lập luận của SSD, YOLOv3) | 6.4 |
| ~~B2b~~ | ~~Train từ đầu so với từ trọng số COCO~~ **Bỏ (2026-11-10):** cô không cho dùng YOLO đã học trước, mọi bản đều train từ đầu | | |

**Câu trả lời của cô (2026-11-10): không được dùng mạng đã học trước.** Mọi mạng của cả hai hướng đều train từ đầu. Hệ quả:
- Bỏ A6 và B2b. Câu hỏi transfer learning (giáo trình 4.10) chỉ trả lời bằng lý thuyết.
- Lần chạy thử 1 epoch dùng `yolov8n.pt` (trọng số COCO) chỉ để kiểm tra quy trình, không dùng số liệu của nó.
- YOLO train từ đầu trên 2138 ảnh cần nhiều epoch hơn và có thể đạt điểm thấp hơn nhiều so với khi tinh chỉnh. Đây là kết quả cần báo cáo, không phải lỗi. Hướng B1 (dùng mạng phân loại của hướng A) có thể bù một phần.
- Hướng B còn B0, B1, B2a. Với E = 100 epoch là khoảng 900 epoch GPU. Nếu E phải lên 200–300 thì gấp 2–3 lần. Chốt E sau khi chạy thử ở giai đoạn 0.

**Chọn lr cho YOLO train từ đầu (bạn đồng ý thêm, 2026-11-10):** Cấu hình tạm của YOLO (`AdamW lr0=0.000179`) là giá trị Ultralytics tự chọn cho lần chạy ngắn, vốn thường dùng khi tinh chỉnh từ trọng số có sẵn. Train từ đầu thường cần lr lớn hơn (giáo trình 4.10 khuyên lr thấp là cho tinh chỉnh). Ở giai đoạn 0, người 2 chạy ngắn khoảng 30 epoch, 1 seed, với 3 mức: AdamW 0.000179, AdamW 0.001, SGD 0.01 momentum 0.937. Chọn mức có mAP50-95 val cao nhất rồi cố định cho mọi bản YOLO (B0, B1, B2a), ghi giá trị vào lịch sử. Không chấm test ở bước này. Tốn khoảng 90 epoch. Hạn chế: chọn từ lần chạy ngắn, 1 seed, nên phải ghi rõ trong báo cáo.

#### Các thí nghiệm cũ

| Thí nghiệm | Trạng thái mới |
|---|---|
| 1 (box nhỏ), 2 (gộp S.505a), 4 (xoá cả lớp) | Làm thêm nếu còn thời gian. Thí nghiệm 4 bản C đã có bằng chứng là tệ hơn trước khi train |
| 3 (nhóm dễ nhầm) | Chỉ phân tích confusion matrix, trên cả mạng phân loại và B0 |
| 5 (ảnh không giống train) | Làm trên mạng phân loại trước, YOLO là phần làm thêm |
| 6 (ảnh ngoài phân phối) | Giữ. Chấm cả B0 và B1 |
| 7 (optimizer) | Chuyển thành A5 |

#### Thứ tự

| Giai đoạn | Việc | Cần xong trước |
|---|---|---|
| 0. Chuẩn bị | Đóng gói dataset, notebook YOLO mẫu, chạy ngắn chọn lr cho YOLO train từ đầu (3 mức × 30 epoch), script dựng bộ ảnh cắt và kiểm tra, khung code PyTorch chung (vòng train, đo, lưu kết quả). Đo thời gian 1 epoch YOLOv8s trên GPU Kaggle/Colab. Bắt đầu thu thập ảnh cho Thí nghiệm 6 | – |
| 1 | A0–A4. B0. Thí nghiệm 5 trên mạng phân loại | 0 |
| 2 | A5, A7. B2a. YOLO 1 lớp cho B1 | 1 |
| 3 | Ghép B1 với mạng tốt nhất của A. Chấm Thí nghiệm 6. Phân tích confusion matrix (Thí nghiệm 3) | 2 |
| 4 | Viết báo cáo, điền H1–H4 | 3 |

#### Phân công (chốt tên 2026-11-10)

**Hết hiệu lực từ 2026-11-19:** nhóm chỉ còn Dũng. Phạm vi và lịch mới ở mục "Kế hoạch rút gọn khi làm một mình" ngay dưới đây và ở [`ke_hoach/dung_lam_mot_minh.md`](ke_hoach/dung_lam_mot_minh.md). Bảng dưới giữ lại làm lịch sử.

Kế hoạch chi tiết cho từng người (việc, lệnh, tiêu chí xong, bàn giao, câu hỏi phải trả lời) ở thư mục [`ke_hoach/`](ke_hoach/00_chung.md): `00_chung.md` và 4 file `nguoi_1` … `nguoi_4`. Khi kế hoạch đổi, sửa ở sổ này trước rồi sửa các file đó theo.

| Người | Việc | Phần báo cáo |
|---|---|---|
| 1 (Dũng) | Phần dữ liệu còn lại (đóng gói, tải lên, các bản biến thể nếu cần), script dựng bộ ảnh cắt và kiểm tra, khung code PyTorch chung. Làm A0–A2. Lập bảng kích thước đầu ra, số tham số, receptive field cho mọi mạng của hướng A | Chương dữ liệu (mục A–D, H1), A0–A2, bảng tham số |
| 2 (Sỹ Nguyên) | Notebook YOLO, đo thời gian epoch, chạy ngắn chọn lr cho YOLO train từ đầu. B0, B2a. Train YOLO 1 lớp cho B1. Giải thích kiến trúc YOLOv8 (backbone, neck, head, C2f, SPPF) | Hướng B phần phát hiện, kiến trúc YOLO |
| 3 (Minh Quân) | A3, A4, A5 (thay cho Thí nghiệm 7), A7. Phân tích confusion matrix của mạng phân loại | A3–A5, A7, câu hỏi về huấn luyện (chương 4), transfer learning bằng lý thuyết |
| 4 (Anh Đức) | Thu thập và gán nhãn ảnh cho Thí nghiệm 6 từ giai đoạn 0. Thí nghiệm 5 trên mạng phân loại. Ghép và chấm B1. Chấm Thí nghiệm 6 | H2, H3, H4, B1, Thí nghiệm 5, 6 |

**Quy tắc chung cho cả nhóm:** Giữ các quy tắc ở đầu mục E (phiên bản tập nhãn, 3 seed, cùng test). YOLO cài đúng `ultralytics==8.4.191`, train từ đầu bằng file `.yaml` (không dùng file `.pt` có sẵn), tham số cố định `imgsz=960 fliplr=0 optimizer=AdamW lr0=0.000179 momentum=0.9` (sẽ thay bằng mức thắng của lần chạy chọn lr ở giai đoạn 0). Không sửa `data/vn_signs_yolo`. Mỗi lần chạy ghi một dòng vào bảng kết quả chung. Ảnh tự chụp có người và biển số xe, không đăng công khai.

**Đã hỏi cô (2026-11-10):** không được dùng mạng đã học trước. Đã bỏ A6, B2b; B0, B1, B2a train từ đầu.

**Chưa biết, nhóm sẽ trả lời sau:** MSSV và thế mạnh từng người, hạn nộp báo cáo, mỗi người có tài khoản Kaggle hoặc Colab hay không.

### Kế hoạch rút gọn khi làm một mình (2026-11-19)

**Vì sao đổi:** nhóm chỉ còn Dũng, hạn còn 2–3 tuần. Chi tiết từng bước, lệnh và tiêu chí xong ở [`ke_hoach/dung_lam_mot_minh.md`](ke_hoach/dung_lam_mot_minh.md).

**Nguyên tắc:**
- Giữ hướng A, vì đây đúng phần "tự xây mạng" mà cô chấm nặng, và rẻ: 1–2 giây mỗi epoch trên M2.
- Cắt hướng B: YOLO tạo từ `yolov8s.yaml` của Ultralytics, và tốn 551 giây mỗi epoch trên M2.
- Giữ các câu H1–H4, trả lời bằng thí nghiệm rẻ nhất có số liệu.

| Việc | Trạng thái mới |
|---|---|
| A0–A4 | Giữ |
| A5 | Rút gọn còn optimizer và lịch lr. Khởi tạo và hàm kích hoạt là phần làm thêm |
| A7 | Giữ, vì chỉ dùng lại các lần chạy đã có |
| B0 | Làm thêm, 1 seed, `optimizer=SGD lr0=0.01 momentum=0.937`. Chỉ làm nếu 100 epoch vừa một phiên Kaggle (đo 1 epoch trước) |
| B1, B2a, chọn lr cho YOLO | Bỏ |
| Thí nghiệm 3, 5 | Giữ |
| Thí nghiệm 6 | Bỏ. H3 trả lời bằng lý thuyết, cộng một phép thử: vùng nền không chứa biển |
| Thí nghiệm 1, 2, 4 | Bỏ |

**Hạn chế phải ghi trong báo cáo:**
- Không có hệ hai giai đoạn, nên H2 không có số liệu.
- Không có ảnh tự chụp, nên H3 không có số đo trên ảnh thật ngoài phân phối.
- B0, nếu làm, không tìm lr và chỉ chạy 1 seed.

### Chạy thử 1 epoch (2026-11-09) ✅
**Mục đích:** Kiểm tra Ultralytics đọc đúng dataset và chạy hết quy trình train, chấm val, chấm test. Không dùng để đánh giá chất lượng model.

**Môi trường:** MacBook M2, RAM 16 GB, GPU qua MPS. Python 3.14.7, PyTorch 2.14.1, Ultralytics 8.4.191 (đã có bản 8.4.208, nhưng giữ 8.4.191 cho mọi thí nghiệm để kết quả tái lập được). Bạn tự chạy trong Terminal, kết quả ở `data/runs/`.

```bash
yolo detect train data=<đường dẫn tuyệt đối>/data/vn_signs_yolo/data.yaml model=yolov8n.pt epochs=1 imgsz=960 batch=8 fliplr=0 device=mps workers=4 project=<...>/data/runs name=sanity_1epoch
yolo detect val model=<...>/data/runs/sanity_1epoch/weights/best.pt data=<...>/data.yaml split=test imgsz=960 batch=8 device=mps project=<...>/data/runs name=sanity_test
```

**Ultralytics đọc dữ liệu** (số lấy từ file `labels/{train,val,test}.cache` do Ultralytics tạo):

| Tập | Ảnh | Thiếu nhãn | Nhãn rỗng | Hỏng | Cảnh báo | Box | Số lớp có box |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2138 | 0 | 0 | 0 | 0 | 5621 | |
| val | 474 | 0 | 0 | 0 | 0 | 1078 | 35 |
| test | 579 | 0 | 0 | 0 | 0 | 1630 | 41 |

- Khớp `verify_dataset.py`: 3191 ảnh, 8329 box. Không ảnh nào bị Ultralytics loại.
- Nhận đúng 52 lớp (`Overriding model.yaml nc=80 with nc=52`). Tên tiếng Việt hiển thị đúng trong bảng kết quả.
- Test có 41 lớp, đúng như dự kiến (52 trừ 11 lớp không đánh giá được). File `splits/class_counts.csv` còn ghi 42 lớp vì được tạo trước khi chuyển 1193#2 sang lớp 30.

**Thời gian và bộ nhớ:** 1 epoch train mất 551 giây (268 lượt × 1,9 giây, batch 8), chấm val 25 giây, chấm test 28 giây. GPU dùng 5,28 GB. Ước tính thô: 100 epoch khoảng 16 giờ cho mỗi lần chạy trên M2. Kế hoạch mục E có hàng chục lần chạy, nên train thật vẫn cần Colab/Kaggle; M2 dùng cho các lần thử ngắn.

**Kết quả sau 1 epoch** (chỉ để biết quy trình chạy được):

| Tập | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0,460 | 0,302 | 0,235 | 0,162 |
| test | 0,433 | 0,241 | 0,161 | 0,114 |

**Phát hiện cần lưu ý cho lần train thật:**
1. **Optimizer tự chọn.** Với `optimizer=auto`, Ultralytics chọn AdamW với `lr=0.000179` và bỏ qua `lr0=0.01`. Theo mã nguồn, lr này tính từ số lớp (`0.002 × 5 / (4 + nc)`), và AdamW được chọn khi tổng số bước dưới 10 000 (trên mức này là MuSGD, `lr=0.01`). Đã đối chiếu với `ultralytics/engine/trainer.py` bản 8.4.191. Mức ảnh hưởng nếu để `auto`: gộp S.505a (52 xuống 50 lớp) chỉ làm lr tăng 3,4% (0,000185), gần như không đáng kể; gộp còn khoảng 10 lớp thì lr tăng gấp 4; train từ 295 epoch trở lên thì đổi hẳn sang MuSGD `lr=0.01`; đổi phiên bản Ultralytics thì quy tắc có thể khác (bản cũ dùng SGD). Lý do cố định là để cấu hình không tự đổi và báo cáo ghi được giá trị cụ thể. **Quyết định (2026-11-09):** đặt cố định `optimizer` và `lr0` cho mọi thí nghiệm. Giá trị cố định được chọn bằng Thí nghiệm 7. **Trong lúc chờ Thí nghiệm 7** (bạn chọn "tạm dùng"): dùng `optimizer=AdamW lr0=0.000179 momentum=0.9`, tức cấu hình 1 (mốc) của Thí nghiệm 7. Phải ghi cả `momentum=0.9`, vì nếu chỉ ghi `optimizer=AdamW` thì Ultralytics lấy mặc định 0,937. Khi Thí nghiệm 7 có cấu hình thắng thì thay giá trị này và ghi vào lịch sử. **Cập nhật 2026-11-10:** Thí nghiệm 7 chuyển sang mạng phân loại (A5), nên YOLO dùng cấu hình này cố định cho mọi lần chạy. **Cập nhật tiếp 2026-11-10:** vì phải train từ đầu, giá trị này sẽ được chọn lại bằng lần chạy ngắn ở giai đoạn 0 (xem "Kế hoạch chính").
2. **Mosaic bị tắt cả lần chạy thử:** `close_mosaic=10` tắt mosaic ở 10 epoch cuối, mà lần này chỉ có 1 epoch. Train thật thì không bị như vậy.
3. **Không tái lập tuyệt đối trên MPS:** PyTorch cảnh báo hai phép tính (`scatter_reduce_mps`, `index_put_with_accumulate_mps`) không có bản tất định. Cùng seed vẫn có thể ra số hơi khác. Đây thêm một lý do để mỗi cấu hình chạy 3 seed.
4. **Không dùng worker phụ trên MPS:** Log ghi `Using 0 dataloader workers` dù đặt `workers=4`, vì `ultralytics/engine/trainer.py` đặt `workers = 0` khi chạy trên CPU hoặc MPS. Đọc ảnh chạy chung luồng với train, đây là một phần lý do chậm.
5. **File `.cache` trong dataset:** Ultralytics tạo `labels/train.cache`, `val.cache`, `test.cache`. Khi tính md5 thư mục `labels` theo quy tắc ở đầu mục E, phải bỏ qua file `.cache`. Khi đóng gói zip cũng nên bỏ chúng.
6. **Cách đọc bảng từng lớp:** Lớp có P = 1 và R = 0 nghĩa là model chưa dự đoán box nào cho lớp đó. Ultralytics quy ước như vậy, không phải model đoán đúng hết.
7. Ultralytics tải `yolov8n.pt` (6,2 MB) vào `data/runs/` và tải một font Unicode (22 MB) để vẽ chữ tiếng Việt, nên lần chạy đầu cần mạng.

### Thí nghiệm 4: xoá cả lớp (thêm 2026-11-09)

**Yêu cầu của bạn:** Thử xoá một số lớp xem có vấn đề gì không, báo cáo kết quả và hệ quả. Bạn chốt là xoá cả lớp, không phải xoá từng box.

**Lớp chọn xoá:** 11 lớp không đánh giá được (xem A1): 15, 21, 24, 27, 28, 29, 42, 44, 47, 48, 51. Lý do chọn: các lớp này không có box ở val và test, nên xoá chúng không làm đổi tập test, và mọi bản được chấm trên cùng 41 lớp.

**Phạm vi ảnh hưởng** (đo 2026-11-09 trên `data/vn_signs_yolo`):
- 11 lớp có 289 box, nằm trong 289 ảnh train (5,1% số box train, 13,5% số ảnh train).
- 66 ảnh chỉ chứa các lớp này. Nếu xoá nhãn mà giữ ảnh, chúng thành ảnh không có nhãn nào.
- 223 ảnh còn lại chứa thêm **895 box của lớp khác**.

| Lớp | Tên | Box train |
|---|---|---:|
| 15 | Đường bị thu hẹp về phía phải | 29 |
| 21 | Đường bị thu hẹp về phía trái | 11 |
| 24 | Kiểm tra | 56 |
| 27 | Trẻ em | 36 |
| 28 | Xe tải và xe công* | 26 |
| 29 | Cấm mô tô và xe máy | 45 |
| 42 | Các xe chỉ được rẽ trái | 21 |
| 44 | Nguy hiểm khác | 3 |
| 47 | Cấm ô tô quay đầu xe (được rẽ trái) | 16 |
| 48 | Giao nhau với đường sắt có rào chắn | 41 |
| 51 | Chú ý chướng ngại vật, vòng tránh sang bên phải | 5 |

**Các bản so sánh:**

| Bản | Làm gì | Trả lời câu hỏi |
|---|---|---|
| A | Dataset hiện tại, 52 lớp | Mốc so sánh |
| B | Xoá dòng nhãn của 11 lớp, **giữ ảnh** | Biển có thật nhưng không có nhãn thì model học thành nền. Có làm hại các lớp giống chúng không? |
| C | Xoá **cả ảnh** chứa 11 lớp | Tránh được vấn đề của B, nhưng mất bao nhiêu dữ liệu? |

**Bằng chứng có trước khi train (bản C):** Bỏ 289 ảnh làm mất 895 box của 41 lớp còn lại, gấp 3 lần số box định xoá. 13 lớp mất từ 20% box train trở lên, trong đó:
- Lớp 20 (Cấm ô tô) mất 30/30 và lớp 43 mất 5/5: **mất toàn bộ dữ liệu train**, trong khi lớp 20 vẫn có box ở test.
- Lớp 22 mất 90/100, lớp 38 mất 82/100, lớp 16 mất 13/18.

Vì vậy bản C gần như chắc chắn tệ hơn. Vẫn nên chạy để có số liệu cho báo cáo, nhưng đây không phải phương án dùng thật.

**Cách làm kỹ thuật:** Giữ nguyên 52 tên lớp trong `data.yaml`, chỉ xoá dòng nhãn, không đánh số lại id. Lý do: đánh số lại dễ gây lỗi lệch nhãn, và Ultralytics chỉ tính mAP trên các lớp có box trong tập đánh giá, nên cả ba bản đều được chấm trên cùng 41 lớp. Mỗi bản là một thư mục dataset riêng, không sửa `data/vn_signs_yolo`.

**Đo gì:**
1. mAP50, mAP50-95, precision, recall trên test (41 lớp). Mỗi bản 3 seed.
2. AP từng lớp, chú ý các lớp gần giống lớp bị xoá:
   - 29 (cấm mô tô) với 23 (cấm xe hai và ba bánh)
   - 47 (cấm ô tô quay đầu) với 17, 19, 49 (các biển cấm quay đầu)
   - 42 (chỉ được rẽ trái) với 3 (đi vòng sang phải)
   - 28 với 30 (biển phụ S.505a)
   - 51 với 26 (W.246)
3. **Hệ quả khi dùng thật:** Không đo được trên test vì test không có 11 lớp này. Cách đo: chạy model A và B trên chính các biển của 11 lớp (lấy từ ảnh train) và ghi lại model dự đoán gì. Model A đã học các ảnh này nên chỉ dùng để tham khảo. Câu hỏi chính là model B bỏ qua biển, hay nhận nhầm thành lớp gần giống (ví dụ thấy biển cấm mô tô mà báo là cấm xe hai và ba bánh).

**Dự đoán ban đầu (chưa phải kết quả):**
- B: mAP trên 41 lớp thay đổi ít, vì chỉ 5% box train bị ảnh hưởng. Các lớp gần giống có thể giảm recall, vì biển trông gần như nhau nhưng một bên có nhãn, một bên bị dạy là nền.
- C: mAP giảm rõ, đặc biệt các lớp 20, 22, 38, 43.
- Khi dùng thật: 11 loại biển này sẽ không bao giờ được phát hiện, hoặc bị báo nhầm thành lớp khác. Đây là hệ quả cần nêu rõ trong báo cáo, dù con số mAP có thể không đổi nhiều.

### Thí nghiệm 5: ảnh mới không giống ảnh train (thêm 2026-11-09, trả lời câu H4)

**Ý tưởng:** So cùng một cấu hình model trên hai kiểu chia tập. Khoảng chênh cho biết điểm bị thổi phồng bao nhiêu khi test giống train.

| Bản | Train trên | Đánh giá trên | Độ giống test với train (median) |
|---|---|---|---|
| R | Split gốc (ngẫu nhiên từng ảnh) | Test gốc | 0,921 |
| G | Split theo nhóm (dataset hiện tại) | Test hiện tại | 0,730 |

**Đo:** mAP50, mAP50-95 trên 41 lớp chung (bỏ 11 lớp không có ở test của bản G khi tính cho R, để cùng tập nhãn). Ghi đường cong train loss và val loss của cả hai.

**Đọc kết quả:** Nếu R cao hơn G nhiều, phần chênh là do model nhớ cảnh đã thấy, không phải do nhận biết biển tốt hơn.

### Thí nghiệm 6: ảnh nằm ngoài phân phối (thêm 2026-11-09, trả lời câu H3)

**Ý tưởng:** Lập một tập test nhỏ khác hẳn dữ liệu train, gán nhãn bằng cùng 52 lớp, rồi đánh giá model G trên đó.

Nguồn gợi ý, chưa thu thập:
- Ảnh tự chụp bằng điện thoại ở quận khác, ban đêm hoặc trời mưa.
- Một phần dataset Zalo 2020 (Google Street View), sau khi ánh xạ lớp.

**Đo:** mAP trên tập này so với test G, số phát hiện nhầm trên ảnh không có biển, và độ tin cậy của các phát hiện nhầm đó.

### Thí nghiệm 7: chọn optimizer và lr0 (thêm 2026-11-09)

**Cập nhật 2026-11-10:** Chuyển sang mạng phân loại tự xây (bậc A5, xem "Kế hoạch chính" ở đầu mục E). Không chạy trên YOLO nữa; YOLO giữ cấu hình tạm. Phần dưới giữ làm cơ sở lý thuyết và quy tắc chọn cho A5.

**Yêu cầu của bạn:** Lý thuyết của môn nói optimizer và tốc độ học là siêu tham số phải chọn, vậy làm thành một thí nghiệm để quyết định giá trị cố định (xem lưu ý 1 của phần chạy thử).

**Cơ sở lý thuyết** (môn Học sâu và ứng dụng IT4653: slide bài 5 `l5_training_neural_networks_part_2`, slide 22, 27–33, 51–53; giáo trình `dl_lecturenotes` mục 4.5, 4.6, 4.10):
- Adam với beta1 = 0,9, beta2 = 0,999, lr = 1e-3 hoặc 5e-4 là mặc định tốt cho nhiều mô hình. Adam là lựa chọn mặc định tốt. SGD + momentum thường tốt hơn Adam nhưng phải tinh chỉnh lr và chọn lịch giảm lr hợp lý.
- Tốc độ học là một trong những siêu tham số quan trọng nhất. Lr quá lớn thì loss dao động hoặc vượt qua cực tiểu, quá nhỏ thì hội tụ chậm. Bốn lịch giảm lr: theo bậc, cosin, tuyến tính, tỉ lệ nghịch căn bậc hai số epoch.
- Optimizer, lr và lịch giảm lr là siêu tham số, chọn trên tập val. Slide khuyên random search hơn grid search. Ở đây dùng lưới nhỏ vì mỗi lần chạy tốn nhiều giờ GPU.
- Tinh chỉnh từ mô hình có sẵn nên dùng lr thấp (4.10).

**Thứ tự:** Chạy **trước** Thí nghiệm 1–6, vì các thí nghiệm đó cần cấu hình đã cố định. Việc của người 2 (huấn luyện YOLO), xem phân công ở đầu mục E.

**Giữ cố định:** split seed 0, `data/vn_signs_yolo` (52 lớp), YOLOv8s từ trọng số COCO, `imgsz=960`, `fliplr=0`, cùng `batch`, `nbs=64`, `weight_decay=0.0005`, `warmup_epochs=3`, Ultralytics 8.4.191.

**Giai đoạn 1: lọc thô** (1 seed, 30 epoch):

| # | `optimizer` | `lr0` | `momentum` | Lịch giảm lr | Lấy từ đâu |
|---|---|---:|---:|---|---|
| 1 | `auto` (thành AdamW) | 0,000179, tự tính | 0,9 | tuyến tính | Mốc: cái Ultralytics tự chọn |
| 2 | AdamW | 0,001 | 0,9 | tuyến tính | Mặc định trong slide |
| 3 | AdamW | 0,0005 | 0,9 | tuyến tính | Mặc định trong slide |
| 4 | SGD | 0,01 | 0,937 | tuyến tính | Mặc định của Ultralytics |
| 5 | SGD | 0,01 | 0,937 | cosin (`cos_lr=True`) | Slide: SGD cần lịch giảm lr hợp lý |

Giữ lại 2 cấu hình có mAP50-95 trên val cao nhất.

**Giai đoạn 2: xác nhận** (3 seed, đủ số epoch): Chạy 2 cấu hình còn lại. Nếu cấu hình thắng là AdamW, thêm một bản AdamW với `cos_lr=True` để so lịch giảm lr.

**Quy tắc chọn (đặt trước khi chạy):**
- Chọn cấu hình có trung bình mAP50-95 trên val cao nhất qua 3 seed.
- Nếu chênh lệch nhỏ hơn độ dao động giữa các seed, coi như hoà và chọn cấu hình có độ lệch chuẩn nhỏ hơn.
- **Không chấm test** trong thí nghiệm này. Test chỉ dùng cho Thí nghiệm 1–6.
- Sau khi chốt, ghi `optimizer`, `lr0`, `momentum`, `cos_lr` bằng giá trị cụ thể cho mọi thí nghiệm, kể cả khi cấu hình 1 thắng. Lý do: Thí nghiệm 2 và 4 đổi số lớp, để `auto` thì lr tự đổi theo.

**Đo gì:**
1. mAP50, mAP50-95 trên val: trung bình và độ lệch chuẩn qua 3 seed.
2. Đường cong train loss, val loss và lr theo epoch của từng cấu hình. Đây là phần nối với lý thuyết 4.5 (lr lớn thì dao động, lr nhỏ thì giảm chậm).
3. Epoch đạt mAP val tốt nhất, để so tốc độ hội tụ.

**Câu hỏi trả lời trong báo cáo:**
- Câu "SGD + momentum tốt hơn Adam nhưng khó chỉnh" có đúng với bài toán này không?
- Lr 1e-3 của slide có quá lớn khi fine-tune từ trọng số COCO không (so cấu hình 1 và 2)?
- Lịch cosin hay tuyến tính tốt hơn?

**Chi tiết kỹ thuật đã đối chiếu mã nguồn** (`ultralytics/engine/trainer.py` và `cfg/default.yaml`, bản 8.4.191):
- `momentum` được dùng làm **beta1 của Adam/AdamW**. Mặc định là 0,937, nên các cấu hình AdamW phải đặt `momentum=0.9` thì mới đúng beta1 trong slide.
- `auto` chọn AdamW khi số bước tối ưu ≤ 10 000, ngược lại chọn MuSGD với `lr=0.01`. Số bước tính bằng `ceil(số ảnh train / max(batch, nbs)) × epochs` = 34 × epochs. Vậy với 2138 ảnh train, `auto` vẫn là AdamW tới 294 epoch, nên cấu hình 1 giữ nguyên AdamW ở cả hai giai đoạn.
- Lịch mặc định là giảm tuyến tính từ `lr0` về `lr0 × lrf` (`lrf=0.01`). `cos_lr=True` đổi thành cosin.

**Lưu ý:**
- Thứ hạng ở 30 epoch có thể khác ở 100 epoch, nhất là với lr nhỏ, vốn hội tụ chậm. Vì vậy cần giai đoạn 2.
- Chi phí: giai đoạn 1 là 150 epoch, giai đoạn 2 khoảng 600–900 epoch. Trên M2 (9 phút/epoch) không khả thi, phải chạy trên Colab hoặc Kaggle. Đo thời gian 1 epoch trên T4 trước rồi mới chốt số epoch.
- Nếu không đủ thời gian: chỉ chạy giai đoạn 1, chọn cấu hình đứng đầu và cố định luôn. Khi đó phải ghi trong báo cáo là cấu hình được chọn từ lần chạy ngắn, 1 seed.

**Chưa chạy thí nghiệm nào.**

## G. Box "Không rõ" trong catalog

**Quyết định (2026-11-09):** Giữ nguyên nhãn 80 box bạn đánh dấu Không rõ. Xoá nhãn sẽ biến biển thật thành nền.

| Lớp | Số box | Ảnh |
|---|---:|---|
| 26 | 24 | 0961–0981, 1007, 1120#0, 2438 |
| 16 | 26 | 0551–0571, 0797–0800, 0814 |
| 14 | 13 | 0755#8, 3130–3141 (một box mỗi ảnh) |
| 43 | 10 | 0571#6–0578#6, 0580#6, 0592#0 |
| 34 | 3 | 0687–0689 |
| 17 | 2 | 1017#4, 1018#4 |
| 51 | 2 | 0530#5, 0531#5 |

Phân loại theo lớp đã đối chiếu với nhãn trước khi sửa (thư mục sao lưu). Danh sách này dùng cho Thí nghiệm 1 (box nhỏ): có thể so AP của nhóm box này với phần còn lại.

## H. Câu hỏi cần trả lời trong báo cáo (ghi 2026-11-09)

Bạn ghi lại để trả lời khi báo cáo với cô. Phần trả lời là lập luận ban đầu, cần số liệu từ các thí nghiệm ở mục E để chốt.

### H1. Thí nghiệm cần lưu ý gì về tập nhãn?
- Tập nhãn quyết định mAP đo cái gì. Ultralytics chỉ lấy trung bình trên các lớp có box trong tập đánh giá, nên test hiện tại chỉ đo 41/52 lớp.
- Đổi tập nhãn (gộp lớp ở C5, xoá lớp ở Thí nghiệm 4, sửa box ở B1, B2) thì phải ghi phiên bản mới. Không so điểm giữa hai phiên bản khác nhau trừ khi quy về cùng tập lớp.
- Nhãn sai hoặc sót ở test làm điểm đo sai mà model không đổi (xem B4).
- Đã ghi thành quy tắc ở đầu mục E.

### H2. Phân đoạn ngữ nghĩa trước rồi phân loại, hoặc ngược lại, thì sao?
Mô hình hiện tại (YOLO) làm cả hai việc cùng lúc: một lần chạy cho ra vị trí box và lớp.

**Phân đoạn trước, phân loại sau (hai giai đoạn):** Bước 1 tìm vùng ảnh là biển báo (không cần biết loại). Bước 2 cắt từng vùng ra và đưa qua mô hình phân loại 52 lớp.
- Ưu: Bộ phân loại nhìn vùng cắt ở độ phân giải cao, có lợi cho biển nhỏ và các lớp dễ nhầm (C6, D2). Thêm lớp mới chỉ cần huấn luyện lại bộ phân loại.
- Nhược:
  - Lỗi dồn tầng: bước 1 bỏ sót biển nào thì bước 2 không bao giờ thấy biển đó.
  - Dataset chỉ có box, không có mặt nạ pixel. Phải gán thêm hoặc dùng box làm mặt nạ thô.
  - Phân đoạn ngữ nghĩa không tách các biển dính nhau (biển chồng trên cùng cột như 1924, biển phụ sát biển chính ở C5), nên cần phân đoạn theo từng đối tượng (instance segmentation).
  - Chạy hai mô hình nên chậm hơn.

**Phân loại trước, phân đoạn sau:** Bước 1 xác định ảnh có những loại biển nào (đa nhãn trên cả ảnh). Bước 2 mới tìm vị trí.
- Không hợp với dữ liệu này: mỗi ảnh có trung bình 2,6 biển (tối đa 12) và biển chỉ rộng khoảng 2,9% ảnh. Ở độ phân giải cả ảnh, biển gần như biến mất nên bước 1 dễ sót.
- Chỉ hữu ích như một bộ lọc nhanh "ảnh có biển hay không", để bỏ qua các khung hình trống khi xử lý video.

**Cách trả lời bằng số liệu:** Có thể thêm một thí nghiệm: dùng YOLO một lớp "biển báo" để tìm vùng, rồi phân loại vùng cắt, so với YOLO 52 lớp trên cùng test. Chưa đưa vào mục E vì cần thêm một bộ phân loại.

### H3. Gặp ảnh nằm ngoài phân phối (out-of-distribution) thì sao?
- YOLO không có lớp "không biết". Với mọi ảnh, nó vẫn chấm điểm 52 lớp cho từng vùng, nên có thể báo nhầm với độ tin cậy cao. Ví dụ rủi ro: biển hiệu cửa hàng tròn màu đỏ bị nhận thành biển cấm.
- Với ảnh khác điều kiện (ban đêm, mưa, camera khác, thành phố khác), thường sẽ sót biển nhiều hơn và độ tin cậy thấp hơn.
- Dữ liệu hiện tại: quay ở TP.HCM bằng một loại camera. Chưa kiểm tra có ảnh ban đêm không. Mọi ảnh test đều cùng nguồn với train, nên **điểm test hiện tại không nói gì về ảnh ngoài phân phối**.
- Cách đo: Thí nghiệm 6.

### H4. Có tập train rồi, nhưng ảnh mới không giống lắm thì kết quả thế nào? Có phải overfitting không?
- Kết quả sẽ thấp hơn điểm test. Mức giảm tùy ảnh mới khác bao nhiêu.
- Cần phân biệt hai hiện tượng:
  - **Overfitting:** model nhớ dữ liệu train, nên điểm trên train cao nhưng trên dữ liệu chưa thấy (cùng nguồn) thấp. Dấu hiệu: val loss tăng trong khi train loss vẫn giảm.
  - **Lệch phân phối (domain shift):** model không nhất thiết overfit, nhưng ảnh mới khác dữ liệu đã học (H3).
- **Bằng chứng có sẵn từ model của đề tài cũ** (`YOLOV8/*/results.csv` ở nhánh `de-tai-cu`, chỉ còn số liệu epoch 72–100):

| Model | Val cls_loss (epoch 72 → 100) | mAP50 val | mAP50-95 val |
|---|---|---:|---:|
| YOLOv8n | 0,306 → 0,291 | 0,987 | 0,871 |
| YOLOv8s | 0,259 → 0,251 | 0,989 | 0,894 |
| YOLOv8m | 0,244 → 0,241 | 0,990 | 0,901 |

  Val loss không tăng, nên nhìn đường cong thì không thấy overfitting. Nhưng đây là bộ dữ liệu 46 lớp khác và val của nó có thể cũng bị rò rỉ như split gốc của `archive` (30,7% ảnh test gần trùng ảnh train). Nếu val giống train, đường cong không thể lộ ra việc model nhớ cảnh. Các con số trong `results.csv` cũng khác bảng trong README cũ (YOLOv8m: 0,990/0,901 so với 0,984/0,839), chưa rõ vì sao.
- **Cách trả lời bằng số liệu:** Thí nghiệm 5 đo trực tiếp điểm giảm bao nhiêu khi test không giống train. Thí nghiệm 6 đo với ảnh khác hẳn.

## F. Câu hỏi còn mở

1. ✅ **Dataset dùng cho mục đích gì?** Đề tài môn Deep Learning ở HUST (bạn trả lời 2026-11-09). Hệ quả: ưu tiên thí nghiệm sạch, giải thích được, đủ bằng chứng cho báo cáo. Không cần tối ưu tốc độ triển khai.
2. ✅ **Có video gốc không?** Không (bạn trả lời 2026-11-09). Giữ cách chia suy từ độ giống ảnh (A1). Theo yêu cầu của bạn, đã tìm thêm nguồn làm cơ sở, ghi ở A1 phần "Cơ sở bổ sung".
3. ✅ **Giữ đủ 52 lớp hay gộp lại?** Bạn hỏi vì sao 11 lớp không có box ở val và test.

   **Trả lời:** Mỗi lớp trong 11 lớp này chỉ xuất hiện trong **đúng một nhóm frame**, tức một đoạn quay liên tục. Một nhóm phải nằm trọn trong một tập, nếu không ảnh gần như giống hệt sẽ có ở cả train và test (rò rỉ, mục A1). Nên toàn bộ box của lớp chỉ ở được một tập. Thuật toán chia đặt các nhóm này vào train để model còn học được lớp đó. Đặt vào test thì model không bao giờ thấy lớp này khi học, AP gần như chắc chắn bằng 0 và không nói lên điều gì.

   | Lớp | Tên | Box | Dải ảnh | Số ảnh trong nhóm |
   |---:|---|---:|---|---:|
   | 15 | Đường bị thu hẹp về phía phải | 29 | 0435–0463 | 29 |
   | 21 | Đường bị thu hẹp về phía trái | 11 | 0730–0740 | 11 |
   | 24 | Kiểm tra | 56 | 0751–0809 | 66 |
   | 27 | Trẻ em | 36 | 1064–1099 | 98 |
   | 28 | Xe tải và xe công* | 26 | 2888–2913 | 54 |
   | 29 | Cấm mô tô và xe máy | 45 | 1516–1702 | 215 |
   | 42 | Các xe chỉ được rẽ trái | 21 | 0357–0377 | 21 |
   | 44 | Nguy hiểm khác | 3 | 0670–0672 | 10 |
   | 47 | Cấm ô tô quay đầu xe (được rẽ trái) | 16 | 3175–3190 | 25 |
   | 48 | Giao nhau với đường sắt có rào chắn | 41 | 3085–3125 | 45 |
   | 51 | Chú ý chướng ngại vật, vòng tránh sang bên phải | 5 | 0530–0534 | 37 |

   Số liệu tính trên `data/vn_signs_yolo` hiện tại và `splits/groups.csv` (2026-11-09). Nhiều khả năng mỗi lớp là một biển thật duy nhất được quay khi xe đi qua.

   - Trong 52 lớp: 11 lớp có 1 nhóm, 4 lớp có 2 nhóm (11, 20, 33, 43: có ở train và test, thiếu val), còn lại từ 3 nhóm trở lên. Lớp 38 và 50 có từ 3 nhóm nhưng vẫn thiếu val hoặc test, do thuật toán phải cân với tỉ lệ ảnh và box.
   - Trong split gốc (chia ngẫu nhiên từng ảnh), các lớp này có mặt ở test, nhưng chỉ vì frame kề nhau rơi vào hai tập. Điểm đo được khi đó là đo trên ảnh gần như đã thấy.
   - Tiền lệ: GTSRB loại hẳn lớp có dưới 9 track (A1, phần "Cơ sở bổ sung").
   - Muốn đánh giá được các lớp này phải có thêm ảnh từ đoạn quay khác. Cách khác là cắt đôi đoạn quay, bỏ một khoảng frame ở giữa làm vùng đệm, nhưng hai nửa vẫn là cùng một biển nên điểm vẫn bị thổi phồng.

   **Quyết định của bạn (2026-11-09): Giữ, ghi rõ.**
   - Giữ đủ 52 lớp. 11 lớp này vẫn nằm trong train để model học, không gộp, không xoá.
   - Mọi kết quả trong báo cáo ghi rõ: mAP trên test chỉ tính trên 41 lớp có box; 11 lớp còn lại chưa đánh giá được vì mỗi lớp chỉ có một đoạn quay, kèm bảng trên làm bằng chứng.
   - Thí nghiệm 4 (xoá cả lớp) vẫn giữ trong kế hoạch như một thí nghiệm phụ, để trả lời "xoá nhãn thì có hệ quả gì", không thay đổi dataset chính.
