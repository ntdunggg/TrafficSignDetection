# Nhật ký rà soát dataset `archive` (Vietnamese Traffic Signs, YOLO)

Đây là bản ghi duy nhất cho từng ghi chú, quyết định và thay đổi, dùng để viết báo cáo cùng sản phẩm.

- Mỗi mục gồm: **Ghi chú của bạn**, **Trả lời / bằng chứng**, **Quyết định**, **Thay đổi**, **Trạng thái**.
- Mức độ: 🔴 nặng, 🟠 vừa, 🟡 nhẹ. Trạng thái: ✅ xong, 📝 đã ghi để làm sau, ⏳ chờ bạn.

## Tóm tắt hiện trạng (cập nhật 2026-11-09)

| Thứ | Vị trí | Ghi chú |
|---|---|---|
| Dữ liệu gốc | `/Users/tdu/Downloads/archive` | Giữ nguyên, 6437 file khớp md5 (kiểm lại 2026-11-09) |
| Dataset làm việc | `data/vn_signs_yolo/` | 3191 ảnh, 8329 box, 52 lớp, tên tiếng Việt; git ignore |
| Catalog xem và sửa | `catalog/catalog.html` | 7,2 MB; git ignore. Dựng trước đợt sửa 2026-11-09 nên vẫn hiện ảnh chưa xoay và 3 box đã xoá |
| Kết quả catalog | `data/_build/catalog_ket_qua_2026-11-09.txt` | 683 box, dạng `khóa trạng_thái` (d đúng, k không rõ, s sai) |
| Sao lưu trước đợt sửa | `data/_build/backup_truoc_catalog_2026-11-09/` | Nhãn, ảnh, data.yaml trước khi sửa |
| Script, log, bằng chứng | `data/_build/` | git ignore |

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

**2026-11-09**
- Nhận kết quả catalog: đã xem đủ 683/683 box (599 Đúng, 80 Không rõ, 4 Sai). Chi tiết ở mục G.
- Xoay lại 39 ảnh 1319–1357, xoá 3 box sai lớp 16, đổi tên lớp 26 (bản copy). Box giảm từ 8332 còn 8329.
- Chuyển box 1193#2 từ lớp 28 sang lớp 30 (bạn xác nhận). Lớp 28 không còn ở val và test, số lớp không đánh giá được tăng từ 10 lên 11.
- Thêm Thí nghiệm 4 vào kế hoạch (mục E): xoá cả 11 lớp không đánh giá được, so sánh giữ ảnh và bỏ ảnh.
- Thêm mục A1b: cách xáo trộn khi nguồn là video, tên phương pháp, và đo độ ổn định của cách chia với 5 seed (thêm chế độ `seeds` vào `make_split.py`, không thay split đang dùng).
- Ghi 4 câu hỏi cần trả lời trong báo cáo (mục H). Thêm quy tắc ghi phiên bản tập nhãn, Thí nghiệm 5 (ảnh không giống train) và 6 (ảnh ngoài phân phối).

## Đính chính

Các điều tôi từng nói sai hoặc suy luận chưa đúng. Không dùng các nhận định cũ này trong báo cáo.

| Từng nói | Thực tế | Bằng chứng |
|---|---|---|
| W.246a là tránh sang phải, W.246c là tránh hai bên, nên hai lớp có thể gán sai tên (C3) | Ngược lại: W.246a là tránh hai bên, W.246c là tránh sang phải. Hai mã có vẻ đúng | [W.246a](https://hoatieu.vn/bieu-mau/bien-bao-chu-y-chuong-ngai-vat-w246a-168207), [W.246c](https://hoatieu.vn/bieu-mau/bien-bao-chu-y-chuong-ngai-vat-tranh-sang-ben-phai-w246c-168213), `data/_build/bang_chung_w246.jpg` |
| Dùng hardlink thì ảnh gốc sẽ bị đổi nếu Ultralytics ghi đè JPEG lỗi (A4) | Sai với bản hiện tại: Ultralytics ghi file tạm rồi `os.replace`, nên hardlink không bị đổi. Dữ liệu cũng có 0 JPEG lỗi | [`ultralytics/data/utils.py`](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/data/utils.py), `verify_log.txt` |
| Biển xa không có nhãn có thể do quy tắc 0–20 m của README (A3) | Không áp dụng: `archive` có gán cả biển rất xa (0562, 2668) và biển bị cắt ở mép (1318). README có vẻ mô tả bộ 46 lớp khác | 24 ảnh mẫu ở mục B4 |

---

## A. Chia tập dữ liệu

### 🔴 A1. Rò rỉ dữ liệu giữa train và test ✅
**Ghi chú của bạn:** Hỏi có bài báo nào về cách chia dữ liệu và cách sửa. Sau đó: chọn bài có lời giải rõ ràng (ok), dùng cách tôi thấy tốt nhất.

**Vấn đề:** Ảnh đánh số liền nhau là frame từ video. Split gốc chia ngẫu nhiên từng ảnh, nên 612/639 ảnh test có ảnh train số liền kề.

**Tài liệu chọn:**
- [Find the Leak, Fix the Split](https://arxiv.org/html/2511.13944v2) (Glazner và cộng sự, preprint, 2026): trích đặc trưng frame, giảm chiều bằng PaCMAP, gom cụm bằng HDBSCAN, gán nguyên cụm vào một tập. Bài không đo ảnh hưởng lên model và không nêu chi tiết cách gán cụm.
- [Roberts et al. 2017, Ecography](https://pred.uni-regensburg.de/id/eprint/1455/): cơ sở lý thuyết về chia theo khối khi dữ liệu phụ thuộc nhau.
- [Ultralytics Academy](https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/split-the-dataset-correctly): khuyến nghị chia theo cảnh hoặc video.

Mới đọc tóm tắt hoặc trang HTML qua công cụ, cần đọc toàn văn trước khi trích dẫn.

**Cách làm:** Cùng ý tưởng "giữ nguyên cụm" của bài báo, nhưng dùng đặc trưng đơn giản hơn.
1. Tính tương quan giữa ảnh thu nhỏ xám 64×36 của mọi cặp ảnh.
2. Cắt nhóm khi hai ảnh liền số có tương quan dưới `cut`, hoặc khi đổi kích thước ảnh.
3. Gộp hai nhóm nếu có cặp ảnh tương quan trên `merge` (cảnh gần trùng).
4. Gán nguyên nhóm vào một tập bằng simulated annealing, ưu tiên mọi lớp có ở train, lớp có từ 2 nhóm trở lên có mặt ở val và test, rồi đến tỉ lệ ảnh và box gần 70/15/15.

**Lý do không dùng DINO + HDBSCAN:** Dữ liệu có sẵn thứ tự frame, cách đơn giản không cần cài thư viện nặng và đạt mục tiêu đo được bên dưới.

**Bằng chứng so sánh** (`data/_build/split_compare.txt`, phạt thiếu lớp 200/300):

| Cấu hình | Số nhóm | Ảnh test giống train > 0,95 | Lớp thiếu ở val hoặc test |
|---|---:|---:|---:|
| Split gốc | không có | 30,7% | 0 |
| cut 0,3, merge 0,95 | 86 | 0,0% | 22 |
| cut 0,4, merge 0,95 | 129 | 0,0% | 19 |
| cut 0,5, merge 0,95 | 171 | 0,0% | 17 |
| cut 0,5, merge 0,97 | 192 | 0,4% (val 1,6%) | 16 |

**Quyết định:** `cut=0.5, merge=0.95, seed=0`, phạt thiếu lớp 800/1200. Đây là cấu hình giữ 0% ảnh gần trùng mà thiếu ít lớp nhất. Tăng phạt giúp giảm số lớp thiếu ở test từ 12 xuống 10.

**Kết quả cuối** (`data/_build/splits/split_report.txt`):

| | Độ giống lớn nhất với train: median | p90 | > 0,95 |
|---|---:|---:|---:|
| Test split gốc | 0,921 | 0,974 | 30,7% |
| Val mới | 0,677 | 0,860 | 0,0% |
| Test mới | 0,730 | 0,812 | 0,0% |

**Cái giá phải trả:**
- 10 lớp chỉ có ở train, **không đánh giá được**: 15 (W.203c), 21 (W.203b), 24 (P.129), 27 (W.225), 29 (P.104), 42 (R.301e), 44 (W.233), 47 (P.124b1), 48 (W.210), 51 (W.246c). Từ 2026-11-09 thêm lớp 28 (S.505a xe tải và công) sau khi chuyển box 1193#2 (xem C5), thành 11 lớp.
- 7 lớp nữa thiếu ở val nhưng có ở test: 11, 20, 28, 33, 38, 43, 50.
- Một số lớp có rất ít box: test có lớp 28 (1), 14, 25, 34 (2–3); val có lớp 9, 26, 36, 37 (1). AP của các lớp này dao động mạnh.
- `mAP` trên test mới chỉ tính trên 41 lớp (42 trước khi chuyển box 1193#2), nên không so trực tiếp được với split gốc (52 lớp) hay kết quả 0,984 trong README.

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

**4. Đổi seed thì kết quả thay đổi bao nhiêu** (đo 2026-11-09, `data/_build/split_seeds.txt`; chỉ đo, không thay split đang dùng; số box tính trên nhãn `archive` trước khi sửa):

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

**Thay đổi:** Không có trong `data/vn_signs_yolo`, danh sách ở `data/_build/splits/excluded_empty.txt`: 0060, 0061, 0187, 0544, 1514, 1515, 1703–1708, 2015–2017, 2031, 2051, 2723, 2766, 2793–2797, 2999. `archive` vẫn còn đủ.

### 🟡 A4. Cấu trúc chuẩn Ultralytics ✅
**Ghi chú của bạn:** Làm nếu đảm bảo an toàn, đặt trong folder nhưng git ignore trước.

**Thay đổi:**
- `.gitignore` tạo trước, đã kiểm tra bằng `git check-ignore`.
- `data/vn_signs_yolo/{images,labels}/{train,val,test}` và `data.yaml`.
- Ảnh là bản clone APFS (`cp -c`), không dùng chung inode với ảnh gốc, tốn thêm gần như 0 dung lượng. Nhãn là file mới.
- `data.yaml` không ghi `path`: Ultralytics lấy thư mục chứa file yaml làm gốc, nên mang sang Colab/Kaggle vẫn chạy được.

**Kiểm tra** (`data/_build/verify_log.txt`):
- 3191 cặp ảnh/nhãn, không ảnh nào nằm ở hai tập. Bằng đúng `archive` trừ 25 ảnh rỗng.
- 8332 box (8334 trừ 2 dòng B1), 0 box lỗi biên.
- 0 ảnh khác nội dung so với gốc, 0 ảnh dùng chung inode, 0 JPEG lỗi.
- `archive` khớp md5 trước và sau: 6437 file (`archive_md5_truoc.txt`).

**Chưa kiểm tra:** Chưa chạy bằng Ultralytics thật vì máy này chưa cài. Cấu trúc được đối chiếu với mã nguồn `check_det_dataset` và `img2label_paths`. Lệnh thử đầu tiên khi có môi trường:

```bash
yolo detect train data=data/vn_signs_yolo/data.yaml imgsz=960 fliplr=0 epochs=1
```

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

**Số liệu** (`data/_build/measure_log.txt`):
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
| Bộ của repo (README) | 46 lớp, 11.577 ảnh, GoPro, TP.HCM | Gần nhất, nhưng không có link và bạn không có |
| [Zalo Traffic Sign 2020](https://datasetninja.com/zalo-traffic-sign) | 4.500 ảnh, Google Street View | Kiểu ảnh khác hẳn |
| [Vietnamese Traffic Signs](https://datasetninja.com/vietnamese-traffic-signs) | 1.170 ảnh | Không rõ giấy phép |
| [VN_traffic_sign_2](https://universe.roboflow.com/trafficsignvietnamese/vn_traffic_sign_2) | 111 ảnh, CC BY 4.0 | Quá ít |

Chưa tải và chưa kiểm tra chất lượng các bộ trên.

**Đánh giá:** Trong 5 lớp hiếm nhất của `archive`, 4 lớp (W.233, W.246c, P.124b1, W.202b) không có trong bảng 46 biển của repo, lớp còn lại (W.203b) chỉ khớp chung với mã `203`. So bảng đó với 52 lớp: 33 lớp có mã tương ứng, trong đó 7 lớp bị gộp chung (201, 203, 205).

**Quyết định:** Chưa gộp. Bộ gần nhất không có sẵn, và các bộ còn lại phải ánh xạ lớp, kiểm tra trùng ảnh và giấy phép.

**Bản gán nhãn của editor:** Bài trao đổi chỉ cho thấy một người đề nghị gán bổ sung, chưa cho thấy đã có bản hoàn chỉnh. Bạn không có link, nên không dùng bản đó (2026-11-02). Bài trao đổi được giữ làm bằng chứng cho B4.

**Hệ quả:** Lớp hiếm vẫn giữ nguyên số mẫu hiện có. Các lớp chỉ có ở train (xem A1) vẫn chưa đánh giá được.

### 🟠 C2. Tên lớp ✅
**Ghi chú của bạn:** Dùng tên tiếng Việt.

**Thay đổi:** `names` trong `data.yaml` lấy từ `classes_vie.txt`. Sửa tên tiếp trong catalog.

**Lệch giữa file Anh và Việt (nhận định theo hiểu biết, chưa đối chiếu nguồn):** id 1, 22, 37.

### 🟠 C3. W.246a và W.246c ✅ kiểm tra, ⏳ đặt lại tên
**Ghi chú của bạn:** Check 0961–0964 có phải box nhỏ không. Nếu gán sai thì đổi tên hay gán lại. Cho vào catalog để vừa nhìn vừa sửa.

**0961–0964:** Có. Box lớp 26 chỉ 13–15 × 11–12 px trên ảnh 960×540. Lớp 26 có 41/80 box nhỏ, lớp 51 có 5/5. Các box lớn hơn mới thấy rõ biển (1120, 1221, 1006).

**Đối chiếu với ảnh** (`data/_build/bang_chung_w246.jpg`):
- Lớp 26 (W.246a): mũi tên rẽ ra hai bên, khớp "vòng tránh hai bên". Mã đúng, nhưng tên "Chướng ngoại vật phía trước" chung chung và có lỗi chính tả.
- Lớp 51 (W.246c): mũi tên một hướng, ảnh rất mờ. Có vẻ khớp "tránh sang bên phải" và tên trong file.

Nguồn là trang tra cứu [hoatieu.vn](https://hoatieu.vn/bieu-mau/bien-bao-chu-y-chuong-ngai-vat-w246a-168207), chưa đối chiếu văn bản QCVN gốc. Mới xem 8 box lớn, chưa xem hết.

**Trả lời "đổi tên hay gán lại":** Không cần gán lại. Id chỉ là số, tên chỉ là chú thích. Chỉ cần gán lại những box lẫn biển khác.

**Kết quả catalog (2026-11-09):** Lớp 26: 56 Đúng, 24 Không rõ (chủ yếu 0961–0981, box rất nhỏ). Lớp 51: 3 Đúng, 2 Không rõ. Không box nào bị đánh Sai, nên không cần gán lại.

**Thay đổi:** Đổi tên lớp 26 trong `data.yaml` thành "Chú ý chướng ngại vật, vòng tránh hai bên" (bạn đồng ý). `classes_vie.txt` trong `archive` giữ nguyên.

### 🟠 C4. Mã không chuẩn và dấu `*` ⏳
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

### 🟠 C5. Biển phụ S.505a ⏳
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
| 3 | C6 | Dataset chưa lọc và đã lọc | Confusion matrix nhóm dễ nhầm | Tôi hiểu "đã lọc" là sau khi áp dụng kết quả catalog. Nếu có gộp lớp thì so như Thí nghiệm 2 |

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
- **Bằng chứng có sẵn từ model cũ trong repo** (`YOLOV8/*/results.csv`, chỉ còn số liệu epoch 72–100):

| Model | Val cls_loss (epoch 72 → 100) | mAP50 val | mAP50-95 val |
|---|---|---:|---:|
| YOLOv8n | 0,306 → 0,291 | 0,987 | 0,871 |
| YOLOv8s | 0,259 → 0,251 | 0,989 | 0,894 |
| YOLOv8m | 0,244 → 0,241 | 0,990 | 0,901 |

  Val loss không tăng, nên nhìn đường cong thì không thấy overfitting. Nhưng đây là bộ dữ liệu 46 lớp khác và val của nó có thể cũng bị rò rỉ như split gốc của `archive` (30,7% ảnh test gần trùng ảnh train). Nếu val giống train, đường cong không thể lộ ra việc model nhớ cảnh. Các con số trong `results.csv` cũng khác bảng trong README (YOLOv8m: 0,990/0,901 so với 0,984/0,839), chưa rõ vì sao.
- **Cách trả lời bằng số liệu:** Thí nghiệm 5 đo trực tiếp điểm giảm bao nhiêu khi test không giống train. Thí nghiệm 6 đo với ảnh khác hẳn.

## F. Câu hỏi còn mở

1. Dataset dùng cho mục đích gì (đồ án, so sánh model, triển khai thật)?
2. Bạn có video gốc không? Nếu có, nên chia lại theo video thật thay vì suy từ độ giống ảnh.
3. Giữ đủ 52 lớp hay gộp lại? 11 lớp hiện không đánh giá được, xem A1.
