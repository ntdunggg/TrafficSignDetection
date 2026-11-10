# Hướng dẫn cho AI agent

File này dành cho mọi AI agent làm việc trong repo (Claude Code, Codex, Cursor, Copilot...). Đọc hết trước khi làm bất cứ việc gì.

Đây là bài tập lớn môn Học sâu (IT4653, HUST) của nhóm 4 người: nhận diện biển báo giao thông Việt Nam. Cô chấm nặng nhất hai phần:
1. **Nhóm tự xây dựng mạng.**
2. **Trả lời được câu hỏi trong quá trình làm, có số liệu làm bằng chứng.**

Agent giúp người dùng làm và hiểu bài. Không làm thay đến mức người dùng không giải thích lại được.

## 1. Đầu mỗi phiên

1. Hỏi người dùng là ai trong 4 người, nếu họ chưa nói:

   | Vai | Người | Kế hoạch | Sổ ghi |
   |---|---|---|---|
   | Người 1 | Dũng | [`ke_hoach/nguoi_1_dung_du_lieu_va_A0-A2.md`](ke_hoach/nguoi_1_dung_du_lieu_va_A0-A2.md) | [`NHAT_KY.md`](NHAT_KY.md) (sổ chung, Dũng giữ) |
   | Người 2 | Sỹ Nguyên | [`ke_hoach/nguoi_2_sy_nguyen_yolo.md`](ke_hoach/nguoi_2_sy_nguyen_yolo.md) | [`nhat_ky/sy_nguyen.md`](nhat_ky/sy_nguyen.md) |
   | Người 3 | Minh Quân | [`ke_hoach/nguoi_3_minh_quan_A3-A7.md`](ke_hoach/nguoi_3_minh_quan_A3-A7.md) | [`nhat_ky/minh_quan.md`](nhat_ky/minh_quan.md) |
   | Người 4 | Anh Đức | [`ke_hoach/nguoi_4_anh_duc_danh_gia_va_B1.md`](ke_hoach/nguoi_4_anh_duc_danh_gia_va_B1.md) | [`nhat_ky/anh_duc.md`](nhat_ky/anh_duc.md) |

2. Đọc theo thứ tự:
   - [`README.md`](README.md).
   - [`ke_hoach/00_chung.md`](ke_hoach/00_chung.md): quy tắc chung, bàn giao.
   - File kế hoạch của người đó.
   - Các mục của `NHAT_KY.md` mà file kế hoạch chỉ tới.
   - Vài mục mới nhất trong sổ ghi của người đó, để biết lần trước làm đến đâu.
3. Nói lại ngắn cho người dùng: đang ở bước nào, việc tiếp theo là gì, đang chờ ai bàn giao gì.

## 2. Cấm

Không làm những việc sau, kể cả khi người dùng nhờ. Nếu người dùng muốn, nói rõ việc đó vi phạm quy tắc nào và để nhóm quyết.

- **Dùng mạng hoặc trọng số đã học trước.** Không file `.pt` có sẵn (`yolov8s.pt`...), không `pretrained=True`, không `weights="IMAGENET1K_..."`, không tải checkpoint. YOLO tạo bằng `model=yolov8s.yaml`. Đây là yêu cầu của cô.
- **Lật ngang ảnh.** Không `RandomHorizontalFlip`, không lật dọc. YOLO phải đặt `fliplr=0`. Lý do: nhiều lớp chỉ khác nhau ở chiều trái/phải (`NHAT_KY.md` mục D1).
- **Sửa dữ liệu dùng chung.** Không sửa bản gốc `archive` (trên máy Dũng), `data/vn_signs_yolo/`, `du_lieu/splits/`. Cần bản biến thể thì báo Dũng.
- **Chấm test để chọn cấu hình.** Mọi lựa chọn dựa trên val. Test chỉ chấm một lần cho cấu hình đã chốt, đúng thời điểm ghi trong kế hoạch.
- **Đưa vào git:** ảnh, ảnh cắt, trọng số (`.pt`, `.pth`, `.ckpt`), thư mục `data/`, `catalog/`, `runs/`. Ảnh có người và biển số xe, không đăng công khai.
- **Sửa file của người khác:** kế hoạch, sổ ghi, code đang do người khác làm. Cần sửa thì báo người dùng để họ trao đổi với người kia.
- **Sửa `NHAT_KY.md`**, trừ khi người dùng là Dũng.

## 3. Cách làm

- **Có thắc mắc hoặc muốn làm khác kế hoạch thì hỏi trước, chưa làm ngay.** Nêu rõ: muốn đổi gì, vì sao, bằng chứng nào. Chỉ làm sau khi người dùng đồng ý.
- **Mỗi lựa chọn phải có lý do kiểm chứng được:** số liệu đã đo, bài giảng (giáo trình chương mấy, slide bài mấy) hoặc tài liệu có link. Không đoán rồi ghi như sự thật. Phát hiện mình từng nói sai thì ghi đính chính.
- **Thí nghiệm:**
  - Mỗi cấu hình chạy 3 seed (0, 1, 2), báo cáo trung bình ± độ lệch chuẩn.
  - Đổi một yếu tố mỗi lần so với bản trước.
  - Ghi md5 thư mục nhãn (hàm `label_md5` trong `ke_hoach/00_chung.md`).
- **YOLO:** `ultralytics==8.4.191`, `imgsz=960`, `fliplr=0`, cùng `batch` cả nhóm. Với AdamW phải ghi rõ `momentum`.
- **Mạng phân loại:** dùng khung code chung trong `src/` và `manifest.csv`, không dùng `ImageFolder`. Chuẩn hóa bằng mean/std tính trên train.
- **Test chỉ có 41 lớp, val có 35.** Mọi mAP và macro-F1 phải ghi rõ tính trên bao nhiêu lớp.
- **Giải thích cho người dùng hiểu.** Với mỗi mạng: kích thước đầu ra, số tham số từng lớp, vì sao chọn như vậy. Người dùng phải tự trả lời được khi cô hỏi.

## 4. Ghi chép (bắt buộc, làm ngay khi xong mỗi việc)

**Kết quả chạy:** mỗi lần chạy thêm một dòng vào `ket_qua/ket_qua.csv`. Các cột xem `ke_hoach/00_chung.md`, quy tắc 6. Nếu file chưa có thì báo Dũng, không tự tạo cột khác.

**Sổ ghi của người dùng.** Ghi mỗi việc, thay đổi, quyết định hoặc phát hiện một mục, **mục mới ở trên cùng**, theo mẫu:

```markdown
## 2026-11-12: <tên việc ngắn>
- **Làm gì:** ...
- **Vì sao:** ... (bài giảng, số liệu hoặc tài liệu làm cơ sở)
- **Kết quả, bằng chứng:** số liệu, đường dẫn file, tên lần chạy trong ket_qua.csv
- **Quyết định của người dùng:** ... (nếu có)
- **Câu hỏi còn mở:** ... (nếu có)
- **Chuyển vào NHAT_KY.md:** có / không
```

- Ghi đúng sự thật. Chưa chạy thì ghi "chưa chạy", lỗi thì ghi lỗi.
- **"Chuyển vào NHAT_KY.md: có"** dành cho: kết quả chính, thay đổi kế hoạch, phát hiện về dữ liệu, câu trả lời cho câu hỏi báo cáo.
- Câu trả lời cho các câu hỏi trong mục "Phần báo cáo của bạn" của file kế hoạch cũng ghi vào sổ của người dùng, kèm số liệu.

**Nếu người dùng là Dũng:**
- Ghi thẳng vào `NHAT_KY.md`: một dòng ở "Lịch sử thay đổi", chi tiết ở mục liên quan.
- Định kỳ đọc các mục "Chuyển vào NHAT_KY.md: có" trong `nhat_ky/` và chép sang `NHAT_KY.md`, ghi rõ nguồn (ví dụ `nhat_ky/minh_quan.md`, mục 2026-11-12).
- Không sửa file trong `nhat_ky/` của người khác.

## 5. Git

- Không làm trên `main`. Mỗi việc một nhánh `<tên>/<việc>`, ví dụ `quan/a3-dropout`, `nguyen/b0`, `duc/ood`, `dung/crops`.
- Commit nhỏ, message tiếng Việt nói rõ làm gì. Gộp vào `main` qua pull request.
- Chỉ commit khi người dùng đồng ý.
- Trước khi commit, kiểm tra không có ảnh hoặc trọng số trong danh sách file (`git status`).

## 6. Ngôn ngữ

Trả lời và ghi chép bằng tiếng Việt có dấu. Tên biến, tên file và lệnh giữ nguyên.
