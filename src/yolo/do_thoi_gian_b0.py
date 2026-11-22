"""Đo thời gian 1 epoch B0 (YOLOv8s train từ đầu) trên GPU Kaggle. Chỉ đo, không dùng để đánh giá.

Kế hoạch: ke_hoach/dung_lam_mot_minh.md, mục 1.5. Quy tắc quyết định (đặt trước khi đo):
  - 100 epoch vừa trong một phiên Kaggle (dưới khoảng 10 giờ, giới hạn 12 giờ) thì làm B0 ở tuần 2.
  - Không vừa thì bỏ B0, ghi số đã đo làm lý do.

Chạy trên Kaggle (notebook có GPU, bật Internet, Add Input dataset dngnguyntrng/vn-sign):
  Ô 1:  !pip install -q ultralytics==8.4.191
  Ô 2:  dán toàn bộ file này vào rồi chạy (hoặc !python do_thoi_gian_b0.py).
Kết quả in ra cuối cùng và ghi vào /kaggle/working/b0_thoi_gian.json. Chép đoạn JSON đó về cho Dũng.

Cấu hình giống lần chạy B0 thật, chỉ khác epochs=1:
  - model=yolov8s.yaml (khởi tạo ngẫu nhiên), pretrained=False. imgsz=960, fliplr=0.
  - optimizer=SGD lr0=0.01 momentum=0.937, ghi cố định thay vì optimizer=auto (ke_hoach, mục 1.5).
  - batch=16 (mặc định của Ultralytics). nbs=64 nên gradient cộng dồn qua 4 batch mới cập nhật một lần.
  - close_mosaic=0: giữ mosaic trong epoch đo. Mặc định close_mosaic=10 sẽ tắt mosaic vì chỉ có 1 epoch,
    làm epoch nhẹ hơn epoch thật (NHAT_KY, "Chạy thử 1 epoch", lưu ý 2). B0 thật giữ mặc định 10.
  - Các tham số khác để mặc định (deterministic=True, workers=8, cache=False, amp=True, plots=True).

Không tải trọng số có sẵn (AGENTS.md mục 2):
  Ultralytics 8.4.191 kiểm tra AMP bằng cách tải yolo26n.pt (utils/checks.py, hàm check_amp), dù không dùng
  để train. Script thay check_amp bằng hàm trả về True: AMP vẫn bật, không tải file .pt nào.
  Đây cũng là điều Ultralytics làm khi không có mạng (bỏ qua kiểm tra, giữ amp=True).
  Nếu loss ra NaN thì chạy lại với AMP = False ở dưới.
"""
import hashlib
import json
import os
import pathlib
import time

import torch
import ultralytics
import ultralytics.engine.trainer as trainer_mod
from ultralytics import YOLO

DATA = "/kaggle/input/datasets/dngnguyntrng/vn-sign/vn_signs_yolo/data.yaml"
OUT = "/kaggle/working/runs_b"
NAME = "b0_do_thoi_gian"
BATCH = 16          # nếu báo hết bộ nhớ GPU (CUDA out of memory) thì đổi thành 8 và ghi lại
AMP = True
MD5_NHAN = "80a393b981121f57e7080674c2750722"  # md5 nhãn của bản dữ liệu đang dùng (du_lieu/crops_info.json)
GIO_TOI_DA = 10     # quy tắc quyết định: 100 epoch phải dưới khoảng 10 giờ


def label_md5(labels_dir):
    """md5 thư mục labels, bỏ qua file .cache (ke_hoach/00_chung.md, quy tắc 5)."""
    h, root = hashlib.md5(), pathlib.Path(labels_dir)
    for p in sorted(root.rglob("*.txt")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


assert ultralytics.__version__ == "8.4.191", f"Cần ultralytics 8.4.191, đang có {ultralytics.__version__}"
assert torch.cuda.is_available(), "Notebook chưa bật GPU (Settings → Accelerator)"
assert os.path.isfile(DATA), f"Không thấy {DATA}. Tìm lại: !find /kaggle/input -name data.yaml"

md5 = label_md5(pathlib.Path(DATA).parent / "labels")
print("md5 nhãn:", md5, "(khớp)" if md5 == MD5_NHAN else f"(KHÔNG khớp {MD5_NHAN}, báo Dũng)")

trainer_mod.check_amp = lambda model: True  # không tải yolo26n.pt, xem docstring

moc = {}


def on_train_epoch_start(tr):
    torch.cuda.synchronize()
    moc["train_bat_dau"] = time.time()


def on_train_epoch_end(tr):
    torch.cuda.synchronize()
    moc["train_xong"] = time.time()


def on_fit_epoch_end(tr):
    # Gọi lần đầu sau khi chấm val, ghi kết quả và lưu last.pt/best.pt của epoch.
    # final_eval() gọi lại callback này sau khi chấm best.pt lần cuối, nên chỉ lấy lần đầu.
    torch.cuda.synchronize()
    moc.setdefault("val_xong", time.time())


model = YOLO("yolov8s.yaml")
model.add_callback("on_train_epoch_start", on_train_epoch_start)
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model.train(
    data=DATA, epochs=1, imgsz=960, batch=BATCH, fliplr=0,
    optimizer="SGD", lr0=0.01, momentum=0.937,
    close_mosaic=0, pretrained=False, amp=AMP, seed=0, device=0,
    project=OUT, name=NAME, exist_ok=False,
)
tong = time.time() - t0

giay_train = moc["train_xong"] - moc["train_bat_dau"]
giay_val = moc["val_xong"] - moc["train_xong"]
giay_epoch = giay_train + giay_val
gio_100 = 100 * giay_epoch / 3600
ket_qua = dict(
    gpu=torch.cuda.get_device_name(0),
    so_gpu=torch.cuda.device_count(),
    bo_nho_gpu_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    so_cpu=os.cpu_count(),
    torch=torch.__version__,
    ultralytics=ultralytics.__version__,
    md5_nhan=md5,
    batch=BATCH,
    amp=AMP,
    giay_train_1_epoch=round(giay_train, 1),
    giay_val_1_epoch=round(giay_val, 1),
    giay_moi_epoch=round(giay_epoch, 1),
    giay_tong_ca_chuan_bi_va_cham_cuoi=round(tong, 1),
    bo_nho_dinh_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
    uoc_tinh_gio_100_epoch=round(gio_100, 2),
    quyet_dinh="làm B0 ở tuần 2" if gio_100 < GIO_TOI_DA else "bỏ B0 (100 epoch không vừa một phiên)",
)
with open("/kaggle/working/b0_thoi_gian.json", "w", encoding="utf-8") as f:
    json.dump(ket_qua, f, ensure_ascii=False, indent=2)
print(json.dumps(ket_qua, ensure_ascii=False, indent=2))
