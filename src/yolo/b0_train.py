"""B0: YOLOv8s train từ đầu trên Kaggle, 100 epoch, mỗi lần một seed, rồi chấm test một lần (kế hoạch mục 2.3).

Chạy trên Kaggle bằng Save & Run All (notebook có GPU, bật Internet, Add Input dataset dngnguyntrng/vn-sign):
  Ô 1:  !pip install -q ultralytics==8.4.191
  Ô 2:  dán toàn bộ file này (hoặc !wget -q <đường dẫn raw của file này> && python b0_train.py)
Seed: sửa SEED ở dưới (0, 1 hoặc 2). Mỗi seed một lần chạy, tên kết quả có số seed.
Mất khoảng 1,7 giờ mỗi seed trên T4 (seed 0 chạy 2026-11-23: 60,7 giây mỗi epoch). Kết quả ở /kaggle/working:
  runs_b/b0_s<seed>/        results.csv, args.yaml, weights/best.pt, last.pt, các hình của Ultralytics
  runs_b/b0_s<seed>_test/   hình của lần chấm test
  b0_ket_qua_s<seed>.json   số liệu tóm tắt: mAP val và test, AP từng lớp trên test, thời gian, md5 nhãn
                            (lần chạy seed 0 ngày 2026-11-23 dùng tên cũ b0_ket_qua.json)

Cấu hình (giống lần đo thời gian, ke_hoach/dung_lam_mot_minh.md mục 1.5, chỉ khác epochs và close_mosaic):
  - model=yolov8s.yaml (khởi tạo ngẫu nhiên), pretrained=False. imgsz=960, fliplr=0.
  - optimizer=SGD lr0=0.01 momentum=0.937 (cố định, không để auto). batch=16, nbs=64.
  - epochs=100, close_mosaic=10 (mặc định), seed=SEED. Các tham số khác mặc định của Ultralytics 8.4.191.
  - Thay check_amp để không tải yolo26n.pt (AGENTS.md mục 2; NHAT_KY mục 2026-11-22).
Chép dữ liệu từ /kaggle/input sang /tmp trước khi train: lần đo thời gian, Ultralytics cảnh báo đọc ảnh chậm
từ /kaggle/input. Kiểm md5 nhãn ở cả bản gốc và bản chép.
Chấm test: một lần, bằng best.pt (epoch có mAP50-95 val cao nhất), split=test. mAP tính trên các lớp có ở test (41).
Nếu phiên bị ngắt giữa chừng: đặt RESUME = True (hoặc B0_RESUME=1) để tiếp tục từ last.pt của chính lần chạy này.
Biến môi trường B0_* chỉ để chạy thử trên máy, trên Kaggle để mặc định.
"""
import csv
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import time

import torch
import ultralytics
import ultralytics.engine.trainer as trainer_mod
from ultralytics import YOLO

DATA_IN = os.environ.get("B0_DATA", "/kaggle/input/datasets/dngnguyntrng/vn-sign/vn_signs_yolo")
LOCAL = os.environ.get("B0_LOCAL", "/tmp/vn_signs_yolo")
OUT = os.environ.get("B0_OUT", "/kaggle/working")
EPOCHS = int(os.environ.get("B0_EPOCHS", 100))
IMGSZ = int(os.environ.get("B0_IMGSZ", 960))
FRACTION = float(os.environ.get("B0_FRACTION", 1.0))
DEVICE = os.environ.get("B0_DEVICE", "0")
RESUME = os.environ.get("B0_RESUME", "0") == "1"
SEED = int(os.environ.get("B0_SEED", 0))  # trên Kaggle: sửa số 0 thành 1 hoặc 2 để chạy seed khác
BATCH, NAME = 16, f"b0_s{SEED}"
MD5_NHAN = "80a393b981121f57e7080674c2750722"


def label_md5(labels_dir):
    """md5 thư mục labels, bỏ qua file .cache (ke_hoach/00_chung.md, quy tắc 5)."""
    h, root = hashlib.md5(), pathlib.Path(labels_dir)
    for p in sorted(root.rglob("*.txt")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def per_class(metrics, names):
    b = metrics.box
    return [dict(class_id=int(c), ten=names[int(c)], ap50=round(float(b.ap50[i]), 4), ap50_95=round(float(b.ap[i]), 4))
            for i, c in enumerate(b.ap_class_index)]


assert ultralytics.__version__ == "8.4.191", f"Cần ultralytics 8.4.191, đang có {ultralytics.__version__}"
if DEVICE != "mps":
    assert torch.cuda.is_available(), "Notebook chưa bật GPU (Settings → Accelerator)"

md5_goc = label_md5(os.path.join(DATA_IN, "labels"))
assert md5_goc == MD5_NHAN, f"md5 nhãn {md5_goc} khác {MD5_NHAN}: báo Dũng, không train"
if not os.path.isdir(LOCAL):
    t = time.time()
    shutil.copytree(DATA_IN, LOCAL, ignore=shutil.ignore_patterns("*.cache"))
    print(f"Đã chép dữ liệu sang {LOCAL} trong {time.time() - t:.0f} giây")
assert label_md5(os.path.join(LOCAL, "labels")) == MD5_NHAN, "Bản chép khác bản gốc"
data = os.path.join(LOCAL, "data.yaml")
print("md5 nhãn khớp:", MD5_NHAN)

trainer_mod.check_amp = lambda model: True  # không tải yolo26n.pt; AMP vẫn bật

project = os.path.join(OUT, "runs_b")
run_dir = os.path.join(project, NAME)
device = DEVICE if DEVICE == "mps" else int(DEVICE)
t0 = time.time()
if RESUME:
    YOLO(os.path.join(run_dir, "weights", "last.pt")).train(resume=True)
else:
    YOLO("yolov8s.yaml").train(
        data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, fliplr=0,
        optimizer="SGD", lr0=0.01, momentum=0.937,
        pretrained=False, seed=SEED, device=device, fraction=FRACTION,
        project=project, name=NAME, exist_ok=False,
    )
gio_train = (time.time() - t0) / 3600

best = os.path.join(run_dir, "weights", "best.pt")
rows = list(csv.DictReader(open(os.path.join(run_dir, "results.csv"))))
key = next(k for k in rows[0] if k.strip() == "metrics/mAP50-95(B)")
best_row = max(rows, key=lambda r: float(r[key]))

model = YOLO(best)
names = model.names
mv = model.val(data=data, split="val", imgsz=IMGSZ, batch=BATCH, device=device,
               project=project, name=NAME + "_val", exist_ok=True)
mt = model.val(data=data, split="test", imgsz=IMGSZ, batch=BATCH, device=device,
               project=project, name=NAME + "_test", exist_ok=True)  # chấm test một lần

so_tham_so = sum(p.numel() for p in model.model.parameters())
ket_qua = dict(
    ngay=datetime.date.today().isoformat(),
    gpu=torch.cuda.get_device_name(0) if DEVICE != "mps" else "mps",
    torch=torch.__version__, ultralytics=ultralytics.__version__, md5_nhan=MD5_NHAN,
    cau_hinh=dict(model="yolov8s.yaml", pretrained=False, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, fliplr=0,
                  optimizer="SGD", lr0=0.01, momentum=0.937, close_mosaic=10, seed=SEED, fraction=FRACTION),
    so_tham_so=so_tham_so,
    gio_train=round(gio_train, 2),
    epoch_tot_nhat=int(float(best_row["epoch"])),
    val=dict(so_lop=len(mv.box.ap_class_index), map50=round(float(mv.box.map50), 4), map50_95=round(float(mv.box.map), 4),
             precision=round(float(mv.box.mp), 4), recall=round(float(mv.box.mr), 4)),
    test=dict(so_lop=len(mt.box.ap_class_index), map50=round(float(mt.box.map50), 4), map50_95=round(float(mt.box.map), 4),
              precision=round(float(mt.box.mp), 4), recall=round(float(mt.box.mr), 4)),
    test_tung_lop=per_class(mt, names),
)
with open(os.path.join(OUT, f"b0_ket_qua_s{SEED}.json"), "w", encoding="utf-8") as f:
    json.dump(ket_qua, f, ensure_ascii=False, indent=2)
tom = {k: v for k, v in ket_qua.items() if k != "test_tung_lop"}
print(json.dumps(tom, ensure_ascii=False, indent=2))
print("AP từng lớp trên test ở", os.path.join(OUT, f"b0_ket_qua_s{SEED}.json"))
