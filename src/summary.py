"""In từng lớp của mạng: kích thước đầu ra và số tham số. Dùng để kiểm tra kết quả tính tay.

Dùng: python -m src.summary --model a0 [--size 64]

Receptive field không in ở đây: tự tính tay bằng r_ra = r_vào + (k − 1) × j_vào, j_ra = j_vào × s
(ke_hoach/nguoi_1...), vì đó là câu hỏi phải tự trả lời được.
"""
import argparse

import torch

from src.models import get_model


def summarize(model, size=64):
    rows = []

    def hook(name):
        def f(module, inp, out):
            n = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append((name, type(module).__name__, tuple(out.shape[1:]), n))
        return f

    handles = [m.register_forward_hook(hook(name)) for name, m in model.named_modules()
               if len(list(m.children())) == 0]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in handles:
        h.remove()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    model = get_model(args.model, size=args.size)
    rows = summarize(model, args.size)
    print(f"{'lớp':<16}{'loại':<14}{'đầu ra':<20}{'tham số':>12}")
    print(f"{'đầu vào':<16}{'':<14}{str((3, args.size, args.size)):<20}{'':>12}")
    for name, kind, shape, n in rows:
        print(f"{name:<16}{kind:<14}{str(shape):<20}{n:>12,}")
    total = sum(r[3] for r in rows)
    print(f"{'tổng':<50}{total:>12,}")
    buffers = sum(b.numel() for b in model.buffers())
    if buffers:
        print(f"{'thống kê không học (buffer, ví dụ running mean/var của BN)':<50}{buffers:>12,}")


if __name__ == "__main__":
    main()
