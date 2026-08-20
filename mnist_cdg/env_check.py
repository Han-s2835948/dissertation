from __future__ import annotations

import json
import platform
import sys

import torch
import torchvision


def main():
    cuda = torch.cuda.is_available()
    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
        raise SystemExit("FAIL: use Python 3.10-3.12 for reproducible PyTorch compatibility")
    if not cuda:
        print("WARN: CUDA unavailable; smoke tests work, full training will be slow")
    print("PASS: core environment is usable")


if __name__ == "__main__":
    main()

