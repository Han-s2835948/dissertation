from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .checkpoint import load_weights
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .models import ScoreUNet
from .sde import VPSDE, reverse_sde_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp.yaml")
    parser.add_argument("--checkpoint", default="outputs/score/latest.pt")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    model = ScoreUNet(**cfg["model"]).to(device).eval()
    load_weights(args.checkpoint, model, device)
    n = args.samples or cfg["sampling"]["batch_size"]
    samples = reverse_sde_sample(model, VPSDE(**cfg["sde"]), (n, 1, 28, 28),
                                 args.steps or cfg["sampling"]["steps"], device)
    out = ensure_dir(Path(cfg["output_dir"]) / "samples") / "unconditional.png"
    save_image((samples.cpu() + 1) / 2, out, nrow=max(1, int(n**0.5)))
    print(f"saved {n} samples to {out}")


if __name__ == "__main__":
    main()

