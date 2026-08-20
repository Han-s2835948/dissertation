from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .checkpoint import load_weights
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .models import HNet, ScoreUNet
from .sde import VPSDE


def main():
    parser = argparse.ArgumentParser(description="Build variance-reduced CDG-MCL targets.")
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--h-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-anchors", type=int, default=5000)
    parser.add_argument("--repetitions", type=int, default=64)
    parser.add_argument("--repeat-chunk", type=int, default=8)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    score = ScoreUNet(**cfg["model"]).to(device).eval()
    hnet = HNet(**cfg["model"]).to(device).eval()
    load_weights(args.score, score, device)
    load_weights(args.h_checkpoint, hnet, device)
    for model in (score, hnet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    sde = VPSDE(**cfg["sde"])
    steps = args.steps or cfg["sampling"]["steps"]
    delta = (1.0 - sde.eps) / steps
    output = ensure_dir(args.output)
    files = sorted(Path(args.anchors).glob("shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"no anchor shards found in {args.anchors}")

    total = 0
    dot = target_sq = grad_sq = 0.0
    target_abs_sum = grad_abs_sum = 0.0
    elements = 0
    saved_shards = 0
    for path in tqdm(files, desc="variance-reduced MCL targets"):
        if total >= args.max_anchors:
            break
        payload = torch.load(path, map_location="cpu", weights_only=False)
        x = payload["state"].float()
        tau = payload["tau"].float()
        keep = tau + delta <= 1.0 - sde.eps + 1e-7
        x, tau = x[keep], tau[keep]
        remaining = args.max_anchors - total
        x, tau = x[:remaining].to(device), tau[:remaining].to(device)
        if x.numel() == 0:
            continue

        with torch.no_grad():
            forward_t = 1.0 - tau
            score_value = score(x, forward_t)
            drift = sde.reverse_drift(forward_t, x, score_value)
            diffusion = sde.diffusion(forward_t, x)
            h0 = hnet(x, tau)
            denominator = sde.beta(forward_t) * delta
            target_sum = torch.zeros_like(x)
            completed = 0
            while completed < args.repetitions:
                repeats = min(args.repeat_chunk, args.repetitions - completed)
                noise = torch.randn((repeats,) + x.shape, device=device)
                delta_x = drift.unsqueeze(0) * delta + diffusion.unsqueeze(0) * (delta ** 0.5) * noise
                x_next = x.unsqueeze(0) + delta_x
                tau_next = (tau + delta).repeat(repeats)
                h_next = hnet(x_next.flatten(0, 1), tau_next).view(repeats, x.shape[0])
                increment = ((h_next - h0.unsqueeze(0))[:, :, None, None, None] * delta_x
                             / denominator[None, :, None, None, None].clamp_min(1e-8))
                target_sum += increment.sum(0)
                completed += repeats
            target = target_sum / args.repetitions

        x_for_grad = x.detach().requires_grad_(True)
        with torch.enable_grad():
            h_value = hnet(x_for_grad, tau)
            autograd_value = torch.autograd.grad(h_value.sum(), x_for_grad)[0].detach()
        dot += (target * autograd_value).sum().item()
        target_sq += target.square().sum().item()
        grad_sq += autograd_value.square().sum().item()
        target_abs_sum += target.abs().sum().item()
        grad_abs_sum += autograd_value.abs().sum().item()
        elements += target.numel()

        torch.save(
            {"state": x.detach().cpu().half(), "tau": tau.cpu(),
             "q_target": target.cpu().half(), "repetitions": args.repetitions,
             "delta": delta},
            output / f"target_{saved_shards:05d}.pt",
        )
        saved_shards += 1
        total += x.shape[0]

    cosine = dot / max((target_sq * grad_sq) ** 0.5, 1e-12)
    report = {
        "anchors": total,
        "repetitions": args.repetitions,
        "delta": delta,
        "target_vs_autograd_cosine": cosine,
        "target_rms": (target_sq / elements) ** 0.5,
        "autograd_rms": (grad_sq / elements) ** 0.5,
        "target_mean_abs": target_abs_sum / elements,
        "autograd_mean_abs": grad_abs_sum / elements,
        "pilot_pass_cosine_threshold": 0.1,
        "pilot_pass": cosine > 0.1,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
