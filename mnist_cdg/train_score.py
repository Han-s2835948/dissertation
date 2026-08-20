from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .data import mnist_loaders
from .models import ScoreUNet
from .sde import VPSDE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int, help="Smoke-test limiter; omit for a real run")
    parser.add_argument("--resume", help="Resume model, optimizer, EMA, epoch and step from a checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    out = ensure_dir(Path(cfg["output_dir"]) / "score")
    writer = SummaryWriter(out / "tb")
    train_loader, _ = mnist_loaders(cfg["data_dir"], **cfg["data"])
    model = ScoreUNet(**cfg["model"]).to(device)
    sde = VPSDE(**cfg["sde"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["score_training"]["lr"])
    epochs = args.epochs or cfg["score_training"]["epochs"]
    step = 0
    start_epoch = 1
    ema_decay = float(cfg["score_training"].get("ema_decay", 0.0))
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}

    # Resuming restores the optimiser and EMA as well as the network weights.
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("ema") is not None:
            ema = payload["ema"]
        step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(f"resuming after epoch {start_epoch - 1}, step {step}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"score epoch {epoch}/{epochs}")
        for batch_index, (x0, _) in enumerate(bar):
            x0 = x0.to(device, non_blocking=True)
            # A fresh time and noise draw gives a different perturbation level
            # for every image in the batch.
            t = torch.rand(x0.shape[0], device=device) * (1.0 - sde.eps) + sde.eps
            xt, noise, std = sde.marginal(x0, t)
            # Variance-weighted DSM: E ||sigma_t s_theta(t,X_t) + z||^2.
            loss = (std * model(xt, t) + noise).square().flatten(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg["score_training"]["grad_clip"])
            optimizer.step()
            if ema_decay > 0.0:
                with torch.no_grad():
                    for name, value in model.state_dict().items():
                        if value.is_floating_point():
                            ema[name].lerp_(value.detach(), 1.0 - ema_decay)
                        else:
                            ema[name].copy_(value.detach())
            running += loss.item()
            writer.add_scalar("score/loss", loss.item(), step)
            step += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break
        mean_loss = running / (batch_index + 1)
        save_checkpoint(out / "latest.pt", model, optimizer, epoch=epoch, step=step,
                        mean_loss=mean_loss, config=cfg, ema=ema)
        if epoch % cfg["score_training"]["save_every"] == 0:
            save_checkpoint(out / f"epoch_{epoch:03d}.pt", model, epoch=epoch, step=step,
                            mean_loss=mean_loss, config=cfg, ema=ema)
        print(f"epoch={epoch} mean_weighted_dsm={mean_loss:.6f}")
    writer.close()


if __name__ == "__main__":
    main()
