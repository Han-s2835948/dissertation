from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .checkpoint import load_weights, save_checkpoint
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .models import HNet, QNet
from .sde import VPSDE


class CovariationDataset(Dataset):
    def __init__(self, directory: str | Path):
        files = sorted(Path(directory).glob("shard_*.pt"))
        if not files:
            raise FileNotFoundError(f"no trajectory shards found in {directory}")
        chunks = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
        if not all("state_next" in chunk and "tau_next" in chunk for chunk in chunks):
            raise ValueError("CDG-MCL requires shards generated with --save-pairs")
        self.x = torch.cat([chunk["state"] for chunk in chunks]).float()
        self.x_next = torch.cat([chunk["state_next"] for chunk in chunks]).float()
        self.tau = torch.cat([chunk["tau"] for chunk in chunks]).float()
        self.tau_next = torch.cat([chunk["tau_next"] for chunk in chunks]).float()

    def __len__(self):
        return len(self.tau)

    def __getitem__(self, index):
        return self.x[index], self.x_next[index], self.tau[index], self.tau_next[index]


@torch.no_grad()
def covariation_target(hnet, sde, x, x_next, tau, tau_next):
    h = hnet(x, tau)
    h_next = hnet(x_next, tau_next)
    delta_tau = (tau_next - tau).clamp_min(1e-6)
    forward_t = 1.0 - tau
    denominator = sde.beta(forward_t) * delta_tau
    return ((h_next - h)[:, None, None, None] * (x_next - x)
            / denominator[:, None, None, None].clamp_min(1e-6))


@torch.no_grad()
def evaluate(qnet, hnet, sde, loader, device):
    qnet.eval()
    squared_error = squared_target = dot = squared_prediction = 0.0
    elements = 0
    for x, x_next, tau, tau_next in loader:
        x, x_next = x.to(device), x_next.to(device)
        tau, tau_next = tau.to(device), tau_next.to(device)
        target = covariation_target(hnet, sde, x, x_next, tau, tau_next)
        prediction = qnet(x, tau)
        squared_error += (prediction - target).square().sum().item()
        squared_target += target.square().sum().item()
        squared_prediction += prediction.square().sum().item()
        dot += (prediction * target).sum().item()
        elements += target.numel()
    cosine = dot / max((squared_target * squared_prediction) ** 0.5, 1e-12)
    return squared_error / elements, squared_target / elements, cosine


def main():
    parser = argparse.ArgumentParser(description="Train q_psi approx grad h using CDG-MCL.")
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--h-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--name", default="geometry_7")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    dataset = CovariationDataset(args.trajectories)
    valid_n = max(1, int(0.1 * len(dataset)))
    train_set, valid_set = random_split(
        dataset, [len(dataset) - valid_n, valid_n],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    batch_size = cfg["h_training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    hnet = HNet(**cfg["model"]).to(device).eval()
    load_weights(args.h_checkpoint, hnet, device)
    for parameter in hnet.parameters():
        parameter.requires_grad_(False)
    qnet = QNet(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(qnet.parameters(), lr=cfg["h_training"]["lr"])
    sde = VPSDE(**cfg["sde"])

    for epoch in range(1, args.epochs + 1):
        qnet.train()
        for x, x_next, tau, tau_next in tqdm(train_loader, desc=f"CDG-MCL q epoch {epoch}/{args.epochs}"):
            x, x_next = x.to(device), x_next.to(device)
            tau, tau_next = tau.to(device), tau_next.to(device)
            target = covariation_target(hnet, sde, x, x_next, tau, tau_next)
            loss = (qnet(x, tau) - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(qnet.parameters(), cfg["score_training"]["grad_clip"])
            optimizer.step()
        valid_mse, target_energy, cosine = evaluate(qnet, hnet, sde, valid_loader, device)
        print(f"epoch={epoch} valid_mse={valid_mse:.8f} "
              f"target_energy={target_energy:.8f} cosine={cosine:.4f}")

    output = ensure_dir(Path(cfg["output_dir"]) / "q") / f"{args.name}.pt"
    save_checkpoint(output, qnet, optimizer=optimizer, epoch=args.epochs,
                    valid_mse=valid_mse, target_energy=target_energy,
                    cosine=cosine, h_checkpoint=str(args.h_checkpoint), config=cfg)
    print(f"saved q model to {output}")


if __name__ == "__main__":
    main()
