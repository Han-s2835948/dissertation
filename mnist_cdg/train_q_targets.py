from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .models import QNet


class QTargetDataset(Dataset):
    def __init__(self, directory: str | Path):
        files = sorted(Path(directory).glob("target_*.pt"))
        if not files:
            raise FileNotFoundError(f"no target shards found in {directory}")
        chunks = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
        self.x = torch.cat([chunk["state"] for chunk in chunks]).float()
        self.tau = torch.cat([chunk["tau"] for chunk in chunks]).float()
        self.target = torch.cat([chunk["q_target"] for chunk in chunks]).float()

    def __len__(self):
        return len(self.tau)

    def __getitem__(self, index):
        return self.x[index], self.tau[index], self.target[index]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    error_sq = target_sq = prediction_sq = dot = 0.0
    elements = 0
    for x, tau, target in loader:
        x, tau, target = x.to(device), tau.to(device), target.to(device)
        prediction = model(x, tau)
        error_sq += (prediction - target).square().sum().item()
        target_sq += target.square().sum().item()
        prediction_sq += prediction.square().sum().item()
        dot += (prediction * target).sum().item()
        elements += target.numel()
    mse = error_sq / elements
    zero_mse = target_sq / elements
    cosine = dot / max((target_sq * prediction_sq) ** 0.5, 1e-12)
    return mse, zero_mse, cosine


def main():
    parser = argparse.ArgumentParser(description="Train q on averaged CDG-MCL targets.")
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--name", default="geometry_7_mcl_r64_pilot")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    dataset = QTargetDataset(args.targets)
    valid_n = max(1, int(0.1 * len(dataset)))
    train_set, valid_set = random_split(
        dataset, [len(dataset) - valid_n, valid_n],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    batch_size = cfg["h_training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    model = QNet(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["h_training"]["lr"])
    best_mse = float("inf")
    best_state = None
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, tau, target in tqdm(train_loader, desc=f"averaged CDG-MCL q epoch {epoch}/{args.epochs}"):
            x, tau, target = x.to(device), tau.to(device), target.to(device)
            loss = (model(x, tau) - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg["score_training"]["grad_clip"])
            optimizer.step()
        mse, zero_mse, cosine = evaluate(model, valid_loader, device)
        print(f"epoch={epoch} valid_mse={mse:.8f} zero_mse={zero_mse:.8f} cosine={cosine:.4f}")
        if mse < best_mse:
            best_mse = mse
            best_state = deepcopy(model.state_dict())
            best_metrics = (epoch, mse, zero_mse, cosine)

    model.load_state_dict(best_state)
    epoch, mse, zero_mse, cosine = best_metrics
    output = ensure_dir(Path(cfg["output_dir"]) / "q") / f"{args.name}.pt"
    save_checkpoint(output, model, optimizer=optimizer, epoch=epoch, valid_mse=mse,
                    zero_mse=zero_mse, cosine=cosine, config=cfg)
    print(f"saved best q model to {output}; epoch={epoch} improvement={1.0 - mse / zero_mse:.4%}")


if __name__ == "__main__":
    main()
