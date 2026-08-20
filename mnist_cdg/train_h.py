from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .models import HNet


class TrajectoryDataset(Dataset):
    def __init__(self, directory: str | Path, target_class: int):
        files = sorted(Path(directory).glob("shard_*.pt"))
        if not files:
            raise FileNotFoundError(f"no trajectory shards found in {directory}")
        chunks = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
        self.x = torch.cat([c["state"] for c in chunks]).float()
        self.tau = torch.cat([c["tau"] for c in chunks]).float()
        # Newer shards store the set indicator directly.  The class-label
        # fallback keeps earlier classifier trajectories readable.
        if all("terminal_event" in c for c in chunks):
            self.y = torch.cat([c["terminal_event"] for c in chunks]).float()
        else:
            classes = torch.cat([c["terminal_class"] for c in chunks]).long()
            self.y = (classes == target_class).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.tau[index], self.y[index]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = n = 0
    pos, neg = [], []
    for x, tau, y in loader:
        x, tau, y = x.to(device), tau.to(device), y.to(device)
        p = model(x, tau)
        total_loss += (p - y).square().sum().item()
        n += y.numel()
        pos.append(p[y.bool()].cpu())
        neg.append(p[~y.bool()].cpu())
    pos = torch.cat([v for v in pos if v.numel()]) if any(v.numel() for v in pos) else torch.tensor([])
    neg = torch.cat([v for v in neg if v.numel()]) if any(v.numel() for v in neg) else torch.tensor([])
    return total_loss / n, pos.mean().item() if pos.numel() else float("nan"), neg.mean().item() if neg.numel() else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp.yaml")
    parser.add_argument("--trajectories", default="outputs/trajectories")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--name", default="geometry_7")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    hc = cfg["h_training"]
    dataset = TrajectoryDataset(args.trajectories, hc["target_class"])
    valid_n = max(1, int(0.1 * len(dataset)))
    train_set, valid_set = random_split(dataset, [len(dataset) - valid_n, valid_n],
                                        generator=torch.Generator().manual_seed(cfg["seed"]))
    train = DataLoader(train_set, batch_size=hc["batch_size"], shuffle=True)
    valid = DataLoader(valid_set, batch_size=hc["batch_size"])
    model = HNet(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hc["lr"])
    epochs = args.epochs or hc["epochs"]
    # One state-time pair from each unconditional path is used in the
    # Monte Carlo approximation of the martingale-loss time integral.
    for epoch in range(1, epochs + 1):
        model.train()
        for x, tau, y in tqdm(train, desc=f"CDG-ML h epoch {epoch}/{epochs}"):
            x, tau, y = x.to(device), tau.to(device), y.to(device)
            # Paper CDG-ML objective: E integral (h_phi(t,Y_t)-1_S(Y_T))^2 dt.
            loss = (model(x, tau) - y).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        brier, ppos, pneg = evaluate(model, valid, device)
        print(f"epoch={epoch} valid_brier={brier:.6f} mean_h_pos={ppos:.4f} mean_h_neg={pneg:.4f}")
    out = ensure_dir(Path(cfg["output_dir"]) / "h") / f"{args.name}.pt"
    save_checkpoint(out, model, optimizer=opt, epoch=epochs, target_class=hc["target_class"],
                    valid_brier=brier, mean_h_pos=ppos, mean_h_neg=pneg, config=cfg)
    print(f"saved h model to {out}")


if __name__ == "__main__":
    main()
