from __future__ import annotations

"""Train a separate LeNet evaluator and apply it to the formal samples."""

"""Train an independent MNIST classifier and audit generated eta samples.

The architecture and training loop are original project code. This classifier
is not used to define S or train h; it is only a held-out evaluator.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from mnist_cdg.checkpoint import load_weights
from mnist_cdg.common import load_config, resolve_device, seed_everything
from mnist_cdg.models import MNISTClassifier


class IndependentLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 5), nn.GELU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(48 * 4 * 4, 120), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(120, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = total = 0
    model.eval()
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        correct += predictions.eq(labels).sum().item()
        total += len(labels)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--formal-run", default="outputs_formal/mnist/formal_eta_multiseed_v1")
    parser.add_argument("--primary-classifier", default="outputs_formal/mnist/classifier/latest.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--output-dir", default="outputs_formal/mnist/independent_classifier_audit_v1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    target = cfg["h_training"]["target_class"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda value: value * 2 - 1),
    ])
    train_data = datasets.MNIST(cfg["data_dir"], train=True, download=False, transform=transform)
    test_data = datasets.MNIST(cfg["data_dir"], train=False, download=False, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)

    independent = IndependentLeNet().to(device)
    optimizer = torch.optim.AdamW(independent.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    # This model uses a separate seed and is never involved in trajectory
    # labels, h training, or the conditional guidance field.
    for epoch in range(1, args.epochs + 1):
        independent.train()
        loss_sum = count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(independent(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        test_accuracy = accuracy(independent, test_loader, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / count,
                        "test_accuracy": test_accuracy})
        print(f"epoch={epoch} train_loss={loss_sum / count:.5f} "
              f"test_accuracy={test_accuracy:.4%}", flush=True)
    torch.save({"model": independent.state_dict(), "seed": args.seed,
                "history": history}, output / "independent_lenet.pt")

    primary = MNISTClassifier().to(device).eval()
    load_weights(args.primary_classifier, primary, device)
    independent.eval()
    audit = {
        "design": {
            "independent_classifier_not_used_for_S_or_h": True,
            "architecture": "IndependentLeNet",
            "seed": args.seed,
            "epochs": args.epochs,
            "real_test_accuracy": history[-1]["test_accuracy"],
            "target_class": target,
        },
        "training_history": history,
        "generated": {},
    }
    # Evaluate generated samples only after the independent model is fixed.
    for seed_dir in sorted(Path(args.formal_run).glob("seed_*")):
        seed_results = {}
        for sample_path in sorted(seed_dir.glob("samples_eta_*.pt"),
                                  key=lambda path: float(path.stem.split("_")[-1])):
            eta = sample_path.stem.split("_")[-1]
            images = torch.load(sample_path, map_location="cpu", weights_only=False)
            primary_parts, independent_parts = [], []
            with torch.no_grad():
                for start in range(0, len(images), args.batch_size):
                    batch = images[start:start + args.batch_size].to(device)
                    primary_parts.append(primary(batch).softmax(1).cpu())
                    independent_parts.append(independent(batch).softmax(1).cpu())
            primary_prob = torch.cat(primary_parts)
            independent_prob = torch.cat(independent_parts)
            primary_pred = primary_prob.argmax(1)
            independent_pred = independent_prob.argmax(1)
            primary_target = primary_pred.eq(target)
            independent_target = independent_pred.eq(target)
            disagreement = primary_target & (~independent_target)
            seed_results[f"eta_{eta}"] = {
                "samples": len(images),
                "primary_target_rate": primary_target.float().mean().item(),
                "independent_target_rate": independent_target.float().mean().item(),
                "primary_independent_class_agreement": primary_pred.eq(independent_pred).float().mean().item(),
                "primary_target_independent_rejection_count": int(disagreement.sum().item()),
                "mean_independent_target_probability": independent_prob[:, target].mean().item(),
            }
            if disagreement.any():
                save_image((images[disagreement][:64] + 1) / 2,
                           output / f"{seed_dir.name}_eta_{eta}_primary7_independent_not7.png",
                           nrow=8)
        audit["generated"][seed_dir.name] = seed_results
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(f"saved independent classifier audit to {output}")


if __name__ == "__main__":
    main()
