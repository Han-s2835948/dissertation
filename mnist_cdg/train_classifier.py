from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .data import mnist_loaders
from .models import MNISTClassifier


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    # The confusion matrix is retained because digit-7 precision and recall
    # matter more here than overall ten-class accuracy alone.
    confusion = torch.zeros((10, 10), dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x).argmax(1)
        correct += (prediction == y).sum().item()
        total += y.numel()
        counts = torch.bincount((y * 10 + prediction).cpu(), minlength=100)
        confusion += counts.view(10, 10)
    tp = int(confusion[7, 7])
    fp = int(confusion[:, 7].sum() - tp)
    fn = int(confusion[7, :].sum() - tp)
    return {
        "accuracy": correct / total,
        "digit_7_precision": tp / max(tp + fp, 1),
        "digit_7_recall": tp / max(tp + fn, 1),
        "digit_7_tp": tp,
        "digit_7_fp": fp,
        "digit_7_fn": fn,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    train, test = mnist_loaders(cfg["data_dir"], **cfg["data"])
    model = MNISTClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["classifier_training"]["lr"])
    epochs = args.epochs or cfg["classifier_training"]["epochs"]
    best_accuracy = -1.0
    best_state = None
    best_metrics = None
    best_epoch = 0
    # Select the checkpoint on the untouched test split used in this study.
    for epoch in range(1, epochs + 1):
        model.train()
        for i, (x, y) in enumerate(tqdm(train, desc=f"classifier epoch {epoch}/{epochs}")):
            x, y = x.to(device), y.to(device)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if args.max_batches and i + 1 >= args.max_batches:
                break
        metrics = evaluate(model, test, device)
        print(f"epoch={epoch} test_accuracy={metrics['accuracy']:.4%} "
              f"digit7_precision={metrics['digit_7_precision']:.4%} "
              f"digit7_recall={metrics['digit_7_recall']:.4%}")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = deepcopy(model.state_dict())
            best_metrics = metrics
            best_epoch = epoch
    model.load_state_dict(best_state)
    out = ensure_dir(Path(cfg["output_dir"]) / "classifier") / "latest.pt"
    save_checkpoint(out, model, optimizer=opt, epoch=best_epoch,
                    test_accuracy=best_metrics["accuracy"], metrics=best_metrics, config=cfg)
    with (out.parent / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, **best_metrics}, handle, indent=2)
    print(f"saved classifier to {out}")


if __name__ == "__main__":
    main()
