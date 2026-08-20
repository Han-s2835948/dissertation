from __future__ import annotations

"""Compare paired unconditional and guided class outcomes path by path."""

"""Paired class-transition and failure audit for formal MNIST eta samples."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from mnist_cdg.checkpoint import load_weights
from mnist_cdg.common import load_config, resolve_device
from mnist_cdg.models import MNISTClassifier


def predict(model: MNISTClassifier, images: torch.Tensor, device: torch.device,
            batch_size: int) -> torch.Tensor:
    parts = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            parts.append(model(images[start:start + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--formal-run", default="outputs_formal/mnist/formal_eta_multiseed_v1")
    parser.add_argument("--classifier", default="outputs_formal/mnist/classifier/latest.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="outputs_formal/mnist/failure_transition_audit_v1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    target = cfg["h_training"]["target_class"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    classifier = MNISTClassifier().to(device).eval()
    load_weights(args.classifier, classifier, device)

    pooled: dict[str, torch.Tensor] = {}
    images_for_failures: dict[str, list[torch.Tensor]] = {}
    # Grid positions are paired across eta values, allowing each guided class
    # to be compared with its unconditional counterpart.
    for seed_dir in sorted(Path(args.formal_run).glob("seed_*")):
        baseline_images = torch.load(seed_dir / "samples_eta_0.pt", map_location="cpu", weights_only=False)
        baseline_predictions = predict(classifier, baseline_images, device, args.batch_size)
        for sample_path in sorted(seed_dir.glob("samples_eta_*.pt"),
                                  key=lambda path: float(path.stem.split("_")[-1])):
            eta = sample_path.stem.split("_")[-1]
            images = torch.load(sample_path, map_location="cpu", weights_only=False)
            final_predictions = predict(classifier, images, device, args.batch_size)
            matrix = torch.zeros(10, 10, dtype=torch.long)
            for start_class in range(10):
                mask = baseline_predictions.eq(start_class)
                matrix[start_class] = torch.bincount(final_predictions[mask], minlength=10)
            pooled[eta] = pooled.get(eta, torch.zeros_like(matrix)) + matrix
            failure_mask = ~final_predictions.eq(target)
            images_for_failures.setdefault(eta, []).append(images[failure_mask])

    report = {"target_class": target, "etas": {}}
    for eta, matrix in sorted(pooled.items(), key=lambda item: float(item[0])):
        starting_counts = matrix.sum(1)
        target_by_start = matrix[:, target]
        success_by_start = [
            target_by_start[index].item() / starting_counts[index].item()
            if starting_counts[index].item() else float("nan")
            for index in range(10)
        ]
        failure_histogram = matrix.sum(0).clone()
        failure_histogram[target] = 0
        report["etas"][f"eta_{eta}"] = {
            "baseline_to_final_class_matrix": matrix.tolist(),
            "baseline_class_counts": starting_counts.tolist(),
            "target_success_counts_by_baseline_class": target_by_start.tolist(),
            "target_success_rates_by_baseline_class": success_by_start,
            "final_failure_class_histogram": failure_histogram.tolist(),
        }
        failures = torch.cat(images_for_failures[eta])
        if len(failures):
            save_image((failures[:100] + 1) / 2, output / f"failures_eta_{eta}.png", nrow=10)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved failure transition audit to {output}")


if __name__ == "__main__":
    main()
