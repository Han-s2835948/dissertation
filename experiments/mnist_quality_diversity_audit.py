from __future__ import annotations

"""Measure feature-space quality and within-set diversity of saved samples."""

"""Independent feature-space quality and within-class diversity audit."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_cdg.common import load_config, resolve_device, seed_everything
from experiments.mnist_independent_classifier_audit import IndependentLeNet


@torch.no_grad()
def embed(model: IndependentLeNet, images: torch.Tensor, device: torch.device,
          batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    feature_parts, prediction_parts = [], []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        features = model.features(batch).flatten(1)
        logits = model.classifier(features)
        feature_parts.append(F.normalize(features, dim=1).cpu())
        prediction_parts.append(logits.argmax(1).cpu())
    return torch.cat(feature_parts), torch.cat(prediction_parts)


def pairwise_cosine_distance(features: torch.Tensor, maximum: int,
                             generator: torch.Generator) -> float | None:
    if len(features) < 2:
        return None
    if len(features) > maximum:
        features = features[torch.randperm(len(features), generator=generator)[:maximum]]
    similarities = features @ features.T
    mask = ~torch.eye(len(features), dtype=torch.bool)
    return (1 - similarities[mask]).mean().item()


def nearest_bank_distance(features: torch.Tensor, bank: torch.Tensor,
                          batch_size: int = 512) -> tuple[float, float]:
    values = []
    for start in range(0, len(features), batch_size):
        distance = 1 - features[start:start + batch_size] @ bank.T
        values.append(distance.min(1).values)
    result = torch.cat(values)
    return result.mean().item(), torch.quantile(result, 0.9).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--formal-run", default="outputs_formal/mnist/formal_eta_multiseed_v1")
    parser.add_argument("--independent-checkpoint",
                        default="outputs_formal/mnist/independent_classifier_audit_v1/independent_lenet.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--diversity-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=27182)
    parser.add_argument("--output", default="outputs_formal/mnist/quality_diversity_audit_v1/metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    target = cfg["h_training"]["target_class"]
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = IndependentLeNet().to(device).eval()
    checkpoint = torch.load(args.independent_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(lambda value: value * 2 - 1)])
    test = datasets.MNIST(cfg["data_dir"], train=False, download=False, transform=transform)
    real_images = torch.stack([image for image, label in test if label == target])
    real_features, real_predictions = embed(model, real_images, device, args.batch_size)
    real_target_features = real_features[real_predictions.eq(target)]
    report = {
        "design": {
            "feature_model": "independent_lenet_not_used_for_S_or_h",
            "target_class": target,
            "real_test_target_images": len(real_images),
            "real_test_target_recognized_by_independent": int(real_predictions.eq(target).sum().item()),
            "diversity_subsample_maximum": args.diversity_samples,
        },
        "real_target": {
            "feature_pairwise_cosine_distance": pairwise_cosine_distance(
                real_target_features, args.diversity_samples, generator),
        },
        "generated": {},
    }
    eta_paths: dict[str, list[Path]] = {}
    # Pool the three formal seeds before computing each eta-level summary.
    for seed_dir in Path(args.formal_run).glob("seed_*"):
        for path in seed_dir.glob("samples_eta_*.pt"):
            eta_paths.setdefault(path.stem.split("_")[-1], []).append(path)
    for eta, paths in sorted(eta_paths.items(), key=lambda item: float(item[0])):
        images = torch.cat([torch.load(path, map_location="cpu", weights_only=False) for path in paths])
        features, predictions = embed(model, images, device, args.batch_size)
        target_mask = predictions.eq(target)
        target_features = features[target_mask]
        nearest_mean, nearest_p90 = nearest_bank_distance(target_features, real_target_features)
        vertical_tv = (images[:, :, 1:] - images[:, :, :-1]).abs().mean().item()
        horizontal_tv = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean().item()
        report["generated"][f"eta_{eta}"] = {
            "samples": len(images),
            "independent_target_count": int(target_mask.sum().item()),
            "independent_target_rate": target_mask.float().mean().item(),
            "all_feature_pairwise_cosine_distance": pairwise_cosine_distance(
                features, args.diversity_samples, generator),
            "target_feature_pairwise_cosine_distance": pairwise_cosine_distance(
                target_features, args.diversity_samples, generator),
            "target_mean_nearest_real_feature_distance": nearest_mean,
            "target_p90_nearest_real_feature_distance": nearest_p90,
            "mean_total_variation": vertical_tv + horizontal_tv,
            "mean_ink_fraction": ((images + 1) / 2).mean().item(),
        }
        print(f"eta={eta} target_rate={target_mask.float().mean().item():.2%} "
              f"target_diversity={report['generated'][f'eta_{eta}']['target_feature_pairwise_cosine_distance']:.4f}",
              flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved quality/diversity audit to {output}")


if __name__ == "__main__":
    main()
