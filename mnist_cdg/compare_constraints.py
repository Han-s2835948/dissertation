from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from mnist_cdg.checkpoint import load_weights
from mnist_cdg.common import ensure_dir, load_config, resolve_device, seed_everything
from mnist_cdg.constraints import GeometricSevenConstraint, SevenGeometryThresholds
from mnist_cdg.models import HNet, MNISTClassifier, ScoreUNet
from mnist_cdg.sde import VPSDE, reverse_sde_sample


def make_guidance(hnet: HNet, h_floor: float, guidance_clip: float):
    def guidance(tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            xin = x.detach().requires_grad_(True)
            h = hnet(xin, tau).clamp_min(h_floor)
            grad = torch.autograd.grad(torch.log(h).sum(), xin)[0]
            flat = grad.flatten(1)
            norm = flat.norm(dim=1).clamp_min(1e-8)
            clip_scale = (guidance_clip / norm).clamp(max=1.0)
            return (grad * clip_scale[:, None, None, None]).detach()

    return guidance


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of MNIST hard constraints.")
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--score", default="outputs_formal/mnist/score/latest.pt")
    parser.add_argument("--classifier", default="outputs_formal/mnist/classifier/latest.pt")
    parser.add_argument("--classifier-h", default="outputs_formal/mnist/h/classifier_7.pt")
    parser.add_argument("--geometry-h", default="outputs_formal/mnist/h/geometry_7.pt")
    parser.add_argument("--thresholds", default="configs/geometry_v2_thresholds.json")
    parser.add_argument("--output", default="outputs_formal/mnist/constraint_comparison")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    seed = int(cfg["seed"])
    model_cfg = cfg["model"]
    h_cfg = cfg["h_training"]
    output = ensure_dir(args.output)

    score = ScoreUNet(**model_cfg).to(device).eval()
    classifier = MNISTClassifier().to(device).eval()
    classifier_h = HNet(**model_cfg).to(device).eval()
    geometry_h = HNet(**model_cfg).to(device).eval()
    load_weights(args.score, score, device)
    load_weights(args.classifier, classifier, device)
    load_weights(args.classifier_h, classifier_h, device)
    load_weights(args.geometry_h, geometry_h, device)
    geometry = GeometricSevenConstraint(SevenGeometryThresholds.from_json(args.thresholds))
    sde = VPSDE(**cfg["sde"])

    methods = {
        "unconditional": None,
        "classifier_cdg_ml": make_guidance(classifier_h, h_cfg["h_floor"], h_cfg["guidance_clip"]),
        "geometry_cdg_ml": make_guidance(geometry_h, h_cfg["h_floor"], h_cfg["guidance_clip"]),
    }
    summary: dict[str, dict[str, float | int | str]] = {}

    for name, guidance in methods.items():
        # Resetting the seed makes the initial Gaussian sample and all Euler--Maruyama
        # Brownian increments identical across the three branches.
        seed_everything(seed)
        samples = reverse_sde_sample(
            score, sde, (args.samples, 1, 28, 28), args.steps, device,
            guidance_fn=guidance,
        ).cpu()
        with torch.no_grad():
            probabilities = classifier(samples.to(device)).softmax(dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        classifier_accept = predictions.eq(h_cfg["target_class"])
        geometry_accept = geometry.indicator(samples)
        intersection = classifier_accept & geometry_accept

        tensor_path = output / f"{name}.pt"
        image_path = output / f"{name}.png"
        torch.save({
            "samples": samples,
            "classifier_probabilities": probabilities,
            "classifier_predictions": predictions,
            "classifier_accept": classifier_accept,
            "geometry_accept": geometry_accept,
        }, tensor_path)
        save_image((samples + 1.0) / 2.0, image_path, nrow=max(1, int(args.samples ** 0.5)))
        summary[name] = {
            "samples": args.samples,
            "classifier_7_rate": float(classifier_accept.float().mean()),
            "geometry_7_rate": float(geometry_accept.float().mean()),
            "both_rate": float(intersection.float().mean()),
            "mean_classifier_p7": float(probabilities[:, h_cfg["target_class"]].mean()),
            "tensor": str(tensor_path),
            "image": str(image_path),
        }
        print(name, json.dumps(summary[name], ensure_ascii=False))

    base = summary["unconditional"]
    for name in ("classifier_cdg_ml", "geometry_cdg_ml"):
        summary[name]["classifier_7_rate_change_pp"] = 100.0 * (
            float(summary[name]["classifier_7_rate"]) - float(base["classifier_7_rate"])
        )
        summary[name]["geometry_7_rate_change_pp"] = 100.0 * (
            float(summary[name]["geometry_7_rate"]) - float(base["geometry_7_rate"])
        )

    report = output / "metrics.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved comparison metrics: {report}")


if __name__ == "__main__":
    main()
