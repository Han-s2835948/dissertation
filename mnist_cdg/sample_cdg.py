from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .checkpoint import load_weights
from .canonical_constraint import CanonicalSevenConstraint
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .constraints import GeometricSevenConstraint, SevenGeometryThresholds
from .models import HNet, MNISTClassifier, QNet, ScoreUNet
from .sde import VPSDE, reverse_sde_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp.yaml")
    parser.add_argument("--score", default="outputs/score/latest.pt")
    parser.add_argument("--h-checkpoint")
    parser.add_argument("--q-checkpoint")
    parser.add_argument("--method", choices=["ml", "mcl"], default="ml")
    parser.add_argument("--classifier", default="outputs/classifier/latest.pt")
    parser.add_argument("--constraint", choices=["geometry", "canonical", "classifier"], default="geometry")
    parser.add_argument("--thresholds", default="configs/geometry_v2_thresholds.json")
    parser.add_argument("--canonical-metrics", default="outputs_formal/mnist/geometric_constraint_v7_canonical/metrics.json")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="1.0 is the direct Doob-form baseline; other values are engineering approximations")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    hc = cfg["h_training"]
    h_path = args.h_checkpoint or str(Path(cfg["output_dir"]) / "h" / "geometry_7.pt")
    score = ScoreUNet(**cfg["model"]).to(device).eval()
    hnet = HNet(**cfg["model"]).to(device).eval()
    qnet = QNet(**cfg["model"]).to(device).eval() if args.method == "mcl" else None
    load_weights(args.score, score, device)
    load_weights(h_path, hnet, device)
    if args.method == "mcl":
        if not args.q_checkpoint:
            raise ValueError("--q-checkpoint is required for --method mcl")
        load_weights(args.q_checkpoint, qnet, device)
    classifier = None
    geometry = None
    if args.constraint == "classifier":
        classifier = MNISTClassifier().to(device).eval()
        load_weights(args.classifier, classifier, device)
    elif args.constraint == "geometry":
        geometry = GeometricSevenConstraint(SevenGeometryThresholds.from_json(args.thresholds))
    else:
        geometry = CanonicalSevenConstraint.from_files(args.thresholds, args.canonical_metrics)

    def guidance(tau, x):
        # CDG-MCL uses q_phi/h, whereas CDG-ML differentiates log h directly.
        if args.method == "mcl":
            with torch.no_grad():
                h = hnet(x, tau).clamp_min(hc["h_floor"])
                grad = args.guidance_scale * qnet(x, tau) / h[:, None, None, None]
                flat = grad.flatten(1)
                norm = flat.norm(dim=1).clamp_min(1e-8)
                scale = (hc["guidance_clip"] / norm).clamp(max=1.0)
                return grad * scale[:, None, None, None]
        with torch.enable_grad():
            xin = x.detach().requires_grad_(True)
            h = hnet(xin, tau).clamp_min(hc["h_floor"])
            # Autograd differentiates the learned terminal-success probability
            # with respect to the current noisy image, not the network weights.
            grad = args.guidance_scale * torch.autograd.grad(torch.log(h).sum(), xin)[0]
            flat = grad.flatten(1)
            norm = flat.norm(dim=1).clamp_min(1e-8)
            scale = (hc["guidance_clip"] / norm).clamp(max=1.0)
            return (grad * scale[:, None, None, None]).detach()

    # The sampler adds beta(t) times this field to the frozen reverse drift.
    n = args.samples or cfg["sampling"]["batch_size"]
    samples = reverse_sde_sample(score, VPSDE(**cfg["sde"]), (n, 1, 28, 28),
                                 args.steps or cfg["sampling"]["steps"], device, guidance_fn=guidance)
    if args.constraint == "classifier":
        with torch.no_grad():
            pred = classifier(samples).argmax(1)
        accepted = pred == hc["target_class"]
    else:
        accepted = geometry.indicator(samples.cpu())
    success = accepted.float().mean().item()
    scale_name = str(args.guidance_scale).replace(".", "p")
    out = ensure_dir(Path(cfg["output_dir"]) / "samples") / f"cdg_{args.method}_{args.constraint}_7_scale_{scale_name}.png"
    save_image((samples.cpu() + 1) / 2, out, nrow=max(1, int(n**0.5)))
    print(f"method={args.method} constraint={args.constraint} guidance_scale={args.guidance_scale} "
          f"empirical_success={success:.4%} saved={out}")


if __name__ == "__main__":
    main()
