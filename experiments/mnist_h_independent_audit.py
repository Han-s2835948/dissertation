from __future__ import annotations

"""Evaluate h on fresh trajectories at several reverse-diffusion times."""

"""Independent time-resolved audit of the learned MNIST h function.

The audit trajectories are generated from a new seed and are not reused from
the h-training trajectory directory. Metrics are original project analyses of
the paper-defined h(t,x)=P(Y_T in S | Y_t=x).
"""

import argparse
import json
from pathlib import Path

import torch

from mnist_cdg.checkpoint import load_weights
from mnist_cdg.common import load_config, resolve_device, seed_everything
from mnist_cdg.constraints import GeometricSevenConstraint, SevenGeometryThresholds
from mnist_cdg.models import HNet, MNISTClassifier, ScoreUNet
from mnist_cdg.sde import VPSDE, reverse_sde_sample


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor,
                               bins: int = 10) -> float:
    total = len(labels)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = probabilities.ge(lower) & (probabilities.lt(upper) if index < bins - 1
                                           else probabilities.le(upper))
        if mask.any():
            result += mask.float().mean().item() * abs(
                probabilities[mask].mean().item() - labels[mask].mean().item())
    return result


def binary_auc(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUC with half credit for ties."""
    positive = probabilities[labels.bool()]
    negative = probabilities[~labels.bool()]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return (comparisons.gt(0).float().mean() +
            0.5 * comparisons.eq(0).float().mean()).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--score", default="outputs_formal/mnist/score/latest.pt")
    parser.add_argument("--h-checkpoint", default="outputs_formal/mnist/h/classifier_7.pt")
    parser.add_argument("--classifier", default="outputs_formal/mnist/classifier/latest.pt")
    parser.add_argument("--constraint", choices=["classifier", "geometry"], default="classifier")
    parser.add_argument("--thresholds",
                        default="configs/geometry_v2_thresholds.json")
    parser.add_argument("--paths", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--time-points", type=int, default=11)
    parser.add_argument("--seed", type=int, default=11042)
    parser.add_argument("--output-dir", default="outputs_formal/mnist/h_independent_audit_v1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    target = cfg["h_training"]["target_class"]
    h_floor = cfg["h_training"]["h_floor"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    score = ScoreUNet(**cfg["model"]).to(device).eval()
    hnet = HNet(**cfg["model"]).to(device).eval()
    classifier = MNISTClassifier().to(device).eval()
    load_weights(args.score, score, device)
    load_weights(args.h_checkpoint, hnet, device)
    load_weights(args.classifier, classifier, device)
    sde = VPSDE(**cfg["sde"])
    geometry = (GeometricSevenConstraint(SevenGeometryThresholds.from_json(args.thresholds))
                if args.constraint == "geometry" else None)

    trajectory_chunks = []
    terminal_chunks = []
    taus = None
    # These paths are independent of the trajectories used to fit h_phi.
    for start in range(0, args.paths, args.batch_size):
        size = min(args.batch_size, args.paths - start)
        terminal, trajectory, saved_tau = reverse_sde_sample(
            score, sde, (size, 1, 28, 28), args.steps, device,
            return_trajectory=True, save_steps=args.time_points)
        trajectory_chunks.append(trajectory)
        terminal_chunks.append(terminal.cpu())
        taus = saved_tau
    trajectories = torch.cat(trajectory_chunks)
    terminals = torch.cat(terminal_chunks)
    assert taus is not None
    with torch.no_grad():
        terminal_probabilities = classifier(terminals.to(device)).softmax(1).cpu()
    terminal_classes = terminal_probabilities.argmax(1)
    labels = (terminal_classes.eq(target) if geometry is None
              else geometry.indicator(terminals)).float()
    torch.save({
        "trajectory": trajectories,
        "tau": taus,
        "terminal": terminals,
        "terminal_class": terminal_classes,
        "terminal_target_probability": terminal_probabilities[:, target],
        "terminal_event": labels.bool(),
        "seed": args.seed,
    }, output / "audit_paths.pt")

    rows = []
    # Probability quality and gradient size are evaluated at the same saved
    # times so their development can be compared directly.
    for time_index, tau_value in enumerate(taus.tolist()):
        probability_parts = []
        grad_norm_parts = []
        grad_h_norm_parts = []
        for start in range(0, args.paths, args.batch_size):
            x = trajectories[start:start + args.batch_size, time_index].to(device)
            tau = torch.full((len(x),), tau_value, device=device)
            with torch.enable_grad():
                xin = x.detach().requires_grad_(True)
                raw_h = hnet(xin, tau)
                grad_log_h = torch.autograd.grad(
                    torch.log(raw_h.clamp_min(h_floor)).sum(), xin, retain_graph=True)[0]
                grad_h = torch.autograd.grad(raw_h.sum(), xin)[0]
            probability_parts.append(raw_h.detach().cpu())
            grad_norm_parts.append(grad_log_h.detach().flatten(1).norm(dim=1).cpu())
            grad_h_norm_parts.append(grad_h.detach().flatten(1).norm(dim=1).cpu())
        probabilities = torch.cat(probability_parts)
        grad_norms = torch.cat(grad_norm_parts)
        grad_h_norms = torch.cat(grad_h_norm_parts)
        positive = labels.bool()
        brier = (probabilities - labels).square().mean().item()
        auc = binary_auc(probabilities, labels)
        forward_t = torch.tensor(1.0 - tau_value)
        beta = sde.beta(forward_t).item()
        rows.append({
            "tau": tau_value,
            "forward_t": 1.0 - tau_value,
            "observations": len(labels),
            "positive_rate": labels.mean().item(),
            "brier_score": brier,
            "roc_auc": auc,
            "ece_10_bins": expected_calibration_error(probabilities, labels),
            "mean_h": probabilities.mean().item(),
            "mean_h_positive_terminal": probabilities[positive].mean().item(),
            "mean_h_negative_terminal": probabilities[~positive].mean().item(),
            "mean_grad_log_h_norm": grad_norms.mean().item(),
            "median_grad_log_h_norm": grad_norms.median().item(),
            "p90_grad_log_h_norm": torch.quantile(grad_norms, 0.9).item(),
            "mean_grad_h_norm": grad_h_norms.mean().item(),
            "beta": beta,
            "mean_beta_weighted_grad_log_h_norm": beta * grad_norms.mean().item(),
            "h_floor_rate": probabilities.le(h_floor).float().mean().item(),
        })
        print(f"tau={tau_value:.3f} auc={auc:.4f} brier={brier:.4f} "
              f"grad_log_h={grad_norms.mean().item():.4f}", flush=True)

    report = {
        "design": {
            "independent_from_h_training_trajectories": True,
            "seed": args.seed,
            "paths": args.paths,
            "steps": args.steps,
            "time_points": args.time_points,
            "target_class": target,
            "constraint": args.constraint,
            "geometry_thresholds": args.thresholds if geometry is not None else None,
            "terminal_positive_count": int(labels.sum().item()),
            "terminal_positive_rate": labels.mean().item(),
        },
        "time_resolved_metrics": rows,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved independent h audit to {output}")


if __name__ == "__main__":
    main()
