from __future__ import annotations

"""Paired, multi-seed MNIST guidance-scale study.

This file is original experiment orchestration for this dissertation project.
It reuses the local trained models and sampler, records paired outcomes, and
audits the implemented guidance field. It does not reproduce prose or results
from any external report.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from scipy.stats import binomtest
from torchvision.utils import save_image

from mnist_cdg.checkpoint import load_weights
from mnist_cdg.common import load_config, resolve_device, seed_everything
from mnist_cdg.constraints import GeometricSevenConstraint, SevenGeometryThresholds
from mnist_cdg.models import HNet, MNISTClassifier, ScoreUNet
from mnist_cdg.sde import VPSDE, reverse_sde_sample


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - half, centre + half]


def effective_class_count(predictions: torch.Tensor) -> float:
    histogram = torch.bincount(predictions, minlength=10).float()
    probabilities = histogram / histogram.sum()
    nonzero = probabilities[probabilities > 0]
    return math.exp(-(nonzero * nonzero.log()).sum().item())


def empty_diagnostics(device: torch.device, bins: int = 10) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(bins, device=device, dtype=torch.float64)
        for name in [
            "count", "h_sum", "base_norm_sum", "scaled_raw_norm_sum",
            "used_norm_sum", "effective_norm_sum", "clip_count", "floor_count"
        ]
    }


def finish_diagnostics(values: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    cpu = {key: value.detach().cpu() for key, value in values.items()}
    rows = []
    for index in range(len(cpu["count"])):
        count = max(cpu["count"][index].item(), 1.0)
        rows.append({
            "tau_start": index / len(cpu["count"]),
            "tau_end": (index + 1) / len(cpu["count"]),
            "observations": int(cpu["count"][index].item()),
            "mean_h": cpu["h_sum"][index].item() / count,
            "mean_base_gradient_norm": cpu["base_norm_sum"][index].item() / count,
            "mean_scaled_raw_gradient_norm": cpu["scaled_raw_norm_sum"][index].item() / count,
            "mean_used_gradient_norm": cpu["used_norm_sum"][index].item() / count,
            "mean_beta_weighted_used_norm": cpu["effective_norm_sum"][index].item() / count,
            "clip_rate": cpu["clip_count"][index].item() / count,
            "h_floor_rate": cpu["floor_count"][index].item() / count,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--score", default="outputs_formal/mnist/score/latest.pt")
    parser.add_argument("--h-checkpoint", default="outputs_formal/mnist/h/classifier_7.pt")
    parser.add_argument("--classifier", default="outputs_formal/mnist/classifier/latest.pt")
    parser.add_argument("--constraint", choices=["classifier", "geometry"], default="classifier")
    parser.add_argument("--thresholds",
                        default="configs/geometry_v2_thresholds.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[8042, 9042, 10042])
    parser.add_argument("--etas", nargs="+", type=float, default=[0, 1, 4, 16, 64, 128, 256, 512])
    parser.add_argument("--samples-per-seed", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs_formal/mnist/formal_eta_multiseed_v1")
    args = parser.parse_args()

    if not args.etas or args.etas[0] != 0:
        raise ValueError("The first eta must be 0 so paired baseline transitions can be computed")
    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    target = cfg["h_training"]["target_class"]
    h_floor = cfg["h_training"]["h_floor"]
    guidance_clip = cfg["h_training"]["guidance_clip"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    score = ScoreUNet(**cfg["model"]).to(device).eval()
    hnet = HNet(**cfg["model"]).to(device).eval()
    classifier = MNISTClassifier().to(device).eval()
    load_weights(args.score, score, device)
    load_weights(args.h_checkpoint, hnet, device)
    load_weights(args.classifier, classifier, device)
    sde = VPSDE(**cfg["sde"])
    geometry = (GeometricSevenConstraint(SevenGeometryThresholds.from_json(args.thresholds))
                if args.constraint == "geometry" else None)

    report: dict[str, object] = {
        "design": {
            "seeds": args.seeds,
            "etas": args.etas,
            "samples_per_seed": args.samples_per_seed,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "target_class": target,
            "constraint": args.constraint,
            "geometry_thresholds": args.thresholds if geometry is not None else None,
            "paired_randomness_within_seed": True,
            "coefficient_convention": "eta is the total multiplier; eta=1 is the paper-aligned coefficient",
            "guidance_clip": guidance_clip,
            "h_floor": h_floor,
        },
        "per_seed": {},
        "pooled": {},
    }
    pooled: dict[float, dict[str, object]] = {
        eta: {"predictions": [], "probabilities": [], "accepted": [], "changes": [], "diagnostics": []}
        for eta in args.etas
    }

    # Resetting the same seed for every eta value gives paired paths: initial
    # noise and Brownian increments match within a seed.
    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_results: dict[str, object] = {}
        baseline_images = baseline_accepted = None
        for eta in args.etas:
            seed_everything(seed)
            diagnostics = empty_diagnostics(device)
            call_index = 0

            def guidance(tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                nonlocal call_index
                step_index = call_index % args.steps
                call_index += 1
                bin_index = min(9, 10 * step_index // args.steps)
                with torch.enable_grad():
                    xin = x.detach().requires_grad_(True)
                    raw_h = hnet(xin, tau)
                    h = raw_h.clamp_min(h_floor)
                    base_grad = torch.autograd.grad(torch.log(h).sum(), xin)[0]
                base_norm = base_grad.flatten(1).norm(dim=1)
                scaled = eta * base_grad
                scaled_norm = scaled.flatten(1).norm(dim=1)
                clip_scale = (guidance_clip / scaled_norm.clamp_min(1e-8)).clamp(max=1.0)
                used = scaled * clip_scale[:, None, None, None]
                used_norm = used.flatten(1).norm(dim=1)
                forward_t = 1.0 - tau
                beta = sde.beta(forward_t)
                # Time-binned values are saved for the guidance diagnostics in
                # the results chapter; they do not alter the sampled path.
                with torch.no_grad():
                    diagnostics["count"][bin_index] += x.shape[0]
                    diagnostics["h_sum"][bin_index] += raw_h.double().sum()
                    diagnostics["base_norm_sum"][bin_index] += base_norm.double().sum()
                    diagnostics["scaled_raw_norm_sum"][bin_index] += scaled_norm.double().sum()
                    diagnostics["used_norm_sum"][bin_index] += used_norm.double().sum()
                    diagnostics["effective_norm_sum"][bin_index] += (beta * used_norm).double().sum()
                    diagnostics["clip_count"][bin_index] += scaled_norm.gt(guidance_clip).double().sum()
                    diagnostics["floor_count"][bin_index] += raw_h.le(h_floor).double().sum()
                return used.detach()

            sample_chunks = []
            for start in range(0, args.samples_per_seed, args.batch_size):
                size = min(args.batch_size, args.samples_per_seed - start)
                sample_chunks.append(reverse_sde_sample(
                    score, sde, (size, 1, 28, 28), args.steps, device,
                    guidance_fn=None if eta == 0 else guidance).cpu())
            samples = torch.cat(sample_chunks)
            with torch.no_grad():
                probabilities = classifier(samples.to(device)).softmax(1).cpu()
            predictions = probabilities.argmax(1)
            classifier_target = predictions.eq(target)
            accepted = (classifier_target if geometry is None
                        else geometry.indicator(samples))
            if eta == 0:
                baseline_images = samples
                baseline_accepted = accepted
            assert baseline_images is not None and baseline_accepted is not None
            changed = samples.sub(baseline_images).abs().flatten(1).mean(1)
            gained = ((~baseline_accepted) & accepted).sum().item()
            lost = (baseline_accepted & (~accepted)).sum().item()
            metrics = {
                "success_count": int(accepted.sum().item()),
                "success_rate": accepted.float().mean().item(),
                "success_wilson_95": wilson_interval(int(accepted.sum().item()), len(accepted)),
                "mean_target_probability": probabilities[:, target].mean().item(),
                "classifier_target_count": int(classifier_target.sum().item()),
                "classifier_target_rate": classifier_target.float().mean().item(),
                "predicted_class_histogram": torch.bincount(predictions, minlength=10).tolist(),
                "predicted_class_effective_count": effective_class_count(predictions),
                "mean_absolute_change_from_baseline": changed.mean().item(),
                "gained_vs_unconditional": int(gained),
                "lost_vs_unconditional": int(lost),
                "time_binned_guidance": [] if eta == 0 else finish_diagnostics(diagnostics),
            }
            seed_results[f"eta_{eta:g}"] = metrics
            torch.save(samples, seed_dir / f"samples_eta_{eta:g}.pt")
            save_image((samples[:64] + 1) / 2, seed_dir / f"grid_eta_{eta:g}.png", nrow=8)
            pooled[eta]["predictions"].append(predictions)
            pooled[eta]["probabilities"].append(probabilities[:, target])
            pooled[eta]["accepted"].append(accepted)
            pooled[eta]["changes"].append(changed)
            if eta != 0:
                pooled[eta]["diagnostics"].append(metrics["time_binned_guidance"])
            print(f"seed={seed} eta={eta:g} success={metrics['success_rate']:.2%} "
                  f"mean_p7={metrics['mean_target_probability']:.4f}", flush=True)
        report["per_seed"][str(seed)] = seed_results

    # Pool the three seeds only after retaining the per-seed results.
    pooled_baseline = torch.cat(pooled[0.0]["accepted"])
    for eta in args.etas:
        predictions = torch.cat(pooled[eta]["predictions"])
        probabilities = torch.cat(pooled[eta]["probabilities"])
        accepted = torch.cat(pooled[eta]["accepted"])
        changes = torch.cat(pooled[eta]["changes"])
        successes = int(accepted.sum().item())
        gained = int(((~pooled_baseline) & accepted).sum().item())
        lost = int((pooled_baseline & (~accepted)).sum().item())
        difference = accepted.float() - pooled_baseline.float()
        difference_mean = difference.mean().item()
        difference_se = difference.std(unbiased=True).item() / math.sqrt(len(difference))
        discordant_p = 1.0 if gained + lost == 0 else binomtest(gained, gained + lost, 0.5).pvalue
        seed_rates = [report["per_seed"][str(seed)][f"eta_{eta:g}"]["success_rate"] for seed in args.seeds]
        result = {
            "success_count": successes,
            "total": len(accepted),
            "success_rate": successes / len(accepted),
            "success_wilson_95": wilson_interval(successes, len(accepted)),
            "mean_target_probability": probabilities.mean().item(),
            "classifier_target_count": int(predictions.eq(target).sum().item()),
            "classifier_target_rate": predictions.eq(target).float().mean().item(),
            "predicted_class_histogram": torch.bincount(predictions, minlength=10).tolist(),
            "predicted_class_effective_count": effective_class_count(predictions),
            "mean_absolute_change_from_baseline": changes.mean().item(),
            "gained_vs_unconditional": gained,
            "lost_vs_unconditional": lost,
            "paired_rate_change": difference_mean,
            "paired_rate_change_approx_95": [difference_mean - 1.96 * difference_se,
                                                difference_mean + 1.96 * difference_se],
            "exact_discordant_pair_p": discordant_p,
            "seed_rates": seed_rates,
            "seed_rate_mean": sum(seed_rates) / len(seed_rates),
            "seed_rate_sample_sd": torch.tensor(seed_rates).std(unbiased=True).item(),
        }
        report["pooled"][f"eta_{eta:g}"] = result

    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved formal eta study to {output}")


if __name__ == "__main__":
    main()
