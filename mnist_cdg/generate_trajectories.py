from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from .checkpoint import load_weights
from .canonical_constraint import CanonicalSevenConstraint
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .constraints import GeometricSevenConstraint, SevenGeometryThresholds
from .models import MNISTClassifier, ScoreUNet
from .sde import VPSDE, reverse_sde_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp.yaml")
    parser.add_argument("--score", default="outputs/score/latest.pt")
    parser.add_argument("--classifier", default="outputs/classifier/latest.pt")
    parser.add_argument("--constraint", choices=["geometry", "canonical", "classifier"], default="geometry")
    parser.add_argument("--thresholds", default="configs/geometry_v2_thresholds.json")
    parser.add_argument("--canonical-metrics", default="outputs_formal/mnist/geometric_constraint_v7_canonical/metrics.json")
    parser.add_argument("--target-class", type=int, default=7)
    parser.add_argument("--output")
    parser.add_argument("--save-pairs", action="store_true",
                        help="Save adjacent states required by the CDG-MCL covariation loss")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    score = ScoreUNet(**cfg["model"]).to(device).eval()
    load_weights(args.score, score, device)
    classifier = None
    geometry = None
    if args.constraint == "classifier":
        classifier = MNISTClassifier().to(device).eval()
        load_weights(args.classifier, classifier, device)
    elif args.constraint == "geometry":
        geometry = GeometricSevenConstraint(SevenGeometryThresholds.from_json(args.thresholds))
    else:
        geometry = CanonicalSevenConstraint.from_files(args.thresholds, args.canonical_metrics)
    total = args.samples or cfg["trajectory"]["samples"]
    save_steps = cfg["trajectory"]["save_steps"]
    default_name = ({"geometry": "geometry_7", "canonical": "canonical_7"}.get(
        args.constraint, f"classifier_{args.target_class}"
    ))
    output = ensure_dir(args.output or Path(cfg["output_dir"]) / "trajectories" / default_name)
    generated = 0

    # Sharding keeps the 50,000-path data set small enough to load and inspect
    # in ordinary batches rather than as one large tensor file.
    for shard in tqdm(range(math.ceil(total / args.batch_size)), desc="off-policy trajectories"):
        batch = min(args.batch_size, total - generated)
        if args.save_pairs:
            terminal, states, states_next, pair_taus, pair_taus_next = reverse_sde_sample(
                score, VPSDE(**cfg["sde"]), (batch, 1, 28, 28),
                args.steps or cfg["sampling"]["steps"], device,
                return_random_pair=True,
            )
        else:
            terminal, trajectory, taus = reverse_sde_sample(
                score, VPSDE(**cfg["sde"]), (batch, 1, 28, 28),
                args.steps or cfg["sampling"]["steps"], device,
                return_trajectory=True, save_steps=save_steps,
            )
        if args.constraint == "classifier":
            with torch.no_grad():
                prob = classifier(terminal).softmax(1).cpu()
                terminal_event = prob.argmax(1).eq(args.target_class)
        else:
            terminal_event = geometry.indicator(terminal.cpu())
        # One uniformly selected time per path is an unbiased Monte Carlo estimate
        # of the paper's time-integrated martingale loss.
        if not args.save_pairs:
            indices = torch.randint(0, trajectory.shape[1], (batch,))
            states = trajectory[torch.arange(batch), indices]
            pair_taus = taus[indices].float()
        payload = {
            "state": states.half(),
            "tau": pair_taus,
            "terminal_event": terminal_event.to(torch.uint8),
            "constraint": args.constraint,
        }
        if args.save_pairs:
            payload["state_next"] = states_next.half()
            payload["tau_next"] = pair_taus_next.float()
        if args.constraint == "classifier":
            payload["terminal_class"] = prob.argmax(1).to(torch.uint8)
            payload["terminal_confidence"] = prob.max(1).values.half()
        torch.save(payload, output / f"shard_{shard:05d}.pt")
        generated += batch
    positives = 0
    for path in output.glob("shard_*.pt"):
        positives += int(torch.load(path, map_location="cpu", weights_only=False)["terminal_event"].sum())
    print(f"saved {generated} off-policy path samples to {output}; "
          f"positive_terminal_events={positives} ({positives / generated:.4%})")


if __name__ == "__main__":
    main()
