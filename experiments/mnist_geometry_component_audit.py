from __future__ import annotations

"""Component-wise audit of the transparent MNIST geometric hard set."""

import argparse
import json
from pathlib import Path

import torch

from mnist_cdg.constraints import SevenGeometryThresholds, extract_seven_geometry_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-run", default="outputs_formal/mnist/formal_eta_geometry_multiseed_v1")
    parser.add_argument("--thresholds", default="configs/geometry_v2_thresholds.json")
    parser.add_argument("--output", default="outputs_formal/mnist/formal_eta_geometry_multiseed_v1/component_audit.json")
    args = parser.parse_args()
    thresholds = SevenGeometryThresholds.from_json(args.thresholds)

    aggregated: dict[str, dict[str, int]] = {}
    # Count each failed geometric condition rather than treating all rejected
    # images as the same type of failure.
    for seed_dir in sorted(Path(args.formal_run).glob("seed_*")):
        for sample_path in seed_dir.glob("samples_eta_*.pt"):
            eta = sample_path.stem.split("_")[-1]
            counts = aggregated.setdefault(eta, {
                "total": 0, "valid": 0, "width_ratio": 0, "diagonal_slope": 0,
                "lower_multirun": 0, "holes": 0, "main_component": 0, "full": 0,
            })
            images = torch.load(sample_path, map_location="cpu", weights_only=False)
            for image in images:
                feature = extract_seven_geometry_features(
                    image, thresholds.pixel_threshold, thresholds.top_fraction,
                    thresholds.bottom_fraction)
                gates = {
                    "valid": feature.valid,
                    "width_ratio": feature.width_ratio >= thresholds.width_ratio_min,
                    "diagonal_slope": feature.diagonal_slope <= thresholds.diagonal_slope_max,
                    "lower_multirun": feature.lower_multirun_fraction <= thresholds.lower_multirun_fraction_max,
                    "holes": feature.holes <= thresholds.holes_max,
                    "main_component": feature.main_component_ratio >= thresholds.main_component_ratio_min,
                }
                counts["total"] += 1
                for key, passed in gates.items():
                    counts[key] += int(passed)
                counts["full"] += int(all(gates.values()))
    report = {"thresholds": thresholds.to_dict(), "etas": {}}
    for eta, counts in sorted(aggregated.items(), key=lambda item: float(item[0])):
        total = counts["total"]
        report["etas"][f"eta_{eta}"] = {
            "counts": counts,
            "pass_rates": {key: value / total for key, value in counts.items() if key != "total"},
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved geometry component audit to {output}")


if __name__ == "__main__":
    main()
