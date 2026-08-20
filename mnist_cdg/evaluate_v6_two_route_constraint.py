from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

from mnist_cdg.common import ensure_dir, seed_everything
from mnist_cdg.evaluate_geometric_constraint import FEATURE_NAMES, metrics


def predict(old: np.ndarray, persistent: np.ndarray, lower_multi: np.ndarray,
            structural: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
    (standard_width_min, standard_displacement_max, standard_multi_max, standard_wide_max,
     alternative_width_min, alternative_displacement_max, alternative_multi_max,
     component_min, hole_max) = params
    col = {name: old[:, i] for i, name in enumerate(FEATURE_NAMES)}
    thickness, wide_fraction, bar_valid = persistent.T
    hole_ratio, displacement = structural.T
    bar = bar_valid.astype(bool) & (thickness >= 0.08) & (thickness <= 0.40)
    core = (
        col["valid"].astype(bool)
        & (col["main_component_ratio"] >= component_min)
        & (hole_ratio <= hole_max)
    )
    standard_route = (
        bar
        & (col["width_ratio"] >= standard_width_min)
        & (displacement <= standard_displacement_max)
        & (lower_multi <= standard_multi_max)
        & (wide_fraction <= standard_wide_max)
    )
    # This second route is intentionally restricted to cases without a detected
    # canonical bar and therefore demands a much stronger diagonal displacement.
    alternative_route = (
        ~bar
        & (col["width_ratio"] >= alternative_width_min)
        & (displacement <= alternative_displacement_max)
        & (lower_multi <= alternative_multi_max)
    )
    return core & (standard_route | alternative_route)


def save_grid(path: Path, dataset, indices: np.ndarray, old: np.ndarray,
              persistent: np.ndarray, structural: np.ndarray, title: str) -> None:
    selected = indices[:80]
    columns, cell_w, cell_h = 8, 132, 124
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_w, 28 + rows * cell_h), "white")
    font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((6, 7), f"{title}: showing {len(selected)} of {len(indices)}", fill="black", font=font)
    for position, index in enumerate(selected):
        image, label = dataset[int(index)]
        digit = Image.fromarray((image.squeeze().numpy() * 255).astype(np.uint8), mode="L")
        digit = digit.resize((84, 84)).convert("RGB")
        panel = Image.new("RGB", (cell_w, cell_h), "white")
        panel.paste(digit, ((cell_w - 84) // 2, 1))
        draw = ImageDraw.Draw(panel)
        draw.text((2, 89), f"#{index} true={label}", fill="black", font=font)
        draw.text((2, 103), f"W={old[index,0]:.2f} bar={persistent[index,0]:.2f}", fill="black", font=font)
        draw.text((2, 115), f"hole={structural[index,0]:.2f} disp={structural[index,1]:.2f}", fill="black", font=font)
        canvas.paste(panel, ((position % columns) * cell_w, 28 + (position // columns) * cell_h))
    canvas.save(path)


def pilot_coverage(prediction: np.ndarray) -> dict:
    path = Path("outputs_formal/mnist/visual_audit_pilot/pilot_annotations_codex.csv")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    report = {}
    for category in ("A", "B", "C"):
        indices = [int(row["test_index"]) for row in rows if row["visual_category_A_B_C"] == category]
        accepted = int(prediction[indices].sum())
        report[category] = {"accepted": accepted, "total": len(indices), "rate": accepted / len(indices)}
    return report


def main() -> None:
    seed_everything(42)
    output = ensure_dir("outputs_formal/mnist/geometric_constraint_v6_two_route")
    transform = transforms.ToTensor()
    test_data = datasets.MNIST("data", train=False, download=False, transform=transform)
    v2 = Path("outputs_formal/mnist/geometric_constraint_v2")
    train_old = np.load(v2 / "train_features.npz")
    test_old = np.load(v2 / "test_features.npz")
    persistent = np.load("outputs_formal/mnist/geometric_constraint_v4_persistent_width/features.npz")
    lower = np.load("outputs_formal/mnist/geometric_constraint_v4_lower_multirun/features.npz")
    structural = np.load("outputs_formal/mnist/geometric_constraint_v5_voting/features.npz")

    candidates = []
    for params in itertools.product(
        [1.50, 1.80, 2.00], [-0.05, 0.00, 0.05], [0.00, 0.10], [0.00, 0.10],
        [0.80, 1.00, 1.20, 1.40], [-0.15, -0.20, -0.25, -0.30], [0.00, 0.10],
        [0.90, 0.93], [0.00, 0.02],
    ):
        train_prediction = predict(
            train_old["features"], persistent["train"], lower["train_lower_multirun"],
            structural["train_structural"], params,
        )
        result = metrics(train_old["labels"], train_prediction)
        candidates.append((params, result))
    feasible = [item for item in candidates if item[1]["precision"] >= 0.93 and item[1]["recall"] >= 0.55]
    target_met = bool(feasible)
    if feasible:
        selected = max(feasible, key=lambda item: (item[1]["recall"], item[1]["precision"], item[1]["f1"]))
    else:
        selected = max(candidates, key=lambda item: (item[1]["f1"], item[1]["precision"]))
    params, train_result = selected
    test_prediction = predict(
        test_old["features"], persistent["test"], lower["test_lower_multirun"],
        structural["test_structural"], params,
    )
    test_result = metrics(test_old["labels"], test_prediction)
    names = ["standard_width_min", "standard_displacement_max", "standard_lower_multirun_max",
             "standard_persistent_wide_max", "alternative_width_min", "alternative_displacement_max",
             "alternative_lower_multirun_max", "component_min", "closed_hole_area_ratio_max"]
    report = {
        "definition": "connected/no-loop core AND (canonical bar route OR strong-diagonal alternative route)",
        "selection": {"training_precision_target": 0.93, "training_recall_target": 0.55,
                      "target_was_met": target_met, "selected_using": "MNIST training split only"},
        "selected_thresholds": dict(zip(names, params)),
        "train": train_result,
        "test": test_result,
        "visual_pilot_coverage_posthoc_only": pilot_coverage(test_prediction),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", test_predictions=test_prediction)
    truth = test_old["labels"] == 7
    save_grid(output / "false_positive_first80.png", test_data, np.flatnonzero(test_prediction & ~truth),
              test_old["features"], persistent["test"], structural["test_structural"], "V6 false positives")
    save_grid(output / "false_negative_first80.png", test_data, np.flatnonzero(~test_prediction & truth),
              test_old["features"], persistent["test"], structural["test_structural"], "V6 false negatives")
    print(json.dumps(report, indent=2))
    print(f"saved V6 two-route experiment to {output}")


if __name__ == "__main__":
    main()
