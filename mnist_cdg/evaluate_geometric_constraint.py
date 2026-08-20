from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from tqdm import tqdm

from .common import ensure_dir, seed_everything
from .constraints import (
    GeometricSevenConstraint,
    SevenGeometryThresholds,
    extract_seven_geometry_features,
)


FEATURE_NAMES = [
    "width_ratio",
    "diagonal_slope",
    "diagonal_r2",
    "lower_multirun_fraction",
    "holes",
    "main_component_ratio",
    "ink_ratio",
    "valid",
]


def extract_dataset(dataset, limit: int | None, description: str):
    total = len(dataset) if limit is None else min(limit, len(dataset))
    images = torch.empty((total, 1, 28, 28), dtype=torch.float32)
    labels = np.empty(total, dtype=np.int64)
    features = np.empty((total, len(FEATURE_NAMES)), dtype=np.float64)
    for index in tqdm(range(total), desc=description):
        image, label = dataset[index]
        images[index] = image * 2.0 - 1.0
        labels[index] = label
        result = extract_seven_geometry_features(images[index])
        values = result.to_dict()
        features[index] = [float(values[name]) for name in FEATURE_NAMES]
    return images, labels, features


def predictions_from_features(features: np.ndarray, thresholds: SevenGeometryThresholds):
    columns = {name: features[:, i] for i, name in enumerate(FEATURE_NAMES)}
    return (
        columns["valid"].astype(bool)
        & (columns["width_ratio"] >= thresholds.width_ratio_min)
        & (columns["diagonal_slope"] <= thresholds.diagonal_slope_max)
        & (columns["lower_multirun_fraction"] <= thresholds.lower_multirun_fraction_max)
        & (columns["holes"] <= thresholds.holes_max)
        & (columns["main_component_ratio"] >= thresholds.main_component_ratio_min)
    )


def metrics(labels: np.ndarray, predicted: np.ndarray):
    truth = labels == 7
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    prevalence = float(predicted.mean())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "specificity": specificity, "false_positive_rate": fpr,
        "f1": f1, "accepted_fraction": prevalence,
    }


def choose_thresholds(features: np.ndarray, labels: np.ndarray, target_precision: float, min_recall: float):
    positive = features[labels == 7]
    width_values = positive[:, FEATURE_NAMES.index("width_ratio")]
    slope_values = positive[:, FEATURE_NAMES.index("diagonal_slope")]
    component_values = positive[:, FEATURE_NAMES.index("main_component_ratio")]
    multirun_values = positive[:, FEATURE_NAMES.index("lower_multirun_fraction")]
    width_grid = np.unique(np.quantile(width_values, np.linspace(0.02, 0.60, 16)))
    slope_grid = np.unique(np.quantile(slope_values, np.linspace(0.45, 0.98, 16)))
    component_grid = np.unique(np.quantile(component_values, np.linspace(0.01, 0.35, 10)))
    multirun_grid = np.unique(np.quantile(multirun_values, np.linspace(0.45, 0.98, 8)))

    # Candidate cut-offs are quantiles of real training-set sevens.  The test
    # labels are not used until the selected rule is evaluated below.
    candidates = []
    for width in width_grid:
        for slope in slope_grid:
            for component in component_grid:
                for multirun in multirun_grid:
                    threshold = SevenGeometryThresholds(
                        width_ratio_min=float(width),
                        diagonal_slope_max=float(slope),
                        lower_multirun_fraction_max=float(multirun),
                        main_component_ratio_min=float(component),
                    )
                    result = metrics(labels, predictions_from_features(features, threshold))
                    candidates.append((threshold, result))

    feasible = [item for item in candidates
                if item[1]["precision"] >= target_precision and item[1]["recall"] >= min_recall]
    if feasible:
        return max(feasible, key=lambda item: (item[1]["recall"], item[1]["precision"], item[1]["f1"]))

    recall_floor = min(0.20, min_recall)
    fallback = [item for item in candidates if item[1]["recall"] >= recall_floor]
    if not fallback:
        fallback = candidates
    return max(
        fallback,
        key=lambda item: (
            (1.25 * item[1]["precision"] * item[1]["recall"])
            / max(0.25 * item[1]["precision"] + item[1]["recall"], 1e-12),
            item[1]["precision"],
        ),
    )


def save_predictions_csv(path: Path, labels: np.ndarray, features: np.ndarray, predicted: np.ndarray):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["index", "true_label", "accepted_as_geometric_7"] + FEATURE_NAMES
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(labels)):
            row = {"index": i, "true_label": int(labels[i]),
                   "accepted_as_geometric_7": int(predicted[i])}
            row.update({name: features[i, j] for j, name in enumerate(FEATURE_NAMES)})
            writer.writerow(row)


def save_case_grid(path: Path, images: torch.Tensor, labels: np.ndarray, features: np.ndarray,
                   indices: np.ndarray, title: str, max_images: int = 48):
    selected = indices[:max_images]
    columns = 8
    cell_w, cell_h = 118, 132
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_w, 30 + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), f"{title} (showing {len(selected)} of {len(indices)})", fill="black", font=font)
    for position, index in enumerate(selected):
        array = (((images[index, 0] + 1.0) / 2.0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        digit = Image.fromarray(array, mode="L").resize((84, 84)).convert("RGB")
        panel = Image.new("RGB", (cell_w, cell_h), "white")
        panel.paste(digit, ((cell_w - 84) // 2, 2))
        panel_draw = ImageDraw.Draw(panel)
        width = features[index, FEATURE_NAMES.index("width_ratio")]
        slope = features[index, FEATURE_NAMES.index("diagonal_slope")]
        multirun = features[index, FEATURE_NAMES.index("lower_multirun_fraction")]
        holes = int(features[index, FEATURE_NAMES.index("holes")])
        component = features[index, FEATURE_NAMES.index("main_component_ratio")]
        panel_draw.text((3, 90), f"#{index} true={labels[index]}", fill="black", font=font)
        panel_draw.text((3, 104), f"W={width:.2f} b={slope:.2f}", fill="black", font=font)
        panel_draw.text((3, 118), f"multi={multirun:.2f} h={holes} m={component:.2f}", fill="black", font=font)
        canvas.paste(panel, ((position % columns) * cell_w, 30 + (position // columns) * cell_h))
    canvas.save(path)


def _histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]):
    finite = values[np.isfinite(values)]
    return np.histogram(finite, bins=bins, range=value_range)[0]


def save_histograms(path: Path, labels: np.ndarray, features: np.ndarray):
    specs = [
        ("width_ratio", (0.0, 4.0)),
        ("diagonal_slope", (-2.0, 1.0)),
        ("lower_multirun_fraction", (0.0, 1.0)),
        ("main_component_ratio", (0.5, 1.0)),
        ("holes", (-0.5, 3.5)),
    ]
    panel_w, panel_h = 520, 260
    rows = (len(specs) + 1) // 2
    canvas = Image.new("RGB", (panel_w * 2, panel_h * rows), "white")
    font = ImageFont.load_default()
    truth = labels == 7
    for panel_index, (name, value_range) in enumerate(specs):
        panel = Image.new("RGB", (panel_w, panel_h), "white")
        draw = ImageDraw.Draw(panel)
        values = features[:, FEATURE_NAMES.index(name)]
        positive = _histogram(values[truth], 40, value_range).astype(float)
        negative = _histogram(values[~truth], 40, value_range).astype(float)
        positive /= max(positive.sum(), 1.0)
        negative /= max(negative.sum(), 1.0)
        maximum = max(float(positive.max()), float(negative.max()), 1e-9)
        left, top, right, bottom = 48, 28, panel_w - 15, panel_h - 35
        draw.line((left, bottom, right, bottom), fill="black")
        draw.line((left, top, left, bottom), fill="black")
        bin_width = (right - left) / len(positive)
        for i, (p, n) in enumerate(zip(positive, negative)):
            x = left + (i + 0.5) * bin_width
            yp = bottom - (p / maximum) * (bottom - top)
            yn = bottom - (n / maximum) * (bottom - top)
            draw.line((x, bottom, x, yp), fill="red", width=2)
            draw.line((x + 2, bottom, x + 2, yn), fill="blue", width=2)
        draw.text((8, 7), f"{name}: red=true 7, blue=non-7", fill="black", font=font)
        draw.text((left, bottom + 7), f"{value_range[0]:.1f}", fill="black", font=font)
        draw.text((right - 25, bottom + 7), f"{value_range[1]:.1f}", fill="black", font=font)
        canvas.paste(panel, ((panel_index % 2) * panel_w, (panel_index // 2) * panel_h))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description="Fit and audit an explicit geometric hard set for MNIST digit 7.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs_formal/mnist/geometric_constraint")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.40)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-test", type=int)
    args = parser.parse_args()
    seed_everything(42)
    output = ensure_dir(args.output)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(args.data_dir, train=True, download=False, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=False, transform=transform)

    train_images, train_labels, train_features = extract_dataset(train_data, args.max_train, "train geometry")
    thresholds, train_metrics = choose_thresholds(
        train_features, train_labels, args.target_precision, args.min_recall
    )
    train_pred = predictions_from_features(train_features, thresholds)
    test_images, test_labels, test_features = extract_dataset(test_data, args.max_test, "test geometry")
    test_pred = predictions_from_features(test_features, thresholds)
    test_metrics = metrics(test_labels, test_pred)

    with (output / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(thresholds.to_dict(), handle, indent=2)
    report = {
        "selection": {
            "target_precision": args.target_precision,
            "minimum_recall": args.min_recall,
            "thresholds_selected_using": "MNIST training split only",
        },
        "thresholds": thresholds.to_dict(),
        "train": train_metrics,
        "test": test_metrics,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savez_compressed(output / "train_features.npz", labels=train_labels, features=train_features,
                        feature_names=np.asarray(FEATURE_NAMES))
    np.savez_compressed(output / "test_features.npz", labels=test_labels, features=test_features,
                        feature_names=np.asarray(FEATURE_NAMES), predictions=test_pred)
    save_predictions_csv(output / "test_predictions.csv", test_labels, test_features, test_pred)
    save_histograms(output / "feature_distributions.png", train_labels, train_features)

    truth = test_labels == 7
    cases = {
        "true_positive": np.flatnonzero(test_pred & truth),
        "false_positive": np.flatnonzero(test_pred & ~truth),
        "false_negative": np.flatnonzero(~test_pred & truth),
        "true_negative": np.flatnonzero(~test_pred & ~truth),
    }
    for name, indices in cases.items():
        rng = np.random.default_rng(42)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        save_case_grid(output / f"{name}.png", test_images, test_labels, test_features, shuffled, name)

    # Instantiate the reusable constraint once as a final consistency check.
    # Check the reusable per-image rule against the vectorised evaluator before
    # saving the selected thresholds for later trajectory labelling.
    constraint = GeometricSevenConstraint(thresholds)
    check_n = min(128, len(test_images))
    direct = constraint.indicator(test_images[:check_n]).numpy()
    if not np.array_equal(direct, test_pred[:check_n]):
        raise RuntimeError("constraint.indicator disagrees with vectorized evaluator")

    print(json.dumps(report, indent=2))
    print(f"saved geometric constraint audit to {output}")


if __name__ == "__main__":
    main()
