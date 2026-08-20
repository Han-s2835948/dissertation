from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torchvision import datasets, transforms
from tqdm import tqdm

from mnist_cdg.common import ensure_dir, seed_everything
from mnist_cdg.constraints import SevenGeometryThresholds
from mnist_cdg.evaluate_geometric_constraint import FEATURE_NAMES, metrics, predictions_from_features


@dataclass(frozen=True)
class BarThresholds:
    thickness_ratio_min: float
    thickness_ratio_max: float
    outside_width_difference_max: float


def bar_features(image) -> tuple[float, float, float, bool]:
    """Return bar thickness/H, outside-bar width difference/H, bar start/H, valid.

    A bar row is a row in the upper 40% whose ink span is at least 1.5 times
    the median lower-stroke span.  The longest consecutive run is the bar.
    """
    array = image.squeeze().numpy().astype(np.float32)
    binary = array >= 0.35
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0.0, 1.0, 1.0, False
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == int(sizes.argmax())
    rows, _ = np.where(main)
    if rows.size == 0:
        return 0.0, 1.0, 1.0, False
    rmin, rmax = int(rows.min()), int(rows.max())
    height = rmax - rmin + 1
    if height < 5:
        return 0.0, 1.0, 1.0, False

    widths = np.zeros(height, dtype=np.float64)
    for local_row, row in enumerate(range(rmin, rmax + 1)):
        columns = np.flatnonzero(main[row])
        if columns.size:
            widths[local_row] = columns.max() - columns.min() + 1
    bottom_start = int(np.floor(0.70 * height))
    bottom_values = widths[bottom_start:]
    bottom_values = bottom_values[bottom_values > 0]
    if bottom_values.size == 0:
        return 0.0, 1.0, 1.0, False
    bottom_width = float(np.median(bottom_values))

    top_end = max(1, int(np.ceil(0.40 * height)))
    wide = widths[:top_end] >= max(2.0, 1.5 * bottom_width)
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(np.r_[wide, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        return 0.0, 1.0, 1.0, False
    bar_start, bar_end = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
    thickness_ratio = (bar_end - bar_start) / height

    outside_mask = np.ones(height, dtype=bool)
    # Ignore one transition row on either side of the horizontal bar.
    outside_mask[max(0, bar_start - 1):min(height, bar_end + 1)] = False
    outside_widths = widths[outside_mask & (widths > 0)]
    if outside_widths.size:
        difference = np.quantile(np.abs(outside_widths - bottom_width), 0.90) / height
    else:
        difference = 0.0
    return float(thickness_ratio), float(difference), float(bar_start / height), True


def extract_bar_dataset(dataset, description: str):
    result = np.empty((len(dataset), 4), dtype=np.float64)
    for index in tqdm(range(len(dataset)), desc=description):
        image, _ = dataset[index]
        result[index] = bar_features(image)
    return result


def apply_bar(base: np.ndarray, features: np.ndarray, threshold: BarThresholds) -> np.ndarray:
    thickness, difference, _, valid = features.T
    return (
        base & valid.astype(bool)
        & (thickness >= threshold.thickness_ratio_min)
        & (thickness <= threshold.thickness_ratio_max)
        & (difference <= threshold.outside_width_difference_max)
    )


def select_thresholds(base: np.ndarray, features: np.ndarray, labels: np.ndarray,
                      target_precision: float, min_recall: float):
    positive = features[labels == 7]
    thickness = positive[:, 0]
    difference = positive[:, 1]
    valid = positive[:, 3].astype(bool)
    thickness = thickness[valid]
    difference = difference[valid]
    lower_grid = np.unique(np.quantile(thickness, [0.00, 0.01, 0.02, 0.05, 0.10, 0.15]))
    upper_grid = np.unique(np.quantile(thickness, [0.85, 0.90, 0.95, 0.98, 0.99, 1.00]))
    difference_grid = np.unique(np.quantile(difference, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 1.00]))
    candidates = []
    for lower in lower_grid:
        for upper in upper_grid:
            if lower > upper:
                continue
            for difference_max in difference_grid:
                threshold = BarThresholds(float(lower), float(upper), float(difference_max))
                result = metrics(labels, apply_bar(base, features, threshold))
                candidates.append((threshold, result))
    feasible = [item for item in candidates
                if item[1]["precision"] >= target_precision and item[1]["recall"] >= min_recall]
    if feasible:
        return max(feasible, key=lambda item: (item[1]["recall"], item[1]["precision"]))
    return max(candidates, key=lambda item: (item[1]["f1"], item[1]["precision"]))


def save_grid(path: Path, dataset, indices: np.ndarray, bar: np.ndarray, title: str):
    rng = np.random.default_rng(42)
    selected = indices.copy()
    rng.shuffle(selected)
    selected = selected[:48]
    columns, cell_w, cell_h = 8, 116, 124
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_w, 28 + rows * cell_h), "white")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), f"{title}: {len(indices)} total", fill="black", font=font)
    for position, index in enumerate(selected):
        image, label = dataset[int(index)]
        digit = Image.fromarray((image.squeeze().numpy() * 255).astype(np.uint8), mode="L")
        digit = digit.resize((84, 84)).convert("RGB")
        panel = Image.new("RGB", (cell_w, cell_h), "white")
        panel.paste(digit, ((cell_w - 84) // 2, 2))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.text((3, 90), f"#{index} true={label}", fill="black", font=font)
        panel_draw.text((3, 104), f"th={bar[index,0]:.3f} out={bar[index,1]:.3f}", fill="black", font=font)
        canvas.paste(panel, ((position % columns) * cell_w, 28 + (position // columns) * cell_h))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test horizontal-bar geometry proposed for MNIST 7.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--v2-output", default="outputs_formal/mnist/geometric_constraint_v2")
    parser.add_argument("--output", default="outputs_formal/mnist/geometric_constraint_v3_bar")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.40)
    args = parser.parse_args()
    seed_everything(42)
    output = ensure_dir(args.output)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(args.data_dir, train=True, download=False, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=False, transform=transform)

    v2 = Path(args.v2_output)
    v2_thresholds = SevenGeometryThresholds.from_json(v2 / "thresholds.json")
    train_old = np.load(v2 / "train_features.npz")
    test_old = np.load(v2 / "test_features.npz")
    train_labels, train_geometry = train_old["labels"], train_old["features"]
    test_labels, test_geometry = test_old["labels"], test_old["features"]
    train_base = predictions_from_features(train_geometry, v2_thresholds)
    test_base = predictions_from_features(test_geometry, v2_thresholds)

    train_bar = extract_bar_dataset(train_data, "train top-bar")
    test_bar = extract_bar_dataset(test_data, "test top-bar")
    selected, train_result = select_thresholds(
        train_base, train_bar, train_labels, args.target_precision, args.min_recall
    )
    test_prediction = apply_bar(test_base, test_bar, selected)
    test_result = metrics(test_labels, test_prediction)
    report = {
        "definition": {
            "bar_thickness_ratio": "longest upper wide-row run / digit height",
            "outside_width_difference": "90th percentile absolute row-width difference from bottom median / digit height",
            "wide_row": "row width >= 1.5 * median bottom width",
            "selection_split": "MNIST training split only",
        },
        "selection": {"target_precision": args.target_precision, "minimum_recall": args.min_recall},
        "thresholds": asdict(selected),
        "v2_test": metrics(test_labels, test_base),
        "v3_train": train_result,
        "v3_test": test_result,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "bar_features.npz", train=train_bar, test=test_bar,
                        test_labels=test_labels, test_predictions=test_prediction)
    truth = test_labels == 7
    save_grid(output / "false_positive.png", test_data, np.flatnonzero(test_prediction & ~truth), test_bar,
              "v3 false positives")
    save_grid(output / "true_positive.png", test_data, np.flatnonzero(test_prediction & truth), test_bar,
              "v3 true positives")
    save_grid(output / "false_negative.png", test_data, np.flatnonzero(~test_prediction & truth), test_bar,
              "v3 false negatives")
    print(json.dumps(report, indent=2))
    print(f"saved bar-constraint audit to {output}")


if __name__ == "__main__":
    main()
