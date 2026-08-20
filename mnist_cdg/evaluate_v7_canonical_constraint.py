from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torchvision import datasets, transforms
from tqdm import tqdm

from mnist_cdg.common import ensure_dir, seed_everything
from mnist_cdg.constraints import SevenGeometryThresholds
from mnist_cdg.evaluate_bar_constraint import BarThresholds, apply_bar
from mnist_cdg.evaluate_geometric_constraint import FEATURE_NAMES, metrics, predictions_from_features


def canonical_features(image) -> tuple[float, float, float]:
    """Top-edge RMSE/H, absolute top-edge slope, middle/body width ratio."""
    array = image.squeeze().numpy().astype(np.float32)
    binary = array >= 0.35
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 1.0, 10.0, 10.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == int(sizes.argmax())
    rows, cols = np.where(main)
    if rows.size == 0:
        return 1.0, 10.0, 10.0
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    height = rmax - rmin + 1
    if height < 5:
        return 1.0, 10.0, 10.0

    top_limit = rmin + max(1, int(np.ceil(0.40 * height)))
    x_values, top_edges = [], []
    for col in range(cmin, cmax + 1):
        occupied = np.flatnonzero(main[rmin:top_limit, col])
        if occupied.size:
            x_values.append((col - cmin) / max(cmax - cmin, 1))
            top_edges.append(float(occupied.min()) / height)
    if len(x_values) >= 4 and np.var(x_values) > 1e-8:
        x = np.asarray(x_values)
        y = np.asarray(top_edges)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        top_rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
        top_slope = float(abs(slope))
    else:
        top_rmse, top_slope = 1.0, 10.0

    widths = np.zeros(height, dtype=np.float64)
    for local_row, row in enumerate(range(rmin, rmax + 1)):
        row_cols = np.flatnonzero(main[row])
        if row_cols.size:
            widths[local_row] = row_cols.max() - row_cols.min() + 1
    middle = widths[int(np.floor(0.35 * height)):int(np.ceil(0.70 * height))]
    bottom = widths[int(np.floor(0.70 * height)):]
    middle = middle[middle > 0]
    bottom = bottom[bottom > 0]
    if middle.size and bottom.size:
        middle_ratio = float(np.quantile(middle, 0.90) / max(np.median(bottom), 1.0))
    else:
        middle_ratio = 10.0
    return top_rmse, top_slope, middle_ratio


def extract(dataset) -> np.ndarray:
    values = np.empty((len(dataset), 3), dtype=np.float64)
    for index in tqdm(range(len(dataset)), desc="V7 canonical features"):
        image, _ = dataset[index]
        values[index] = canonical_features(image)
    return values


def load_visual_labels() -> tuple[np.ndarray, np.ndarray]:
    root = Path("outputs_formal/mnist/standard7_visual_audit")
    by_index: dict[int, bool] = {}
    for name in ("random_official_7", "random_v3_accepted", "all_official_false_positive"):
        with (root / name / "annotations_codex.csv").open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                index = int(row["test_index"])
                value = row["visual_category_A_B_C"] == "A"
                if index in by_index and by_index[index] != value:
                    raise RuntimeError(f"conflicting visual labels for test index {index}")
                by_index[index] = value
    indices = np.asarray(sorted(by_index), dtype=np.int64)
    labels = np.asarray([by_index[int(index)] for index in indices], dtype=bool)
    return indices, labels


def rates(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(truth & prediction)); fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction)); fp = int(np.sum(~truth & prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "canonical_recall": tp / max(tp + fn, 1),
            "noncanonical_rejection": tn / max(tn + fp, 1),
            "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))}


def save_grid(path: Path, dataset, indices: np.ndarray, visual: np.ndarray,
              features: np.ndarray, title: str) -> None:
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
        draw.text((2, 89), f"#{index} label={label}", fill="black", font=font)
        draw.text((2, 103), f"rmse={features[index,0]:.2f} ts={features[index,1]:.2f}", fill="black", font=font)
        draw.text((2, 115), f"mid/body={features[index,2]:.2f}", fill="black", font=font)
        canvas.paste(panel, ((position % columns) * cell_w, 28 + (position // columns) * cell_h))
    canvas.save(path)


def main() -> None:
    seed_everything(42)
    output = ensure_dir("outputs_formal/mnist/geometric_constraint_v7_canonical")
    dataset = datasets.MNIST("data", train=False, download=False, transform=transforms.ToTensor())
    v2 = Path("outputs_formal/mnist/geometric_constraint_v2")
    old = np.load(v2 / "test_features.npz")
    bar = np.load("outputs_formal/mnist/geometric_constraint_v3_bar/bar_features.npz")["test"]
    base = predictions_from_features(old["features"], SevenGeometryThresholds.from_json(v2 / "thresholds.json"))
    v3 = apply_bar(base, bar, BarThresholds(0.10, 0.35, 0.10))
    features = extract(dataset)
    visual_indices, visual_truth = load_visual_labels()

    rng = np.random.default_rng(20260806)
    calibration_mask = np.zeros(len(visual_indices), dtype=bool)
    for value in (False, True):
        positions = np.flatnonzero(visual_truth == value)
        rng.shuffle(positions)
        calibration_mask[positions[:int(round(0.70 * len(positions)))]] = True
    validation_mask = ~calibration_mask
    candidates = []
    r2 = old["features"][:, FEATURE_NAMES.index("diagonal_r2")]
    for r2_min, rmse_max, slope_max, middle_max in itertools.product(
        [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        [0.10, 0.20, 0.30, 0.40, 0.60, 1.00],
        [1.25, 1.50, 1.75, 2.00, 2.50, 3.00],
    ):
        prediction = v3 & (r2 >= r2_min) & (features[:, 0] <= rmse_max) & (features[:, 1] <= slope_max) & (features[:, 2] <= middle_max)
        result = rates(visual_truth[calibration_mask], prediction[visual_indices][calibration_mask])
        candidates.append(((r2_min, rmse_max, slope_max, middle_max), result, prediction))
    feasible = [item for item in candidates if item[1]["canonical_recall"] >= 0.75]
    target_met = bool(feasible)
    pool = feasible if feasible else candidates
    selected = max(pool, key=lambda item: (item[1]["balanced_accuracy"], item[1]["canonical_recall"]))
    thresholds, calibration_result, prediction = selected
    validation_result = rates(visual_truth[validation_mask], prediction[visual_indices][validation_mask])
    report = {
        "definition": "V3 strict AND straight descending body AND straight top edge AND no middle width spike",
        "human_labels": "Codex pre-audit A positive, B/C negative; human verification required",
        "selection": "70% stratified calibration / 30% validation on 137 unique audited images",
        "calibration_recall_target_0p75_met": target_met,
        "thresholds": {"diagonal_r2_min": thresholds[0], "top_edge_rmse_max": thresholds[1],
                       "top_edge_abs_slope_max": thresholds[2], "middle_to_bottom_width_max": thresholds[3]},
        "calibration": calibration_result,
        "validation": validation_result,
        "official_mnist_digit7_metrics_for_context_only": metrics(old["labels"], prediction),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "features_predictions.npz", features=features, predictions=prediction,
                        visual_indices=visual_indices, visual_truth=visual_truth,
                        calibration_mask=calibration_mask)
    truth_digit7 = old["labels"] == 7
    save_grid(output / "accepted_first80.png", dataset, np.flatnonzero(prediction), visual_truth, features, "V7 accepted")
    save_grid(output / "official_false_positive_first80.png", dataset, np.flatnonzero(prediction & ~truth_digit7), visual_truth, features, "V7 official false positives")
    print(json.dumps(report, indent=2))
    print(f"saved V7 canonical experiment to {output}")


if __name__ == "__main__":
    main()
