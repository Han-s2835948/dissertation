from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torchvision import datasets, transforms
from tqdm import tqdm

from mnist_cdg.common import ensure_dir, seed_everything
from mnist_cdg.constraints import SevenGeometryThresholds
from mnist_cdg.evaluate_geometric_constraint import FEATURE_NAMES, metrics


def lower_multirun_fraction(image) -> float:
    array = image.squeeze().numpy().astype(np.float32)
    binary = array >= 0.35
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 1.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == int(sizes.argmax())
    rows, cols = np.where(main)
    if rows.size == 0:
        return 1.0
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    height = rmax - rmin + 1
    start = rmin + int(np.floor(0.50 * height))
    multi, observed = 0, 0
    for row in range(start, rmax + 1):
        values = main[row, cmin:cmax + 1]
        if not values.any():
            continue
        # Fill a one-pixel gap before counting separate stroke runs.
        values = ndimage.binary_closing(values, structure=np.ones(3, dtype=bool))
        padded = np.pad(values.astype(np.int8), (1, 1), constant_values=0)
        runs = int(np.sum(np.diff(padded) == 1))
        observed += 1
        multi += int(runs >= 2)
    return multi / max(observed, 1)


def extract(dataset, description: str) -> np.ndarray:
    result = np.empty(len(dataset), dtype=np.float64)
    for index in tqdm(range(len(dataset)), desc=description):
        image, _ = dataset[index]
        result[index] = lower_multirun_fraction(image)
    return result


def base_without_old_multirun(features: np.ndarray, threshold: SevenGeometryThresholds) -> np.ndarray:
    columns = {name: features[:, i] for i, name in enumerate(FEATURE_NAMES)}
    return (
        columns["valid"].astype(bool)
        & (columns["width_ratio"] >= threshold.width_ratio_min)
        & (columns["diagonal_slope"] <= threshold.diagonal_slope_max)
        & (columns["holes"] <= threshold.holes_max)
        & (columns["main_component_ratio"] >= threshold.main_component_ratio_min)
    )


def predict(base: np.ndarray, persistent: np.ndarray, lower_multi: np.ndarray,
            wide_fraction_max: float, lower_multi_max: float) -> np.ndarray:
    thickness, wide_fraction, valid = persistent.T
    return (
        base & valid.astype(bool)
        & (thickness >= 0.08) & (thickness <= 0.40)
        & (wide_fraction <= wide_fraction_max)
        & (lower_multi <= lower_multi_max)
    )


def save_grid(path: Path, dataset, indices: np.ndarray, persistent: np.ndarray,
              lower_multi: np.ndarray, title: str) -> None:
    selected = indices[:80]
    columns, cell_w, cell_h = 8, 124, 122
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
        draw.text((2, 104), f"wide={persistent[index,1]:.2f} multi={lower_multi[index]:.2f}", fill="black", font=font)
        canvas.paste(panel, ((position % columns) * cell_w, 28 + (position // columns) * cell_h))
    canvas.save(path)


def main() -> None:
    seed_everything(42)
    output = ensure_dir("outputs_formal/mnist/geometric_constraint_v4_lower_multirun")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST("data", train=True, download=False, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=False, transform=transform)
    v2_path = Path("outputs_formal/mnist/geometric_constraint_v2")
    persistent_path = Path("outputs_formal/mnist/geometric_constraint_v4_persistent_width/features.npz")
    threshold = SevenGeometryThresholds.from_json(v2_path / "thresholds.json")
    train_old = np.load(v2_path / "train_features.npz")
    test_old = np.load(v2_path / "test_features.npz")
    persistent = np.load(persistent_path)
    train_base = base_without_old_multirun(train_old["features"], threshold)
    test_base = base_without_old_multirun(test_old["features"], threshold)
    train_multi = extract(train_data, "train lower-multirun")
    test_multi = extract(test_data, "test lower-multirun")

    candidates = []
    for wide_max in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for multi_max in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            prediction = predict(train_base, persistent["train"], train_multi, wide_max, multi_max)
            result = metrics(train_old["labels"], prediction)
            candidates.append((wide_max, multi_max, result))
    feasible = [item for item in candidates if item[2]["precision"] >= 0.93 and item[2]["recall"] >= 0.40]
    met_target = bool(feasible)
    if feasible:
        selected = max(feasible, key=lambda item: (item[2]["recall"], item[2]["precision"]))
    else:
        diagnostic = [item for item in candidates if item[2]["recall"] >= 0.40]
        selected = max(diagnostic, key=lambda item: (item[2]["precision"], item[2]["recall"]))
    wide_max, multi_max, train_result = selected
    test_prediction = predict(test_base, persistent["test"], test_multi, wide_max, multi_max)
    test_result = metrics(test_old["labels"], test_prediction)
    report = {
        "definition": {
            "old_whole-body_multirun_removed": True,
            "new_multirun_region": "lower 50% of digit bounding box",
            "one_pixel_gap_closing": True,
            "bar_thickness_ratio": [0.08, 0.40],
            "selected_on": "MNIST training split only",
            "target_train_precision": 0.93,
            "target_was_met": met_target,
        },
        "selected": {"persistent_wide_fraction_max": wide_max, "lower_multirun_fraction_max": multi_max},
        "train": train_result,
        "test": test_result,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "features.npz", train_lower_multirun=train_multi,
                        test_lower_multirun=test_multi, test_predictions=test_prediction)
    truth = test_old["labels"] == 7
    save_grid(output / "false_positive_first80.png", test_data, np.flatnonzero(test_prediction & ~truth),
              persistent["test"], test_multi, "lower-multirun false positives")
    save_grid(output / "false_negative_first80.png", test_data, np.flatnonzero(~test_prediction & truth),
              persistent["test"], test_multi, "lower-multirun false negatives")
    print(json.dumps(report, indent=2))
    print(f"saved lower-multirun experiment to {output}")


if __name__ == "__main__":
    main()
