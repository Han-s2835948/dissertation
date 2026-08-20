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
from mnist_cdg.evaluate_geometric_constraint import metrics, predictions_from_features


def persistent_width_features(image, transition_rows: int = 2) -> tuple[float, float, bool]:
    """Return top-bar thickness/H and persistent-wide-row fraction.

    A row is unusually wide only when its width exceeds the median lower-stroke
    width by more than 10% of digit height.  Rows in and immediately below the
    horizontal bar are excluded from this check.
    """
    array = image.squeeze().numpy().astype(np.float32)
    binary = array >= 0.35
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0.0, 1.0, False
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == int(sizes.argmax())
    rows, _ = np.where(main)
    if rows.size == 0:
        return 0.0, 1.0, False
    rmin, rmax = int(rows.min()), int(rows.max())
    height = rmax - rmin + 1
    if height < 5:
        return 0.0, 1.0, False

    widths = np.zeros(height, dtype=np.float64)
    for local_row, row in enumerate(range(rmin, rmax + 1)):
        columns = np.flatnonzero(main[row])
        if columns.size:
            widths[local_row] = columns.max() - columns.min() + 1
    bottom_values = widths[int(np.floor(0.70 * height)):]
    bottom_values = bottom_values[bottom_values > 0]
    if bottom_values.size == 0:
        return 0.0, 1.0, False
    bottom_width = float(np.median(bottom_values))

    top_end = max(1, int(np.ceil(0.40 * height)))
    wide_bar_rows = widths[:top_end] >= max(2.0, 1.5 * bottom_width)
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.r_[wide_bar_rows, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if not runs:
        return 0.0, 1.0, False
    bar_start, bar_end = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
    thickness_ratio = (bar_end - bar_start) / height

    check_start = min(height, bar_end + transition_rows)
    checked_widths = widths[check_start:]
    checked_widths = checked_widths[checked_widths > 0]
    if checked_widths.size == 0:
        return float(thickness_ratio), 0.0, True
    excess = np.maximum(0.0, checked_widths - bottom_width) / height
    persistent_wide_fraction = float(np.mean(excess > 0.10))
    return float(thickness_ratio), persistent_wide_fraction, True


def extract(dataset, description: str) -> np.ndarray:
    result = np.empty((len(dataset), 3), dtype=np.float64)
    for index in tqdm(range(len(dataset)), desc=description):
        image, _ = dataset[index]
        result[index] = persistent_width_features(image)
    return result


def predict(base: np.ndarray, features: np.ndarray, max_wide_fraction: float) -> np.ndarray:
    thickness, wide_fraction, valid = features.T
    return (
        base & valid.astype(bool)
        & (thickness >= 0.08)
        & (thickness <= 0.40)
        & (wide_fraction <= max_wide_fraction)
    )


def save_grid(path: Path, dataset, indices: np.ndarray, features: np.ndarray, title: str) -> None:
    selected = indices[:80]
    columns, cell_w, cell_h = 8, 120, 122
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_w, 28 + rows * cell_h), "white")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), f"{title}: showing {len(selected)} of {len(indices)}", fill="black", font=font)
    for position, index in enumerate(selected):
        image, label = dataset[int(index)]
        digit = Image.fromarray((image.squeeze().numpy() * 255).astype(np.uint8), mode="L")
        digit = digit.resize((84, 84)).convert("RGB")
        panel = Image.new("RGB", (cell_w, cell_h), "white")
        panel.paste(digit, ((cell_w - 84) // 2, 1))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.text((2, 89), f"#{index} true={label}", fill="black", font=font)
        panel_draw.text((2, 104), f"th={features[index,0]:.3f} wide={features[index,1]:.3f}", fill="black", font=font)
        canvas.paste(panel, ((position % columns) * cell_w, 28 + (position // columns) * cell_h))
    canvas.save(path)


def main() -> None:
    seed_everything(42)
    output = ensure_dir("outputs_formal/mnist/geometric_constraint_v4_persistent_width")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST("data", train=True, download=False, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=False, transform=transform)
    old_output = Path("outputs_formal/mnist/geometric_constraint_v2")
    old_thresholds = SevenGeometryThresholds.from_json(old_output / "thresholds.json")
    train_old = np.load(old_output / "train_features.npz")
    test_old = np.load(old_output / "test_features.npz")
    train_base = predictions_from_features(train_old["features"], old_thresholds)
    test_base = predictions_from_features(test_old["features"], old_thresholds)
    train_features = extract(train_data, "train persistent-width")
    test_features = extract(test_data, "test persistent-width")

    # The 0.20 threshold was fixed before inspecting these results.
    train_prediction = predict(train_base, train_features, 0.20)
    test_prediction = predict(test_base, test_features, 0.20)
    report = {
        "definition": {
            "inherits": "all V2 constraints",
            "bar_thickness_ratio": [0.08, 0.40],
            "transition_rows_after_bar": 2,
            "wide_row": "max(0, row_width - bottom_median_width) / digit_height > 0.10",
            "persistent_wide_fraction_max": 0.20,
            "important": "narrower-than-bottom rows are not penalised",
        },
        "v2_test": metrics(test_old["labels"], test_base),
        "v4_train": metrics(train_old["labels"], train_prediction),
        "v4_test": metrics(test_old["labels"], test_prediction),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "features.npz", train=train_features, test=test_features,
                        test_predictions=test_prediction, test_labels=test_old["labels"])
    truth = test_old["labels"] == 7
    save_grid(output / "false_positive_first80.png", test_data,
              np.flatnonzero(test_prediction & ~truth), test_features, "V4 false positives")
    save_grid(output / "false_negative_first80.png", test_data,
              np.flatnonzero(~test_prediction & truth), test_features, "V4 false negatives")
    print(json.dumps(report, indent=2))
    print(f"saved persistent-width experiment to {output}")


if __name__ == "__main__":
    main()
