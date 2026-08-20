from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torchvision import datasets, transforms
from tqdm import tqdm

from mnist_cdg.common import ensure_dir, seed_everything
from mnist_cdg.evaluate_geometric_constraint import FEATURE_NAMES, metrics


def structural_features(image) -> tuple[float, float]:
    """Return closed-hole area/ink area and lower centre displacement/height."""
    array = image.squeeze().numpy().astype(np.float32)
    binary = array >= 0.35
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 1.0, 1.0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == int(sizes.argmax())
    rows, cols = np.where(main)
    if rows.size == 0:
        return 1.0, 1.0
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    height = rmax - rmin + 1
    crop = main[rmin:rmax + 1, cmin:cmax + 1]

    # Close one-pixel breaks so an almost-closed 9 still exposes its loop.
    closed = ndimage.binary_closing(crop, structure=np.ones((3, 3), dtype=bool))
    padded = np.pad(closed, 1, constant_values=False)
    holes = ndimage.binary_fill_holes(padded) & ~padded
    hole_area_ratio = float(holes.sum() / max(main.sum(), 1))

    centres = []
    for row in range(rmin + int(np.floor(0.30 * height)), rmax + 1):
        row_cols = np.flatnonzero(main[row])
        if row_cols.size:
            centres.append(float(row_cols.mean()))
    if len(centres) < 4:
        displacement = 1.0
    else:
        group = max(1, min(3, len(centres) // 3))
        upper = float(np.median(centres[:group]))
        lower = float(np.median(centres[-group:]))
        displacement = (lower - upper) / height
    return hole_area_ratio, float(displacement)


def extract(dataset, description: str) -> np.ndarray:
    values = np.empty((len(dataset), 2), dtype=np.float64)
    for index in tqdm(range(len(dataset)), desc=description):
        image, _ = dataset[index]
        values[index] = structural_features(image)
    return values


def predict(old: np.ndarray, persistent: np.ndarray, lower_multi: np.ndarray,
            structural: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
    width_min, wide_max, multi_max, displacement_max, hole_max, component_min, votes_min = params
    col = {name: old[:, i] for i, name in enumerate(FEATURE_NAMES)}
    thickness, wide_fraction, bar_valid = persistent.T
    hole_ratio, displacement = structural.T
    core = (
        col["valid"].astype(bool)
        & (col["main_component_ratio"] >= component_min)
        & (hole_ratio <= hole_max)
    )
    votes = np.stack([
        col["width_ratio"] >= width_min,
        bar_valid.astype(bool) & (thickness >= 0.08) & (thickness <= 0.40),
        wide_fraction <= wide_max,
        lower_multi <= multi_max,
        displacement <= displacement_max,
    ]).sum(axis=0)
    return core & (votes >= int(votes_min))


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


def main() -> None:
    seed_everything(42)
    output = ensure_dir("outputs_formal/mnist/geometric_constraint_v5_voting")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST("data", train=True, download=False, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=False, transform=transform)
    v2 = Path("outputs_formal/mnist/geometric_constraint_v2")
    train_old = np.load(v2 / "train_features.npz")
    test_old = np.load(v2 / "test_features.npz")
    persistent = np.load("outputs_formal/mnist/geometric_constraint_v4_persistent_width/features.npz")
    lower = np.load("outputs_formal/mnist/geometric_constraint_v4_lower_multirun/features.npz")
    train_structural = extract(train_data, "train V5 structure")
    test_structural = extract(test_data, "test V5 structure")

    grids = itertools.product(
        [1.40, 1.60, 1.80, 2.00],
        [0.00, 0.10, 0.20],
        [0.00, 0.10, 0.20],
        [-0.05, 0.00, 0.05, 0.10],
        [0.00, 0.02, 0.05, 0.10],
        [0.90, 0.93, 0.95],
        [3, 4],
    )
    candidates = []
    for params in grids:
        train_prediction = predict(train_old["features"], persistent["train"],
                                   lower["train_lower_multirun"], train_structural, params)
        result = metrics(train_old["labels"], train_prediction)
        candidates.append((params, result))
    feasible = [item for item in candidates if item[1]["precision"] >= 0.93 and item[1]["recall"] >= 0.55]
    target_met = bool(feasible)
    if feasible:
        selected = max(feasible, key=lambda item: (item[1]["recall"], item[1]["precision"], item[1]["f1"]))
    else:
        selected = max(candidates, key=lambda item: (item[1]["f1"], item[1]["precision"]))
    params, train_result = selected
    test_prediction = predict(test_old["features"], persistent["test"],
                              lower["test_lower_multirun"], test_structural, params)
    test_result = metrics(test_old["labels"], test_prediction)
    names = ["width_ratio_min", "persistent_wide_fraction_max", "lower_multirun_fraction_max",
             "lower_displacement_max", "closed_hole_area_ratio_max", "main_component_ratio_min", "votes_required_of_5"]
    report = {
        "set_definition": "core connectedness and hole constraints AND at least k of five interpretable shape votes",
        "selection": {"training_precision_target": 0.93, "training_recall_target": 0.55,
                      "target_was_met": target_met, "selected_using": "MNIST training split only"},
        "selected_thresholds": dict(zip(names, params)),
        "train": train_result,
        "test": test_result,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "features.npz", train_structural=train_structural,
                        test_structural=test_structural, test_predictions=test_prediction)
    truth = test_old["labels"] == 7
    save_grid(output / "false_positive_first80.png", test_data, np.flatnonzero(test_prediction & ~truth),
              test_old["features"], persistent["test"], test_structural, "V5 voting false positives")
    save_grid(output / "false_negative_first80.png", test_data, np.flatnonzero(~test_prediction & truth),
              test_old["features"], persistent["test"], test_structural, "V5 voting false negatives")
    print(json.dumps(report, indent=2))
    print(f"saved V5 voting experiment to {output}")


if __name__ == "__main__":
    main()
