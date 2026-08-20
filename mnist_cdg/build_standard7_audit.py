from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

from mnist_cdg.common import ensure_dir
from mnist_cdg.constraints import SevenGeometryThresholds
from mnist_cdg.evaluate_bar_constraint import BarThresholds, apply_bar
from mnist_cdg.evaluate_geometric_constraint import predictions_from_features


def save_set(root: Path, name: str, indices: np.ndarray, dataset, prediction: np.ndarray) -> None:
    folder = ensure_dir(root / name / "individual")
    columns, cell_w, cell_h = 6, 164, 148
    rows = max(1, (len(indices) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * cell_w, 32 + rows * cell_h), "white")
    font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((7, 7), f"{name}: {len(indices)} images", fill="black", font=font)
    csv_path = root / name / "annotations.csv"
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["position", "test_index", "mnist_label", "v3_strict_accept",
                  "visual_category_A_B_C", "visual_style", "canonical_standard_7", "reviewer_notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, index in enumerate(indices):
            image, label = dataset[int(index)]
            digit = Image.fromarray((image.squeeze().numpy() * 255).astype(np.uint8), mode="L")
            digit = digit.resize((112, 112)).convert("RGB")
            digit.save(folder / f"pos_{position:02d}_index_{index}_label_{label}.png")
            panel = Image.new("RGB", (cell_w, cell_h), "white")
            panel.paste(digit, ((cell_w - 112) // 2, 1))
            draw = ImageDraw.Draw(panel)
            draw.text((3, 116), f"pos={position:02d} index={index}", fill="black", font=font)
            draw.text((3, 132), f"label={label} V3={int(prediction[index])}", fill="black", font=font)
            canvas.paste(panel, ((position % columns) * cell_w, 32 + (position // columns) * cell_h))
            writer.writerow({"position": position, "test_index": int(index), "mnist_label": int(label),
                             "v3_strict_accept": int(prediction[index]), "visual_category_A_B_C": "",
                             "visual_style": "", "canonical_standard_7": "", "reviewer_notes": ""})
    canvas.save(root / name / "grid.png")


def main() -> None:
    root = ensure_dir("outputs_formal/mnist/standard7_visual_audit")
    dataset = datasets.MNIST("data", train=False, download=False, transform=transforms.ToTensor())
    v2 = Path("outputs_formal/mnist/geometric_constraint_v2")
    old = np.load(v2 / "test_features.npz")
    bar = np.load("outputs_formal/mnist/geometric_constraint_v3_bar/bar_features.npz")["test"]
    base = predictions_from_features(old["features"], SevenGeometryThresholds.from_json(v2 / "thresholds.json"))
    prediction = apply_bar(base, bar, BarThresholds(0.10, 0.35, 0.10))
    labels = old["labels"]
    rng = np.random.default_rng(20260806)
    population_7 = np.sort(rng.choice(np.flatnonzero(labels == 7), size=60, replace=False))
    accepted = np.sort(rng.choice(np.flatnonzero(prediction), size=60, replace=False))
    official_false_positive = np.flatnonzero(prediction & (labels != 7))
    save_set(root, "random_official_7", population_7, dataset, prediction)
    save_set(root, "random_v3_accepted", accepted, dataset, prediction)
    save_set(root, "all_official_false_positive", official_false_positive, dataset, prediction)
    print(f"saved standard-7 visual audit to {root}")


if __name__ == "__main__":
    main()
