from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

from mnist_cdg.common import ensure_dir
from mnist_cdg.constraints import SevenGeometryThresholds
from mnist_cdg.evaluate_geometric_constraint import FEATURE_NAMES
from mnist_cdg.export_constraint_cases import v2_failure


def main() -> None:
    output = ensure_dir("outputs_formal/mnist/visual_audit_pilot")
    individual = ensure_dir(output / "individual")
    dataset = datasets.MNIST("data", train=False, download=False, transform=transforms.ToTensor())
    v2_path = Path("outputs_formal/mnist/geometric_constraint_v2")
    old = np.load(v2_path / "test_features.npz")
    lower = np.load("outputs_formal/mnist/geometric_constraint_v4_lower_multirun/features.npz")
    persistent = np.load("outputs_formal/mnist/geometric_constraint_v4_persistent_width/features.npz")
    threshold = SevenGeometryThresholds.from_json(v2_path / "thresholds.json")
    labels = old["labels"]
    prediction = lower["test_predictions"]
    false_negative = np.flatnonzero((labels == 7) & ~prediction)
    rng = np.random.default_rng(20260806)
    selected = np.sort(rng.choice(false_negative, size=60, replace=False))

    columns, cell_w, cell_h = 6, 172, 154
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, 34 + rows * cell_h), "white")
    font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((7, 8), "Visual audit pilot: 60 fixed random V4 false negatives", fill="black", font=font)
    csv_path = output / "pilot_annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["position", "test_index", "true_label", "v2_failure_reason",
                  "width_ratio", "slope", "old_multirun", "bar_thickness",
                  "persistent_wide_fraction", "lower_multirun", "visual_category_A_B_C",
                  "visual_style", "should_set_accept", "reviewer_notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, index in enumerate(selected):
            image, label = dataset[int(index)]
            array = (image.squeeze().numpy() * 255).astype(np.uint8)
            digit = Image.fromarray(array, mode="L").resize((112, 112)).convert("RGB")
            digit.save(individual / f"pos_{position:02d}_index_{index}.png")
            old_feature = old["features"][index]
            reason = v2_failure(old_feature, threshold)
            panel = Image.new("RGB", (cell_w, cell_h), "white")
            panel.paste(digit, ((cell_w - 112) // 2, 1))
            draw = ImageDraw.Draw(panel)
            draw.text((3, 116), f"pos={position:02d} index={index}", fill="black", font=font)
            draw.text((3, 130), f"V2 fail: {reason[:22]}", fill="black", font=font)
            draw.text((3, 142), f"W={old_feature[0]:.2f} slope={old_feature[1]:.2f}", fill="black", font=font)
            canvas.paste(panel, ((position % columns) * cell_w, 34 + (position // columns) * cell_h))
            writer.writerow({
                "position": position, "test_index": int(index), "true_label": int(label),
                "v2_failure_reason": reason, "width_ratio": old_feature[0], "slope": old_feature[1],
                "old_multirun": old_feature[3], "bar_thickness": persistent["test"][index, 0],
                "persistent_wide_fraction": persistent["test"][index, 1],
                "lower_multirun": lower["test_lower_multirun"][index],
                "visual_category_A_B_C": "", "visual_style": "", "should_set_accept": "",
                "reviewer_notes": "",
            })
    canvas.save(output / "pilot_60.png")
    np.save(output / "selected_indices.npy", selected)
    print(f"selected {len(selected)} of {len(false_negative)} V4 false negatives")
    print(f"saved grid: {output / 'pilot_60.png'}")
    print(f"saved audit table: {csv_path}")


if __name__ == "__main__":
    main()
