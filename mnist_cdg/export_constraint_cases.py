from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

from mnist_cdg.common import ensure_dir
from mnist_cdg.constraints import SevenGeometryThresholds
from mnist_cdg.evaluate_bar_constraint import BarThresholds, apply_bar
from mnist_cdg.evaluate_geometric_constraint import FEATURE_NAMES, metrics, predictions_from_features


OUTPUT = Path("outputs_formal/mnist/geometric_constraint_casebook")
V2_OUTPUT = Path("outputs_formal/mnist/geometric_constraint_v2")
V3_OUTPUT = Path("outputs_formal/mnist/geometric_constraint_v3_bar")
PER_PAGE = 80


def v2_failure(features: np.ndarray, threshold: SevenGeometryThresholds) -> str:
    f = dict(zip(FEATURE_NAMES, features))
    failed = []
    if not bool(f["valid"]):
        failed.append("invalid")
    if f["width_ratio"] < threshold.width_ratio_min:
        failed.append("top/bottom")
    if f["diagonal_slope"] > threshold.diagonal_slope_max:
        failed.append("slope")
    if f["lower_multirun_fraction"] > threshold.lower_multirun_fraction_max:
        failed.append("multi-stroke")
    if f["holes"] > threshold.holes_max:
        failed.append("hole")
    if f["main_component_ratio"] < threshold.main_component_ratio_min:
        failed.append("component")
    return "+".join(failed) or "accepted"


def bar_failure(bar: np.ndarray, threshold: BarThresholds) -> str:
    thickness, difference, _, valid = bar
    failed = []
    if not bool(valid):
        failed.append("no-bar")
    if thickness < threshold.thickness_ratio_min:
        failed.append("bar-thin")
    if thickness > threshold.thickness_ratio_max:
        failed.append("bar-thick")
    if difference > threshold.outside_width_difference_max:
        failed.append("outside-wide")
    return "+".join(failed) or "accepted"


def save_pages(folder: Path, dataset, indices: np.ndarray, labels: np.ndarray,
               old_features: np.ndarray, bar_features: np.ndarray,
               old_threshold: SevenGeometryThresholds, bar_threshold: BarThresholds | None,
               version: str, case_name: str) -> list[str]:
    ensure_dir(folder)
    pages = []
    font = ImageFont.load_default()
    columns, cell_w, cell_h = 8, 132, 132
    for page_number, start in enumerate(range(0, len(indices), PER_PAGE), 1):
        selected = indices[start:start + PER_PAGE]
        rows = max(1, (len(selected) + columns - 1) // columns)
        canvas = Image.new("RGB", (columns * cell_w, 34 + rows * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((7, 7), f"{version} {case_name}: page {page_number}, cases {start + 1}-{start + len(selected)} of {len(indices)}",
                  fill="black", font=font)
        for position, index in enumerate(selected):
            image, label = dataset[int(index)]
            digit = Image.fromarray((image.squeeze().numpy() * 255).astype(np.uint8), mode="L")
            digit = digit.resize((84, 84)).convert("RGB")
            panel = Image.new("RGB", (cell_w, cell_h), "white")
            panel.paste(digit, ((cell_w - 84) // 2, 1))
            panel_draw = ImageDraw.Draw(panel)
            old = old_features[index]
            bar = bar_features[index]
            reason = v2_failure(old, old_threshold)
            if bar_threshold is not None and reason == "accepted":
                reason = bar_failure(bar, bar_threshold)
            panel_draw.text((2, 88), f"#{index} true={label}", fill="black", font=font)
            panel_draw.text((2, 102), f"W={old[0]:.2f} th={bar[0]:.3f}", fill="black", font=font)
            panel_draw.text((2, 116), f"out={bar[1]:.3f} {reason[:17]}", fill="black", font=font)
            canvas.paste(panel, ((position % columns) * cell_w, 34 + (position // columns) * cell_h))
        filename = f"page_{page_number:02d}.png"
        canvas.save(folder / filename)
        pages.append(filename)
    if pages:
        page_images = [Image.open(folder / name).convert("RGB") for name in pages]
        combined = Image.new(
            "RGB",
            (max(page.width for page in page_images), sum(page.height for page in page_images)),
            "white",
        )
        top = 0
        for page in page_images:
            combined.paste(page, (0, top))
            top += page.height
            page.close()
        combined.save(folder / "all.png")
    return pages


def main() -> None:
    output = ensure_dir(OUTPUT)
    dataset = datasets.MNIST("data", train=False, download=False, transform=transforms.ToTensor())
    old_npz = np.load(V2_OUTPUT / "test_features.npz")
    labels = old_npz["labels"]
    old_features = old_npz["features"]
    bar_features = np.load(V3_OUTPUT / "bar_features.npz")["test"]
    old_threshold = SevenGeometryThresholds.from_json(V2_OUTPUT / "thresholds.json")
    old_prediction = predictions_from_features(old_features, old_threshold)

    versions = {
        "v2_original": {
            "description": "Original: top/bottom width ratio, slope, multi-run rows, holes, connected component",
            "bar_threshold": None,
            "prediction": old_prediction,
        },
        "v3_strict": {
            "description": "V2 plus bar thickness/H in [0.10, 0.35] and outside width difference/H <= 0.10",
            "bar_threshold": BarThresholds(0.10, 0.35, 0.10),
            "prediction": apply_bar(old_prediction, bar_features, BarThresholds(0.10, 0.35, 0.10)),
        },
        "v3_widened": {
            "description": "V2 plus widened bar thickness/H in [0.08, 0.40] and outside width difference/H <= 0.10",
            "bar_threshold": BarThresholds(0.08, 0.40, 0.10),
            "prediction": apply_bar(old_prediction, bar_features, BarThresholds(0.08, 0.40, 0.10)),
        },
    }
    truth = labels == 7
    report = {}
    markdown = [
        "# MNIST geometric hard-constraint casebook",
        "",
        "FP means a non-7 was accepted. FN means a real 7 was missed.",
        "Image labels: `W` is top/bottom width ratio, `th` is top-bar thickness/digit height, and `out` is outside-bar width difference/digit height.",
        "",
    ]
    csv_path = output / "all_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["version", "case", "test_index", "true_label", "failure_reason",
                  "width_ratio", "diagonal_slope", "lower_multirun_fraction", "holes",
                  "main_component_ratio", "bar_thickness_ratio", "outside_width_difference", "bar_start_ratio"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for version, info in versions.items():
            prediction = info["prediction"]
            false_positive = np.flatnonzero(prediction & ~truth)
            false_negative = np.flatnonzero(~prediction & truth)
            version_folder = ensure_dir(output / version)
            pages_by_case = {}
            for case_name, indices in (("false_positive", false_positive), ("false_negative", false_negative)):
                pages_by_case[case_name] = save_pages(
                    version_folder / case_name, dataset, indices, labels, old_features, bar_features,
                    old_threshold, info["bar_threshold"], version, case_name,
                )
                for index in indices:
                    old = old_features[index]
                    bar = bar_features[index]
                    reason = v2_failure(old, old_threshold)
                    if info["bar_threshold"] is not None and reason == "accepted":
                        reason = bar_failure(bar, info["bar_threshold"])
                    writer.writerow({
                        "version": version, "case": case_name, "test_index": int(index),
                        "true_label": int(labels[index]), "failure_reason": reason,
                        "width_ratio": old[0], "diagonal_slope": old[1],
                        "lower_multirun_fraction": old[3], "holes": int(old[4]),
                        "main_component_ratio": old[5], "bar_thickness_ratio": bar[0],
                        "outside_width_difference": bar[1], "bar_start_ratio": bar[2],
                    })
            version_metrics = metrics(labels, prediction)
            report[version] = {
                "description": info["description"],
                "bar_threshold": None if info["bar_threshold"] is None else asdict(info["bar_threshold"]),
                "metrics": version_metrics,
                "false_positive_pages": pages_by_case["false_positive"],
                "false_negative_pages": pages_by_case["false_negative"],
            }
            markdown.extend([f"## {version}", "", info["description"], "",
                             f"- Precision: {version_metrics['precision']:.4%}",
                             f"- Recall: {version_metrics['recall']:.4%}",
                             f"- False positives: {len(false_positive)}", f"- False negatives: {len(false_negative)}", ""])
            for case_name in ("false_positive", "false_negative"):
                markdown.extend([f"### {case_name}", ""])
                markdown.extend([f"- [{page}]({version}/{case_name}/{page})" for page in pages_by_case[case_name]])
                markdown.append("")
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({key: value["metrics"] for key, value in report.items()}, indent=2))
    print(f"saved complete casebook to {output}")


if __name__ == "__main__":
    main()
