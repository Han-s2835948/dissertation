from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]


def read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def eta_value(key: str) -> float:
    return float(key.split("_")[-1])


def chart(title: str, x_labels: list[str], series: list[tuple[str, list[float]]],
          y_min: float, y_max: float, y_label: str,
          reference: tuple[float, str] | None = None,
          x_axis_label: str = "Total guidance multiplier eta") -> Image.Image:
    width, height = 650, 430
    left, right, top, bottom = 70, 20, 42, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 12), title, fill="black", font=font)
    draw.line((left, top, left, top + plot_h), fill="black", width=1)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=1)
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = top + plot_h - plot_h * index / 5
        draw.line((left, y, left + plot_w, y), fill=(225, 225, 225), width=1)
        draw.text((4, y - 6), f"{value:.2f}", fill="black", font=font)
    count = len(x_labels)
    xs = [left + (plot_w * index / max(count - 1, 1)) for index in range(count)]
    for x, label in zip(xs, x_labels):
        draw.line((x, top + plot_h, x, top + plot_h + 4), fill="black")
        draw.text((x - 8, top + plot_h + 9), label, fill="black", font=font)
    draw.text((left + plot_w // 2 - 55, height - 20), x_axis_label,
              fill="black", font=font)
    draw.text((4, top - 18), y_label, fill="black", font=font)
    if reference is not None:
        value, label = reference
        y = top + plot_h * (y_max - value) / (y_max - y_min)
        draw.line((left, y, left + plot_w, y), fill=(80, 80, 80), width=1)
        draw.text((left + plot_w - 110, y - 13), label, fill=(80, 80, 80), font=font)
    for series_index, (label, values) in enumerate(series):
        color = COLORS[series_index]
        points = []
        for x, value in zip(xs, values):
            y = top + plot_h * (y_max - value) / (y_max - y_min)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        legend_x = left + 8 + series_index * 185
        draw.rectangle((legend_x, top + 8, legend_x + 12, top + 20), fill=color)
        draw.text((legend_x + 17, top + 8), label, fill="black", font=font)
    return image


def combine(left: Image.Image, right: Image.Image, output: Path) -> None:
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-study", default="outputs_formal/mnist/formal_eta_multiseed_v1/metrics.json")
    parser.add_argument("--geometry-study", default="outputs_formal/mnist/formal_eta_geometry_multiseed_v1/metrics.json")
    parser.add_argument("--independent-audit", default="outputs_formal/mnist/independent_classifier_audit_v1/metrics.json")
    parser.add_argument("--h-classifier", default="outputs_formal/mnist/h_independent_audit_v1/metrics.json")
    parser.add_argument("--h-geometry", default="outputs_formal/mnist/h_geometry_independent_audit_v1/metrics.json")
    parser.add_argument("--quality", default="outputs_formal/mnist/quality_diversity_audit_v1/metrics.json")
    parser.add_argument("--output-dir", default="outputs_formal/mnist/formal_eta_analysis_v1")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # The JSON files remain the numerical source; this script only turns their
    # saved values into the dissertation figures.
    classifier = read(args.classifier_study)
    geometry = read(args.geometry_study)
    independent = read(args.independent_audit)
    quality = read(args.quality)

    classifier_items = sorted(classifier["pooled"].items(), key=lambda item: eta_value(item[0]))
    labels = [key.replace("eta_", "") for key, _ in classifier_items]
    primary_rates = [value["success_rate"] for _, value in classifier_items]
    independent_rates = []
    for key, _ in classifier_items:
        values = [seed[key]["independent_target_rate"] for seed in independent["generated"].values()]
        independent_rates.append(sum(values) / len(values))
    left = chart("Classifier-defined set", labels,
                 [("Set classifier", primary_rates), ("Independent classifier", independent_rates)],
                 0, 1, "Target rate")
    geometry_items = sorted(geometry["pooled"].items(), key=lambda item: eta_value(item[0]))
    labels_g = [key.replace("eta_", "") for key, _ in geometry_items]
    right = chart("Calculable geometric set", labels_g,
                  [("Geometric membership", [v["success_rate"] for _, v in geometry_items]),
                   ("Semantic classifier-7", [v["classifier_target_rate"] for _, v in geometry_items])],
                  0, 1, "Rate")
    combine(left, right, output / "success_vs_eta.png")

    h_classifier = read(args.h_classifier)["time_resolved_metrics"]
    h_geometry = read(args.h_geometry)["time_resolved_metrics"]
    tau_labels = [f"{row['tau']:.1f}" for row in h_classifier]
    left = chart("Independent h discrimination audit", tau_labels,
                 [("Classifier h", [row["roc_auc"] for row in h_classifier]),
                  ("Geometry h", [row["roc_auc"] for row in h_geometry])],
                 0.45, 1.0, "Terminal-event AUC", reference=(0.5, "Chance"),
                 x_axis_label="Generative time tau")
    weighted_values = ([row["mean_beta_weighted_grad_log_h_norm"] for row in h_classifier] +
                       [row["mean_beta_weighted_grad_log_h_norm"] for row in h_geometry])
    right = chart("VP-SDE weighted guidance magnitude", tau_labels,
                  [("Classifier h", [row["mean_beta_weighted_grad_log_h_norm"] for row in h_classifier]),
                   ("Geometry h", [row["mean_beta_weighted_grad_log_h_norm"] for row in h_geometry])],
                  0, max(weighted_values) * 1.08, "Mean beta * ||grad log h||",
                  x_axis_label="Generative time tau")
    combine(left, right, output / "h_time_resolved_audit.png")

    quality_items = sorted(quality["generated"].items(), key=lambda item: eta_value(item[0]))
    labels_q = [key.replace("eta_", "") for key, _ in quality_items]
    diversity_values = [v["target_feature_pairwise_cosine_distance"] for _, v in quality_items]
    left = chart("Within-target feature diversity", labels_q,
                 [("Generated target samples", diversity_values)], 0.25, 0.45,
                 "Mean pairwise cosine distance",
                 reference=(quality["real_target"]["feature_pairwise_cosine_distance"], "Real sevens"))
    nearest_mean = [v["target_mean_nearest_real_feature_distance"] for _, v in quality_items]
    nearest_p90 = [v["target_p90_nearest_real_feature_distance"] for _, v in quality_items]
    right = chart("Distance to real-seven feature bank", labels_q,
                  [("Mean", nearest_mean), ("90th percentile", nearest_p90)],
                  0, max(nearest_p90) * 1.08, "Cosine distance")
    combine(left, right, output / "quality_diversity_tradeoff.png")
    print(f"saved formal plots to {output}")


if __name__ == "__main__":
    main()
