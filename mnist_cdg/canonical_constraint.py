from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch

from .constraints import (
    GeometricSevenConstraint,
    SevenGeometryThresholds,
    extract_seven_geometry_features,
)
from .evaluate_bar_constraint import bar_features
from .evaluate_v7_canonical_constraint import canonical_features


@dataclass(frozen=True)
class CanonicalSevenThresholds:
    bar_thickness_min: float = 0.10
    bar_thickness_max: float = 0.35
    outside_width_difference_max: float = 0.10
    diagonal_r2_min: float = 0.0
    top_edge_rmse_max: float = 0.05
    top_edge_abs_slope_max: float = 0.20
    middle_to_bottom_width_max: float = 1.75

    @classmethod
    def from_v7_metrics(cls, path: str | Path):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        selected = report["thresholds"]
        return cls(
            diagonal_r2_min=float(selected["diagonal_r2_min"]),
            top_edge_rmse_max=float(selected["top_edge_rmse_max"]),
            top_edge_abs_slope_max=float(selected["top_edge_abs_slope_max"]),
            middle_to_bottom_width_max=float(selected["middle_to_bottom_width_max"]),
        )


class CanonicalSevenConstraint:
    """Explicit hard set for the audited canonical MNIST seven style."""

    def __init__(self, v2: SevenGeometryThresholds, canonical: CanonicalSevenThresholds):
        self.v2 = GeometricSevenConstraint(v2)
        self.canonical = canonical

    @classmethod
    def from_files(cls, v2_thresholds: str | Path, v7_metrics: str | Path):
        return cls(
            SevenGeometryThresholds.from_json(v2_thresholds),
            CanonicalSevenThresholds.from_v7_metrics(v7_metrics),
        )

    @staticmethod
    def _unit_tensor(image: torch.Tensor | np.ndarray) -> torch.Tensor:
        value = torch.as_tensor(image).detach().float().cpu().squeeze()
        if float(value.min()) < -0.01:
            value = (value + 1.0) / 2.0
        return value.clamp(0.0, 1.0).unsqueeze(0)

    def accepts(self, image: torch.Tensor | np.ndarray) -> bool:
        if not bool(self.v2.indicator(torch.as_tensor(image)).item()):
            return False
        unit = self._unit_tensor(image)
        thickness, outside_difference, _, bar_valid = bar_features(unit)
        threshold = self.canonical
        if not (
            bool(bar_valid)
            and threshold.bar_thickness_min <= thickness <= threshold.bar_thickness_max
            and outside_difference <= threshold.outside_width_difference_max
        ):
            return False
        old = extract_seven_geometry_features(image)
        top_rmse, top_slope, middle_ratio = canonical_features(unit)
        return bool(
            old.diagonal_r2 >= threshold.diagonal_r2_min
            and top_rmse <= threshold.top_edge_rmse_max
            and top_slope <= threshold.top_edge_abs_slope_max
            and middle_ratio <= threshold.middle_to_bottom_width_max
        )

    def indicator(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        batch = torch.as_tensor(images)
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        return torch.tensor([self.accepts(image) for image in batch], dtype=torch.bool)
