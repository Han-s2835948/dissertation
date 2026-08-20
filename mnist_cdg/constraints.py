from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import torch
from scipy import ndimage


@dataclass(frozen=True)
class SevenGeometryFeatures:
    width_ratio: float
    diagonal_slope: float
    diagonal_r2: float
    lower_multirun_fraction: float
    holes: int
    main_component_ratio: float
    ink_ratio: float
    valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SevenGeometryThresholds:
    pixel_threshold: float = 0.35
    top_fraction: float = 0.30
    bottom_fraction: float = 0.30
    width_ratio_min: float = 1.30
    diagonal_slope_max: float = -0.10
    lower_multirun_fraction_max: float = 0.20
    holes_max: int = 0
    main_component_ratio_min: float = 0.90

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


def _to_unit_numpy(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().float().cpu().numpy()
    else:
        array = np.asarray(image, dtype=np.float32)
    array = np.squeeze(array)
    if array.shape != (28, 28):
        raise ValueError(f"expected a 28x28 MNIST image, got {array.shape}")
    if float(array.min()) < -0.01:
        array = (array + 1.0) / 2.0
    return np.clip(array, 0.0, 1.0)


def _robust_span(columns: np.ndarray) -> float:
    if columns.size == 0:
        return 0.0
    if columns.size < 4:
        return float(columns.max() - columns.min() + 1)
    lower, upper = np.quantile(columns.astype(np.float64), [0.05, 0.95])
    return float(max(upper - lower + 1.0, 0.0))


def extract_seven_geometry_features(
    image: torch.Tensor | np.ndarray,
    pixel_threshold: float = 0.35,
    top_fraction: float = 0.30,
    bottom_fraction: float = 0.30,
) -> SevenGeometryFeatures:
    """Extract explicit morphology features used to define a seven-like set.

    The rule is deliberately non-learned and need not be differentiable: it is
    used only to evaluate the terminal event 1{Y_T in S}. Guidance gradients
    are later obtained from a separately trained h-network.
    """
    unit = _to_unit_numpy(image)
    binary = unit >= pixel_threshold
    ink = int(binary.sum())
    if ink < 3:
        return SevenGeometryFeatures(0.0, 0.0, 0.0, 0.0, 0, 0.0, ink / 784.0, False)

    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return SevenGeometryFeatures(0.0, 0.0, 0.0, 0.0, 0, 0.0, ink / 784.0, False)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main_label = int(sizes.argmax())
    main = labels == main_label
    main_ratio = float(main.sum() / max(ink, 1))

    rows, cols = np.where(main)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    height = rmax - rmin + 1
    width = cmax - cmin + 1
    if height < 4 or width < 2:
        return SevenGeometryFeatures(0.0, 0.0, 0.0, 0.0, 0, main_ratio, ink / 784.0, False)

    row_position = (rows - rmin) / max(height - 1, 1)
    top_cols = cols[row_position <= top_fraction]
    bottom_cols = cols[row_position >= 1.0 - bottom_fraction]
    top_width = _robust_span(top_cols)
    bottom_width = _robust_span(bottom_cols)
    width_ratio = float(top_width / max(bottom_width, 1.0))

    fit_rows, fit_centres = [], []
    start_row = rmin + int(round(0.20 * max(height - 1, 1)))
    for row in range(start_row, rmax + 1):
        row_cols = np.flatnonzero(main[row])
        if row_cols.size:
            fit_rows.append((row - rmin) / max(height - 1, 1))
            fit_centres.append((float(row_cols.mean()) - cmin) / max(width - 1, 1))
    if len(fit_rows) >= 3 and np.var(fit_rows) > 1e-8:
        y = np.asarray(fit_centres, dtype=np.float64)
        x = np.asarray(fit_rows, dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        ss_res = float(np.square(y - fitted).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    else:
        slope, r2 = 0.0, 0.0

    # A typical 7 has one continuous stroke in nearly every row below its top.
    # A 4 often has two separated stroke segments across several middle rows.
    # Count those rows explicitly; this remains a transparent shape rule rather
    # than a learned classifier.
    multirun_rows = 0
    observed_rows = 0
    for row in range(start_row, rmax + 1):
        row_values = main[row, cmin:cmax + 1].astype(np.int8)
        if not row_values.any():
            continue
        observed_rows += 1
        padded_row = np.pad(row_values, (1, 1), constant_values=0)
        run_count = int(np.sum(np.diff(padded_row) == 1))
        multirun_rows += int(run_count >= 2)
    lower_multirun_fraction = multirun_rows / max(observed_rows, 1)

    crop = main[rmin:rmax + 1, cmin:cmax + 1]
    padded = np.pad(crop, 1, constant_values=False)
    filled = ndimage.binary_fill_holes(padded)
    hole_mask = filled & ~padded
    _, holes = ndimage.label(hole_mask, structure=np.ones((3, 3), dtype=np.uint8))

    return SevenGeometryFeatures(
        width_ratio=width_ratio,
        diagonal_slope=float(slope),
        diagonal_r2=float(r2),
        lower_multirun_fraction=float(lower_multirun_fraction),
        holes=int(holes),
        main_component_ratio=main_ratio,
        ink_ratio=ink / 784.0,
        valid=True,
    )


class GeometricSevenConstraint:
    """Transparent hard-set membership rule for seven-like MNIST images."""

    def __init__(self, thresholds: SevenGeometryThresholds):
        self.thresholds = thresholds

    def accepts_features(self, features: SevenGeometryFeatures) -> bool:
        t = self.thresholds
        return bool(
            features.valid
            and features.width_ratio >= t.width_ratio_min
            and features.diagonal_slope <= t.diagonal_slope_max
            and features.lower_multirun_fraction <= t.lower_multirun_fraction_max
            and features.holes <= t.holes_max
            and features.main_component_ratio >= t.main_component_ratio_min
        )

    def indicator(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images if images.ndim == 4 else images.unsqueeze(0)
        else:
            array = np.asarray(images)
            batch = array if array.ndim == 4 else array[None]
        values = []
        for image in batch:
            features = extract_seven_geometry_features(
                image,
                pixel_threshold=self.thresholds.pixel_threshold,
                top_fraction=self.thresholds.top_fraction,
                bottom_fraction=self.thresholds.bottom_fraction,
            )
            values.append(self.accepts_features(features))
        return torch.tensor(values, dtype=torch.bool)
