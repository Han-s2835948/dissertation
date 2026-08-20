from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path("outputs_formal/mnist/visual_audit_pilot")

# Visual pre-audit by Codex from pilot_60.png.  These are intentionally kept
# separate from MNIST ground-truth labels and must not be treated as ground truth.
A_STANDARD = {1, 7, 9, 10, 12, 15, 16, 17, 19, 20, 21, 23, 32, 37, 49, 50, 51, 52, 53, 54, 58, 59}
B_CROSSED = {0, 3, 5, 6, 8, 13, 14, 22, 25, 27, 29, 30, 31, 34, 36, 42, 56, 57}
B_VERTICAL_OR_BOXY = {4, 26, 38}
B_CURVED = {18, 24, 47, 55}
B_FAINT = {43}
C_LOOKS_LIKE_2 = {28, 33}
C_LOOKS_LIKE_1 = {11, 35, 39, 40, 41, 45, 46, 48}
C_BROKEN_OR_NO_BAR = {2, 44}


def annotation(position: int) -> tuple[str, str, str, str]:
    if position in A_STANDARD:
        return "A", "clear_standard_7", "yes", "Clear 7; the geometry should ideally recover it."
    if position in B_CROSSED:
        return "B", "crossed_or_branched_7", "depends_on_scope", "Recognisable 7 with an extra cross/branch; include if S should cover diverse handwriting."
    if position in B_VERTICAL_OR_BOXY:
        return "B", "vertical_or_boxy_7", "depends_on_scope", "Recognisable but weakly diagonal/boxy 7; a mandatory slope rule is risky."
    if position in B_CURVED:
        return "B", "curved_7", "depends_on_scope", "Recognisable curved 7, but it departs from the canonical straight-stroke geometry."
    if position in B_FAINT:
        return "B", "faint_or_small_7", "depends_on_scope", "Recognisable 7 with weak/broken pixels; sensitive to binarisation."
    if position in C_LOOKS_LIKE_2:
        return "C", "ambiguous_resembles_2", "no_for_canonical_S", "Official label is 7, but the image visually resembles 2; reasonable to exclude from canonical S."
    if position in C_LOOKS_LIKE_1:
        return "C", "ambiguous_resembles_1", "no_for_canonical_S", "No convincing top bar; visually close to 1/hook, so exclusion is reasonable."
    if position in C_BROKEN_OR_NO_BAR:
        return "C", "ambiguous_broken_or_no_bar", "no_for_canonical_S", "Broken or unclear seven structure; reasonable to exclude from canonical S."
    raise KeyError(position)


def main() -> None:
    source = ROOT / "pilot_annotations.csv"
    output = ROOT / "pilot_annotations_codex.csv"
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 pilot rows, got {len(rows)}")
    for row in rows:
        category, style, accept, note = annotation(int(row["position"]))
        row["visual_category_A_B_C"] = category
        row["visual_style"] = style
        row["should_set_accept"] = accept
        row["reviewer_notes"] = note
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    category_counts = Counter(row["visual_category_A_B_C"] for row in rows)
    style_counts = Counter(row["visual_style"] for row in rows)
    summary = {
        "sample": "60 fixed random false negatives from V4 lower-multirun on MNIST test",
        "reviewer": "Codex visual pre-audit; user/human verification required",
        "category_definition": {
            "A": "clear standard 7; should be accepted",
            "B": "recognisable but non-canonical 7; depends on scope of S",
            "C": "visually ambiguous; reasonable to reject from a canonical-7 set",
        },
        "category_counts": dict(sorted(category_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
    }
    (ROOT / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved Codex pre-annotations: {output}")


if __name__ == "__main__":
    main()
