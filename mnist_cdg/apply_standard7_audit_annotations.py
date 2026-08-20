from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("outputs_formal/mnist/standard7_visual_audit")

# Codex visual pre-audit positions. A = canonical standard 7, B = readable but
# non-canonical 7, C = ambiguous/non-7. Human verification remains required.
POPULATION_A = {3, 5, 8, 11, 12, 16, 17, 18, 19, 20, 26, 29, 31, 34, 38, 39,
                47, 48, 49, 50, 51, 52, 53, 54, 58}
POPULATION_C = {33, 37, 57}
ACCEPTED_A = {0, 1, 2, 3, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23,
              27, 29, 30, 31, 33, 34, 36, 37, 38, 41, 42, 43, 47, 48, 49, 50,
              52, 53, 54, 59}
ACCEPTED_C = {58}


def category(position: int, a_set: set[int], c_set: set[int]) -> tuple[str, str, str]:
    if position in a_set:
        return "A", "canonical_standard_7", "Clear top bar and single descending main stroke."
    if position in c_set:
        return "C", "ambiguous", "Visually ambiguous or closer to another digit."
    return "B", "readable_noncanonical_7", "Readable 7 with curved, crossed, boxy, or near-vertical style."


def annotate(name: str, a_set: set[int], c_set: set[int]) -> list[dict[str, str]]:
    source = ROOT / name / "annotations.csv"
    output = ROOT / name / "annotations_codex.csv"
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        cat, style, note = category(int(row["position"]), a_set, c_set)
        row["visual_category_A_B_C"] = cat
        row["visual_style"] = style
        row["canonical_standard_7"] = "yes" if cat == "A" else "no"
        row["reviewer_notes"] = note
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    population = annotate("random_official_7", POPULATION_A, POPULATION_C)
    accepted = annotate("random_v3_accepted", ACCEPTED_A, ACCEPTED_C)
    false_positive = annotate("all_official_false_positive", set(), set(range(20)))
    population_a = [row for row in population if row["visual_category_A_B_C"] == "A"]
    accepted_population_a = [row for row in population_a if row["v3_strict_accept"] == "1"]
    accepted_a = [row for row in accepted if row["visual_category_A_B_C"] == "A"]
    summary = {
        "reviewer": "Codex visual pre-audit; human verification required",
        "canonical_definition": "clear top bar and single descending main stroke; crossed/curved/boxy/near-vertical styles are B",
        "random_official_7": {
            "sample_size": len(population),
            "A": sum(row["visual_category_A_B_C"] == "A" for row in population),
            "B": sum(row["visual_category_A_B_C"] == "B" for row in population),
            "C": sum(row["visual_category_A_B_C"] == "C" for row in population),
            "estimated_standard_recall": len(accepted_population_a) / max(len(population_a), 1),
            "standard_A_accepted": len(accepted_population_a),
            "standard_A_total": len(population_a),
        },
        "random_v3_accepted": {
            "sample_size": len(accepted),
            "A": len(accepted_a),
            "B": sum(row["visual_category_A_B_C"] == "B" for row in accepted),
            "C": sum(row["visual_category_A_B_C"] == "C" for row in accepted),
            "estimated_canonical_purity": len(accepted_a) / len(accepted),
            "estimated_readable_7_purity_A_plus_B": sum(row["visual_category_A_B_C"] != "C" for row in accepted) / len(accepted),
        },
        "all_official_false_positive_count": len(false_positive),
    }
    (ROOT / "summary_codex.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
