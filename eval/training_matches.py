"""Match valid generated structures against every training structure of the same formula.

    python -m eval.training_matches --input samples.pt --training data/qmof150/train.pt \
        --validity mofchecker.csv --out matches.csv

Uses pymatgen's StructureMatcher. An unmatched structure differs geometrically
from all training structures at these tolerances; this is not topology novelty.
"""

import argparse
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher

from eval.common import read_structures, write_csv

SETTINGS = {
    "ltol": 0.2,
    "stol": 0.3,
    "angle_tol": 5,
    "primitive_cell": True,
    "scale": True,
    "attempt_supercell": True,
    "allow_subset": False,
}
references = matcher = None


def initialize(groups):
    global references, matcher
    references = groups
    matcher = StructureMatcher(comparator=ElementComparator(), **SETTINGS)


def match_one(item):
    identifier, num_atoms, structure, error = item
    row = {
        "id": identifier,
        "num_atoms": num_atoms,
        "formula": structure.composition.reduced_formula,
        "training_match": None,
        "matched": False,
        "references_checked": 0,
        "error": error,
    }
    try:
        for reference_id, reference in references.get(row["formula"], []):
            row["references_checked"] += 1
            if matcher.fit(reference, structure):
                row.update(training_match=reference_id, matched=True)
                break
    except Exception as error:  # noqa: BLE001 - a failed match is neither matched nor unmatched
        row.update(matched=None, error=f"{type(error).__name__}: {error}")
    return row


def evaluate(source, training, validity, out, workers=1):
    groups = defaultdict(list)
    for identifier, _, structure, error in read_structures(training):
        assert error is None, f"Unreadable training structure {identifier}: {error}"
        groups[structure.composition.reduced_formula].append((identifier, structure))

    candidates = list(read_structures(source))
    valid = pd.read_csv(validity, dtype={"id": str}).set_index("id")
    assert set(valid.index) == {item[0] for item in candidates}, (
        "Validity report covers other structures"
    )
    valid_ids = set(valid.index[valid.valid.eq(True) & valid.error.isna()])
    selected = [item for item in candidates if item[0] in valid_ids]

    initialize(groups)
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        rows = list(pool.map(match_one, selected, chunksize=8))
    write_csv(out, rows)

    matched = sum(row["matched"] is True for row in rows)
    unmatched = sum(row["matched"] is False for row in rows)
    summary = {
        "settings": {**SETTINGS, "comparator": "ElementComparator"},
        "requested": len(candidates),
        "valid": len(rows),
        "matched": matched,
        "unmatched": unmatched,
        "errors": len(rows) - matched - unmatched,
        "unmatched_new_formula": sum(
            row["matched"] is False and row["references_checked"] == 0 for row in rows
        ),
        "valid_unmatched_yield": unmatched / len(candidates),
    }
    Path(out).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", dest="source", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--validity", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
