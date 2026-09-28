"""QMOF150 validity, novelty, and uniqueness from MOFChecker and MOFid reports.

    python -m eval.mof_metrics --samples samples.pt --mofchecker mofchecker.csv \
        --mofid mofid.csv --training-mofid train_mofid.csv --training data/qmof150/train.pt \
        --out metrics.json

Every rate uses all requested structures as its denominator. A structure is novel
when it has a MOFid key absent from the training structures' keys; VNU counts
distinct keys among valid and novel structures. Metrics are reported for both the
strict and the Mofasa-style permissive MOFid key (see eval/mofid.py). Formula
novelty counts samples whose reduced composition does not occur in training.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

KEYS = {"strict": "mofid_key", "permissive": "mofid_permissive_key"}
SIZE_BINS = ((1, 50), (51, 80), (81, 110), (111, 150))


def reduced_formulas(types):
    formulas = []
    for row in np.asarray(types):
        counts = np.bincount(row, minlength=119)
        counts[0] = 0
        present = np.flatnonzero(counts)
        if not len(present):
            formulas.append(None)
            continue
        reduced = counts[present] // np.gcd.reduce(counts[present])
        formulas.append(tuple(zip(present.tolist(), reduced.tolist())))
    return formulas


def metrics(validity, identifiers, training, key):
    rows = validity.merge(identifiers[["id", key]], on="id", validate="one_to_one")
    valid = rows.valid & rows.error.isna()
    identified = rows[key].notna()
    novel = identified & ~rows[key].isin(training[key].dropna())

    report = {
        "requested": len(rows),
        "valid": int(valid.sum()),
        "identified": int(identified.sum()),
        "novel": int(novel.sum()),
        "valid_novel": int((valid & novel).sum()),
        "unique": int(rows[key].nunique()),
        "vnu": int(rows.loc[valid & novel, key].nunique()),
        "training_coverage": float(training[key].notna().mean()),
    }
    report["by_size"] = [
        {
            "atoms": [low, high],
            "requested": int(selected.sum()),
            "valid": int((valid & selected).sum()),
            "novel": int((novel & selected).sum()),
            "valid_novel": int((valid & novel & selected).sum()),
        }
        for low, high in SIZE_BINS
        for selected in [rows.num_atoms.between(low, high)]
    ]
    return report


def evaluate(samples, mofchecker, mofid, training_mofid, training, out):
    samples = torch.load(samples, map_location="cpu")
    validity, identifiers = pd.read_csv(mofchecker), pd.read_csv(mofid)
    training_identifiers = pd.read_csv(training_mofid)
    assert set(validity.id) == set(identifiers.id) == set(samples["ids"]), (
        "Reports cover different structures"
    )
    assert not training_identifiers.id.duplicated().any(), "Duplicate training IDs"

    training_formulas = set(
        reduced_formulas(torch.load(training, map_location="cpu")["types"])
    )
    formula_novel = sum(
        formula is not None and formula not in training_formulas
        for formula in reduced_formulas(samples["types"])
    )

    report = {
        name: metrics(validity, identifiers, training_identifiers, key)
        for name, key in KEYS.items()
    }
    report["formula_novel"] = formula_novel
    with open(out, "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("samples", "mofchecker", "mofid", "training-mofid", "training", "out"):
        parser.add_argument(f"--{name}", required=True)
    report = evaluate(**vars(parser.parse_args()))
    for name in KEYS:
        r = report[name]
        print(
            f"{name}: valid {r['valid'] / r['requested']:.2%}, novel {r['novel'] / r['requested']:.2%}, "
            f"VNU {r['vnu'] / r['requested']:.2%}"
        )
