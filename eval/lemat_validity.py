"""Per-structure LeMat-GenBench validity (charge, distance, plausibility checks).

Run with the benchmark's environment, from the repository root:
    cache/upstream/genbench/.venv/bin/python -m eval.lemat_validity --input samples.pt --out validity.csv
"""

import argparse
import json
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval.common import read_structures, validity_summary, write_csv

UPSTREAM = Path(__file__).resolve().parents[1] / "cache/upstream/genbench/src"
CONFIG = {
    "charge_tolerance": 0.1,
    "distance_scaling_factor": 0.5,
    "plausibility_min_atomic_density": 0.00001,
    "plausibility_max_atomic_density": 0.5,
    "plausibility_min_mass_density": 0.01,
    "plausibility_max_mass_density": 25.0,
    "plausibility_check_format": True,
    "plausibility_check_symmetry": True,
}

preprocessor = process_arguments = None


def initialize():
    global preprocessor, process_arguments
    sys.path.insert(0, str(UPSTREAM))

    # An optional upstream import emits a malformed logging call unrelated to validity.
    raise_exceptions, logging.raiseExceptions = logging.raiseExceptions, False
    try:
        from lemat_genbench.preprocess.validity_preprocess import ValidityPreprocessor
    finally:
        logging.raiseExceptions = raise_exceptions

    preprocessor = ValidityPreprocessor(**CONFIG)
    process_arguments = preprocessor._get_process_attributes()


def score(item):
    identifier, num_atoms, structure, error = item
    row = {
        "id": identifier,
        "num_atoms": num_atoms,
        "error": error,
        "valid": False,
        "charge_valid": False,
        "distance_valid": False,
        "plausibility_valid": False,
        "charge_deviation": None,
    }
    if structure is None:
        return row

    try:
        properties = preprocessor.process_structure(
            structure, **process_arguments, original_source=identifier
        ).properties
        row.update(
            valid=bool(properties["overall_valid"]),
            charge_valid=bool(properties["charge_valid"]),
            distance_valid=bool(properties["distance_valid"]),
            plausibility_valid=bool(properties["plausibility_valid"]),
            charge_deviation=float(properties["charge_deviation"]),
        )
    except Exception as error:  # noqa: BLE001 - failed checks stay in the denominator
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def evaluate(source, out, workers=1):
    initialize()
    structures = read_structures(source)
    if workers == 1:
        rows = list(map(score, structures))
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            rows = list(pool.map(score, structures, chunksize=16))

    write_csv(out, rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    print(json.dumps(validity_summary(evaluate(**vars(parser.parse_args()))), indent=2))
