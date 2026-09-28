"""MOFChecker 0.9.6 structural validity, as used by ADiT and Mofasa.

Run with the MOFChecker environment (python -m eval.setup mofchecker):
    cache/environments/mofchecker/bin/python -m eval.mofchecker --input samples.pt --out mofchecker.csv

A structure is valid when it contains carbon, hydrogen, and a metal and none of the
problem checks fires. A check that raises is recorded as an error, never as passed.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version

from eval.common import read_structures, validity_summary, write_csv

PRESENCE = ("has_carbon", "has_hydrogen", "has_metal")
PROBLEMS = (
    "has_atomic_overlaps",
    "has_overcoordinated_c",
    "has_overcoordinated_n",
    "has_overcoordinated_h",
    "has_undercoordinated_c",
    "has_undercoordinated_n",
    "has_undercoordinated_rare_earth",
    "has_undercoordinated_alkali_alkaline",
    "has_lone_molecule",
    "has_suspicious_terminal_oxo",
    "has_high_charges",
    "has_geometrically_exposed_metal",
)
# Reported separately; not part of validity.
DESCRIPTORS = PRESENCE + PROBLEMS + ("has_3d_connected_graph",)


def check_one(item):
    from mofchecker import MOFChecker

    identifier, num_atoms, structure, error = item
    row = {"id": identifier, "num_atoms": num_atoms, "valid": False, "error": error}
    if error:
        return row

    try:
        checker = MOFChecker(structure)
        for key in DESCRIPTORS:
            # MOFChecker 0.9.6 spells this attribute with a typo.
            attribute = (
                "has_suspicicious_terminal_oxo"
                if key == "has_suspicious_terminal_oxo"
                else key
            )
            value = getattr(checker, attribute)
            if value is None:
                raise ValueError(f"Checker did not complete {attribute}")
            row[key] = bool(value)

        row["valid"] = all(row[key] for key in PRESENCE) and not any(
            row[key] for key in PROBLEMS
        )
    except Exception as error:  # noqa: BLE001 - failed checks stay in the denominator
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def evaluate(source, out, workers=1):
    # A missing EQeq installation would otherwise silently fail every charge check.
    from pyeqeq.main import run_on_cif  # noqa: F401

    assert version("mofchecker") == "0.9.6", "The protocol requires mofchecker==0.9.6"

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(check_one, read_structures(source)))
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
