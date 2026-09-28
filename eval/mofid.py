"""MOFid identifiers (building blocks, topology, catenation) for every structure.

    python -m eval.mofid --input samples.pt --out mofid.csv

Two comparison keys are written. `mofid_key` (strict) requires a resolved
topology; `mofid_permissive_key` follows Mofasa and also keeps failed topologies.
Keys contain only building-block SMILES, topology, and catenation.
"""

import argparse
import importlib.util
import io
import math
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from functools import cache
from numbers import Real
from pathlib import Path

from eval.common import read_structures, write_csv
from eval.setup import CACHE

INVALID_TOPOLOGIES = {"NA", "ERROR", "TIMEOUT", "MISMATCH", "UNKNOWN"}


def permissive_key(smiles, topology, catenation):
    if (
        not isinstance(smiles, str)
        or smiles in ("", "*")
        or not isinstance(topology, str)
        or not topology
        or catenation is None
        or (isinstance(catenation, Real) and math.isnan(catenation))
    ):
        return None
    topology = ",".join(component.strip() for component in topology.split(","))
    catenation = str(catenation).removesuffix(".0")
    return f"{smiles} MOFid-v1.{topology}.cat{catenation}"


def strict_key(smiles, topology, catenation):
    components = [component.strip() for component in topology.split(",")]
    if (
        smiles == "*"
        or catenation is None
        or any(
            not component or component.upper() in INVALID_TOPOLOGIES
            for component in components
        )
    ):
        return None
    return f"{smiles} MOFid-v1.{','.join(components)}.cat{catenation}"


@cache
def load_mofid():
    """Import upstream's nonstandard package layout once per worker."""
    package = CACHE / "upstream/mofid/Python"
    spec = importlib.util.spec_from_file_location(
        "mofid", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mofid"] = module
    spec.loader.exec_module(module)

    from mofid.run_mofid import cif2mofid

    return cif2mofid


def identify_one(item):
    identifier, num_atoms, structure, error = item
    row = {
        "id": identifier,
        "num_atoms": num_atoms,
        "mofid": None,
        "mofid_permissive_key": None,
        "mofid_key": None,
        "error": error,
    }
    if error:
        return row

    diagnostics = io.StringIO()
    try:
        with redirect_stdout(diagnostics), redirect_stderr(diagnostics):
            cif2mofid = load_mofid()
            with tempfile.TemporaryDirectory(prefix="glass-mofid-") as temporary:
                cif = Path(temporary) / "input.cif"
                structure.to(filename=str(cif))
                result = cif2mofid(str(cif), str(Path(temporary) / "output"))

        row.update(
            mofid=result["mofid"],
            mofid_permissive_key=permissive_key(
                result["smiles"], result["topology"], result["cat"]
            ),
            mofid_key=strict_key(result["smiles"], result["topology"], result["cat"]),
            smiles=result["smiles"],
            topology=result["topology"],
            catenation=result["cat"],
        )
        if row["mofid_key"] is None:
            row["error"] = "MOFid did not assign a complete identifier"
    except Exception as error:  # noqa: BLE001 - failures stay in the denominator
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def evaluate(source, out, workers=1):
    load_mofid()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(identify_one, read_structures(source), chunksize=1))
    write_csv(out, rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    rows = evaluate(**vars(parser.parse_args()))
    print(
        f"{sum(row['mofid_key'] is not None for row in rows)}/{len(rows)} structures identified"
    )
