"""Radial distribution functions, nearest-neighbour distances, and space groups.

    python -m eval.distributions --source train=data/mp20/train.pt glass=samples.pt --out rdf.npz

Each RDF normalizes directed periodic pair counts by atoms x number density x shell
volume, per structure. The CSV beside the output lists every structure and its error;
failed structures are absent from the arrays, so read them with the validity counts.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError

from eval.common import read_structures, write_csv

RMAX, BINS = 6.0, 240
EDGES = np.linspace(0, RMAX, BINS + 1)


def one_structure(item):
    identifier, num_atoms, structure, error = item
    result = {"id": identifier, "num_atoms": num_atoms, "error": error}
    if structure is None:
        return result

    try:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        num_atoms, volume = len(atoms), float(atoms.get_volume())
        if volume < 0.1:
            raise ValueError("Cell volume below 0.1 A^3")
        distances = neighbor_list("d", atoms, RMAX)
        counts, _ = np.histogram(distances, bins=EDGES)
        shell_volumes = 4 * np.pi / 3 * (EDGES[1:] ** 3 - EDGES[:-1] ** 3)

        result.update(
            num_atoms=num_atoms,
            volume=volume,
            formula=structure.composition.reduced_formula,
            rdf=counts / (num_atoms * (num_atoms / volume) * shell_volumes),
            nearest=float(distances.min()) if distances.size else float("nan"),
        )
        try:
            result["spacegroup"] = SpacegroupAnalyzer(
                structure, symprec=0.1
            ).get_space_group_number()
        except SymmetryUndeterminedError as error:
            result.update(
                spacegroup=0, spacegroup_error=f"{type(error).__name__}: {error}"
            )
    except Exception as error:  # noqa: BLE001 - failed structures stay in the report
        return {
            "id": identifier,
            "num_atoms": num_atoms,
            "error": f"{type(error).__name__}: {error}",
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    arrays = {"r": (EDGES[1:] + EDGES[:-1]) / 2, "rmax": np.asarray(RMAX)}
    records = []
    for spec in args.source:
        label, source = spec.split("=", 1)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(one_structure, read_structures(source)))
        good = [row for row in rows if row["error"] is None]
        records += [
            {"source": label, **{k: v for k, v in row.items() if k != "rdf"}}
            for row in rows
        ]

        arrays[f"grall_{label}"] = np.array([row["rdf"] for row in good])
        arrays[f"gr_{label}"] = arrays[f"grall_{label}"].mean(0)
        for key, column in (
            ("nn", "nearest"),
            ("na", "num_atoms"),
            ("spacegroup", "spacegroup"),
            ("ids", "id"),
            ("formula", "formula"),
        ):
            arrays[f"{key}_{label}"] = np.array([row[column] for row in good])
        print(f"{label}: {len(good)}/{len(rows)} structures used")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    write_csv(Path(args.out).with_suffix(".csv"), records)
