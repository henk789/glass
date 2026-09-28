"""MP20 and QMOF150: pinned downloads, crystal tokens, and GPU-resident batches.

Prepare a dataset ahead of time (training does this automatically):
    python -m glass.data configs/qmof150.yaml

A crystal is a padded set of atoms: atomic numbers (0 = padding), fractional
coordinates, and six lattice tokens (log lengths, cosines of the angles). Real
atoms come first, sorted by a symmetry-derived canonical order.
"""

import argparse
import json
import math
import multiprocessing
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ADiT releases of both benchmarks, pinned to exact revisions.
SOURCES = {
    "mp20": (
        "chaitjo/MP20_ADiT",
        "cc448f4a85a35d7488d446888a69f8b9cc3aac2d",
        "raw/all.csv",
    ),
    "qmof150": (
        "chaitjo/QMOF150_ADiT",
        "45e5d0f79a5cbd6a1c3008c3ec8602584977b9b8",
        "processed/qmof150.pt",
    ),
}
NMAX = {"mp20": 20, "qmof150": 150}
SPLITS = Path(__file__).with_name("splits.json")


def cell_tokens(lengths, angles):
    lengths, angles = np.asarray(lengths, np.float32), np.asarray(angles, np.float32)
    return np.concatenate((np.log(lengths), np.cos(np.deg2rad(angles)))).astype(
        np.float32
    )


def lattice_from_tokens(cell):
    cell = np.asarray(cell, np.float32)
    lengths = np.exp(cell[:3])
    angles = np.degrees(np.arccos(cell[3:].clip(-1, 1)))
    return Lattice.from_parameters(*lengths.tolist(), *angles.tolist())


def structure_from_tokens(types, coords, cell):
    """Decoded tokens -> pymatgen Structure; raises for an invalid lattice."""
    types, coords, cell = (np.asarray(x) for x in (types, coords, cell))
    real = types > 0
    if not real.any():
        raise ValueError("Empty generated structure")
    if not np.isfinite(cell).all():
        raise ValueError("Invalid lattice metric")

    # Positive determinant alone also admits metrics with two negative eigenvalues.
    np.linalg.cholesky(
        [[1, cell[5], cell[4]], [cell[5], 1, cell[3]], [cell[4], cell[3], 1]]
    )
    lengths = np.exp(cell[:3])
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Invalid lattice lengths")

    if not np.isfinite(coords[real]).all():
        raise ValueError("Nonfinite atomic coordinates")
    angles = np.degrees(np.arccos(cell[3:]))

    return Structure(
        Lattice.from_parameters(*lengths.tolist(), *angles.tolist()),
        types[real],
        coords[real],
    )


def canonical_order(structure):
    """Order symmetry groups by electronegativity, Wyckoff letter, and size."""
    symmetrized = SpacegroupAnalyzer(
        structure, symprec=0.01, angle_tolerance=5
    ).get_symmetrized_structure()
    groups, symbols = symmetrized.equivalent_indices, symmetrized.wyckoff_symbols

    def key(index):
        group = groups[index]
        element = structure[group[0]].specie
        electronegativity = float(element.X)
        letter = "".join(c for c in symbols[index].lower() if c.isalpha())[-1]

        return (
            electronegativity if math.isfinite(electronegativity) else float("inf"),
            ord(letter) - ord("a"),
            len(group),
            element.Z,
            min(group),
        )

    order = [i for j in sorted(range(len(groups)), key=key) for i in sorted(groups[j])]
    assert sorted(order) == list(range(len(structure))), (
        "Ordering lost or duplicated sites"
    )
    return order


def tokenize(record):
    """One raw structure -> padded tokens in canonical atom order."""
    dataset, identifier, numbers, frac, lengths, angles = record

    if dataset == "qmof150":
        structure = Structure(
            Lattice.from_parameters(
                *np.asarray(lengths).tolist(), *np.asarray(angles).tolist()
            ),
            numbers,
            np.asarray(frac, np.float32) % 1,
        ).get_primitive_structure()
        numbers, frac = structure.atomic_numbers, structure.frac_coords
        lengths, angles = structure.lattice.abc, structure.lattice.angles

    # Find symmetry in the same float32 representation used for training.
    cell = cell_tokens(lengths, angles)
    frac = (torch.from_numpy(np.asarray(frac, np.float32)) % 1).numpy()
    structure = Structure(lattice_from_tokens(cell), list(numbers), frac)
    order = canonical_order(structure)

    n, nmax = len(numbers), NMAX[dataset]
    types = torch.zeros(nmax, dtype=torch.long)
    coords = torch.zeros(nmax, 3)
    types[:n] = torch.as_tensor(np.asarray(numbers)[order], dtype=torch.long)
    coords[:n] = torch.from_numpy(frac[order])
    return identifier, types, coords, torch.from_numpy(cell)


def raw_structures(dataset, path):
    if dataset == "mp20":
        import pandas as pd

        for row in pd.read_csv(path).itertuples():
            # MP20 uses Niggli-reduced cells.
            s = Structure.from_str(row.cif, fmt="cif").get_reduced_structure()
            yield (
                dataset,
                str(row.material_id),
                s.atomic_numbers,
                s.frac_coords,
                s.lattice.abc,
                s.lattice.angles,
            )
    else:
        data, slices, _ = torch.load(path, map_location="cpu", weights_only=False)

        for i, identifier in enumerate(data["id"]):
            a, b = slices["atom_types"][i : i + 2]
            c, d = slices["frac_coords"][i : i + 2]
            yield (
                dataset,
                str(identifier),
                data["atom_types"][a:b].numpy(),
                data["frac_coords"][c:d].numpy(),
                data["lengths"][i].numpy(),
                data["angles"][i].numpy(),
            )


def prepare(dataset, root, workers=8):
    """Download, tokenize, and split a dataset once; later calls reuse the files."""
    root = Path(root)
    if (root / "train.pt").exists():
        return root

    # pymatgen warns about rounded CIF coordinates and missing electronegativities.
    warnings.simplefilter("ignore", UserWarning)

    from huggingface_hub import hf_hub_download

    repo, revision, filename = SOURCES[dataset]
    path = hf_hub_download(
        repo, filename, repo_type="dataset", revision=revision, cache_dir=root / "raw"
    )

    with multiprocessing.Pool(workers) as pool:
        tokens = pool.map(tokenize, raw_structures(dataset, path), chunksize=64)
    lookup = {identifier: rest for identifier, *rest in tokens}

    root.mkdir(parents=True, exist_ok=True)
    for split, ids in json.loads(SPLITS.read_text())[dataset].items():
        types, coords, cells = (torch.stack(x) for x in zip(*(lookup[i] for i in ids)))
        split_data = {
            "ids": ids,
            "types": types,
            "coords": coords,
            "cells": cells,
            "padding": types == 0,
        }
        torch.save(split_data, root / f"{split}.pt")

    return root


class AtomisticData:
    """One host-to-device load; batches only index GPU-resident tensors."""

    def __init__(self, root, split, device):
        items = torch.load(Path(root) / f"{split}.pt", map_location=device)
        self.ids = items["ids"]
        self.types = items["types"]
        self.coords = items["coords"]
        self.cells = items["cells"]
        self.padding = items["padding"]
        self.device = torch.device(device)

        # Hungarian rows require real atoms first, followed by type-zero padding.
        first_padding = (~self.padding).sum(1, keepdim=True)
        expected = torch.arange(self.types.shape[1], device=device) >= first_padding
        assert torch.equal(expected, self.padding), (
            "Noncontiguous padding corrupts matching"
        )

    def __len__(self):
        return len(self.types)

    def batch(self, size, generator):
        index = torch.randint(
            len(self), (size,), device=self.device, generator=generator
        )
        return (
            self.types[index],
            self.coords[index],
            self.cells[index],
            self.padding[index],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config", help="configuration whose data section names the dataset"
    )
    config = yaml.safe_load(Path(parser.parse_args().config).read_text())
    print(prepare(**config["data"]))
