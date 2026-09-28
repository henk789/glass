"""Read every candidate structure and write per-candidate tables, keeping failures.

Every evaluator reports rates over *all requested* candidates: structures whose
lattice cannot be built, or that fail a relaxation, stay in the denominator.
"""

import csv
import json
from pathlib import Path

import torch
from pymatgen.core import Structure

from glass.data import structure_from_tokens


def read_structures(path):
    """Yield (id, atom count, structure or None, error or None) for every candidate.

    `path` is a token file (`samples.pt` or a prepared split such as `train.pt`) or a
    directory of CIFs with a `manifest.json` written by a relaxation step.
    """
    path = Path(path)

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
        count = len(data["types"])
        ids = data.get("ids", [f"sample_{i + 1:06d}" for i in range(count)])
        assert len(set(ids)) == count, "Duplicate structure IDs"

        for identifier, types, coords, cell in zip(
            ids, data["types"], data["coords"], data["cells"], strict=True
        ):
            num_atoms = int((types != 0).sum())
            try:
                structure = structure_from_tokens(
                    types.numpy(), coords.numpy(), cell.numpy()
                )
                yield str(identifier), num_atoms, structure, None
            except Exception as error:  # noqa: BLE001 - invalid geometry stays counted
                yield (
                    str(identifier),
                    num_atoms,
                    None,
                    f"{type(error).__name__}: {error}",
                )
        return

    inventory = json.loads((path / "manifest.json").read_text())
    assert inventory["complete"], f"Unfinished candidate inventory: {path}"

    for record in inventory["records"]:
        identifier, num_atoms = record["id"], record.get("num_atoms")
        if record.get("error"):
            yield identifier, num_atoms, None, record["error"]
            continue
        try:
            structure = Structure.from_file(path / record["file"])
            yield identifier, num_atoms, structure, None
        except Exception as error:  # noqa: BLE001 - unreadable outputs stay counted
            yield identifier, num_atoms, None, f"{type(error).__name__}: {error}"


def export(source, out):
    """Write one CIF per candidate plus a manifest that records every failure."""
    out = Path(out)
    out.mkdir(parents=True)
    records = []

    for identifier, num_atoms, structure, error in read_structures(source):
        record = {"id": identifier, "num_atoms": num_atoms, "error": error}
        if structure is not None:
            try:
                structure.to(filename=str(out / f"{identifier}.cif"))
                record["file"] = f"{identifier}.cif"
            except Exception as error:  # noqa: BLE001 - failed exports stay counted
                record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)

    manifest = {
        "input": str(Path(source).resolve()),
        "complete": True,
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = dict.fromkeys(key for row in rows for key in row)

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validity_summary(rows):
    passed = sum(row["valid"] and not row["error"] for row in rows)
    return {
        "total": len(rows),
        "passed": passed,
        "errors": sum(bool(row["error"]) for row in rows),
        "validity": passed / len(rows),
    }
