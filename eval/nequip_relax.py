"""Pre-relax structures with Crystalite's NequIP-OAM-L protocol (FIRE, Frechet cell filter).

Run with Crystalite's environment, from the repository root:
    cache/upstream/crystalite/.venv/bin/python -m eval.nequip_relax --input samples.pt --out relaxed/
The output directory holds one CIF per relaxed structure and a manifest that
records every failure, so it can be passed to any evaluator as --input.
"""

import argparse
import json
import sys
from pathlib import Path

from eval.common import read_structures
from eval.setup import CACHE

MODEL = CACHE / "crystalite/nequip/NequIP-OAM-L-ase.nequip.pt2"
SETTINGS = {
    "steps": 200,
    "optimizer": "FIRE",
    "cell_filter": "frechet",
    "fmax": 0.005,
    "max_force_abort": 1e6,
}


def relax(source, out, device="cuda"):
    sys.path.insert(0, str(CACHE / "upstream/crystalite"))
    from src.utils.sample_stats import make_nequip_relaxer

    _, relaxer, _, _ = make_nequip_relaxer(
        compile_path=MODEL,
        stability_device=device,
        optimizer_name=SETTINGS["optimizer"],
        cell_filter=SETTINGS["cell_filter"],
        fmax=SETTINGS["fmax"],
        max_force_abort=SETTINGS["max_force_abort"],
    )

    out = Path(out)
    out.mkdir(parents=True)
    records = []
    for identifier, num_atoms, structure, error in read_structures(source):
        record = {"id": identifier, "num_atoms": num_atoms, "error": error}
        if structure is not None:
            try:
                result = relaxer.relax(
                    structure, steps=SETTINGS["steps"], verbose=False
                )
                result["final_structure"].to(filename=str(out / f"{identifier}.cif"))
                record.update(
                    file=f"{identifier}.cif",
                    energy_eV=float(result["trajectory"].energies[-1]),
                    steps=int(result["nsteps"]),
                )
            except Exception as error:  # noqa: BLE001 - failed relaxations stay counted
                record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
        if len(records) % 100 == 0:
            print(f"relaxed {len(records)}", flush=True)

    manifest = {
        "protocol": {"model": str(MODEL), **SETTINGS},
        "complete": True,
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    print(relax(**vars(parser.parse_args())))
