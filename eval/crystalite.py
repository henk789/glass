"""Crystalite's de novo generation (DNG) metrics on 10,000 generated MP20 structures.

Run with Crystalite's environment, from the repository root:
    cache/upstream/crystalite/.venv/bin/python -m eval.crystalite --input samples.pt --out crystalite.json

This calls the pinned upstream implementation: Crystalite's MP20 training split for
novelty, its validation split for Wasserstein distances, and NequIP-OAM-L
relaxation for energies above hull and SUN/MSUN.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval.common import write_csv
from eval.setup import CACHE

ASSETS = CACHE / "crystalite"
SEED = 123
THERMO = {
    "batch_size": 64,
    "relax_steps": 200,
    "ehull_method": "mp2020_like",
    "mlip": "nequip",
    "nequip_relax_mode": "sequential",
    "nequip_optimizer": "FIRE",
    "nequip_cell_filter": "frechet",
    "nequip_fmax": 0.005,
    "nequip_max_force_abort": 1e6,
}


def crystalite_items(samples):
    """GLASS tokens -> Crystalite's token dictionaries; empty samples get a dummy atom."""
    items, errors = [], []
    for types, coords, cell in zip(
        samples["types"], samples["coords"], samples["cells"]
    ):
        padding = types.eq(0)
        if padding.all():
            items.append(
                {
                    "A0": torch.tensor([0]),
                    "F1": torch.zeros(1, 3),
                    "Y1": torch.tensor([4.0, 4.0, 4.0, 0.0, 0.0, 0.0]),
                    "pad_mask": torch.tensor([False]),
                }
            )
            errors.append("Empty generated structure")
        else:
            items.append(
                {
                    "A0": types,
                    "F1": coords.float(),
                    "Y1": cell.float(),
                    "pad_mask": padding,
                }
            )
            errors.append(None)
    return items, errors


def evaluate(source, out, expected_count=10_000, thermo_count=10_000, device="cuda"):
    sys.path.insert(0, str(CACHE / "upstream/crystalite"))
    from src.data.mp20_tokens import MP20Tokens
    from src.eval.diagnostics import _compute_sample_diagnostics
    from src.eval.dng_eval import (
        collect_constructed_structures,
        compute_evaluator_metrics,
        compute_novelty_metrics,
        compute_structure_stats_metrics,
        compute_sun_metrics,
        float_or_nan,
    )
    from src.eval.stability import _compute_thermo_metrics
    from src.utils.dataset import dataset_to_structures
    from src.utils.stability_logger import StabilityLogger, _ThermoConfig

    samples = torch.load(source, map_location="cpu")
    items, errors = crystalite_items(samples)
    assert len(items) == expected_count, f"Expected {expected_count}, got {len(items)}"

    splits = {
        split: dataset_to_structures(
            MP20Tokens(
                root=str(ASSETS / "mp20"), augment_translate=False, split=split, nmax=20
            )
        )
        for split in ("train", "val")
    }
    evaluator = compute_evaluator_metrics(
        items,
        limit=len(items),
        ref_structs=splits["val"],
        sample_seed=SEED,
        include_wasserstein=True,
        wasserstein_max_samples=expected_count,
    )
    diagnostics = _compute_sample_diagnostics(evaluator.pred_crys_list)
    novelty = compute_novelty_metrics(
        items, splits["train"], limit=len(items), minimum_nary=1
    )
    structure_stats = compute_structure_stats_metrics(
        items, total_count=len(items), include_summary_stats=True
    )

    metrics = {
        "valid_rate": evaluator.valid_rate,
        "comp_valid_rate": evaluator.comp_valid_rate,
        "struct_valid_rate": evaluator.struct_valid_rate,
        **evaluator.dist_metrics,
        **{f"diagnostic/{key}": value for key, value in diagnostics.items()},
        **{
            f"structure_stats/{key}": value
            for key, value in structure_stats.metrics.items()
        },
        "unique_rate": novelty.unique_rate,
        "novel_rate": novelty.novel_rate,
        "un_rate": novelty.un_rate,
    }

    if thermo_count:
        thermo_config = _ThermoConfig(
            batch_size=THERMO["batch_size"],
            relax_steps=THERMO["relax_steps"],
            ppd_path=str(ASSETS / "thermo/2023-02-07-ppd-mp.pkl"),
            device=device,
            ehull_method=THERMO["ehull_method"],
            mlip=THERMO["mlip"],
            nequip_compile_path=str(ASSETS / "nequip/NequIP-OAM-L-ase.nequip.pt2"),
            nequip_relax_mode=THERMO["nequip_relax_mode"],
            nequip_optimizer=THERMO["nequip_optimizer"],
            nequip_cell_filter=THERMO["nequip_cell_filter"],
            nequip_fmax=THERMO["nequip_fmax"],
            nequip_max_force_abort=THERMO["nequip_max_force_abort"],
        )
        logger = StabilityLogger(gamma_cfg=None, thermo_cfg=thermo_config)
        structures = collect_constructed_structures(
            items,
            pred_crys_list=evaluator.pred_crys_list,
            count=min(thermo_count, len(items)),
        )
        metrics.update(
            _compute_thermo_metrics(
                logger, structures, tag="eval", step=0, enabled=True, show_progress=True
            )
        )
        # SUN and MSUN relax the unique-and-novel set itself: direct intersections.
        sun = compute_sun_metrics(
            novelty.novelty_metrics,
            thermo_logger=logger,
            tag="",
            step=0,
            enabled=True,
            base_seed=SEED,
            sun_target=thermo_count,
            show_progress=True,
        )
        metrics.update(sun.thermo_metrics)
        metrics.update(sun.summary_metrics)

        # Crystalite's Table 2 calls the <= 0.1 eV/atom rate "Stable" and its
        # unique-and-novel intersection "SUN"; the upstream logger calls these
        # metastable and MSUN.
        metrics["paper_table/stable_rate"] = metrics["eval/thermo_metastable_rate"]
        metrics["paper_table/SUN"] = metrics["MSUN"]

    metrics = {key: float_or_nan(value) for key, value in metrics.items()}
    metrics["summary/num_samples_requested"] = float(len(items))
    metrics["summary/num_valid"] = float(sum(c.valid for c in evaluator.pred_crys_list))

    rows = []
    for identifier, error, crystal in zip(
        samples["ids"], errors, evaluator.pred_crys_list
    ):
        rows.append(
            {
                "id": identifier,
                "formula": crystal.structure.composition.reduced_formula
                if crystal.constructed
                else None,
                "error": error,
                "comp_valid": bool(crystal.comp_valid),
                "struct_valid": bool(crystal.struct_valid),
                "valid": bool(crystal.valid) and error is None,
            }
        )
    out = Path(out)
    write_csv(out.with_suffix(".validity.csv"), rows)
    out.write_text(
        json.dumps(
            {"input": str(source), "thermo_count": thermo_count, "metrics": metrics},
            indent=2,
        )
        + "\n"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-count", type=int, default=10_000)
    parser.add_argument(
        "--thermo-count", type=int, default=10_000, help="0 skips stability"
    )
    parser.add_argument("--device", default="cuda")
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
