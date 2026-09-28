"""Autoencoder reconstruction on complete train and validation splits, in FP32.

    python -m eval.reconstruction --config configs/mp20.yaml --autoencoder ae.pt --out recon/

`rmsd_predicted_cell` is the reported reconstruction error: Cartesian RMSD of
Hungarian-assigned atoms, each structure placed in its own *predicted* lattice.
`rmsd` uses the target lattice. Each loss term keeps the reduction used in
training (per atom, per slot, or per structure) when summed across batches.

With --decode N, N random structures per split are also decoded and saved with
their targets, as token files that any evaluator accepts (e.g. MOFChecker).
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from glass.data import AtomisticData, prepare
from glass.geometry import predicted_cell_rmsd
from glass.train_autoencoder import (
    default_device,
    load_autoencoder,
    reconstruction_loss,
)


@torch.no_grad()
def split_metrics(model, data, loss_config, batch_size=128):
    sums = {
        key: 0.0
        for key in (
            "coord",
            "type",
            "cell",
            "pair_distance",
            "accuracy",
            "type_exact",
            "count_exact",
        )
    }
    atoms = type_count = pair_weight = 0.0
    rmsd, cell_rmsd = [], []

    for start in range(0, len(data), batch_size):
        batch = slice(start, start + batch_size)
        types, coords, cells, padding = (
            data.types[batch],
            data.coords[batch],
            data.cells[batch],
            data.padding[batch],
        )
        pred, logits, pred_cell = model(types, coords, padding, cells)
        metrics = reconstruction_loss(
            pred, logits, pred_cell, coords, types, cells, padding, **loss_config
        )

        batch_atoms = (~padding).sum().item()
        batch_type_count = logits.shape[0] * logits.shape[1]
        sums["coord"] += metrics["coord"].item() * batch_atoms
        sums["accuracy"] += metrics["accuracy"].item() * batch_atoms
        sums["type"] += metrics["type"].item() * batch_type_count
        sums["cell"] += metrics["cell"].item() * len(types)
        sums["pair_distance"] += metrics["pair_distance"].item() * max(
            metrics["pair_weight"].item(), 1
        )
        atoms += batch_atoms
        type_count += batch_type_count
        pair_weight += metrics["pair_weight"].item()

        # Exact composition: the predicted atom types match as a multiset.
        predicted_types = logits.argmax(-1)
        sums["type_exact"] += (
            (predicted_types.sort(1).values == types.sort(1).values).all(1).sum().item()
        )
        sums["count_exact"] += (
            (predicted_types.ne(0).sum(1) == types.ne(0).sum(1)).sum().item()
        )

        rmsd.append(metrics["rmsd"].cpu())
        cell_rmsd.append(
            predicted_cell_rmsd(
                pred, pred_cell, metrics["aligned_target"], cells, metrics["real"]
            )
        )

    summary = {
        "structures": len(data),
        "coord": sums["coord"] / atoms,
        "type": sums["type"] / type_count,
        "cell": sums["cell"] / len(data),
        "pair_distance": sums["pair_distance"] / max(pair_weight, 1),
        "accuracy": sums["accuracy"] / atoms,
        "type_exact": sums["type_exact"] / len(data),
        "count_exact": sums["count_exact"] / len(data),
        "rmsd": torch.cat(rmsd).mean().item(),
    }
    summary["loss"] = sum(
        summary[key] * loss_config[weight]
        for key, weight in (
            ("coord", "coord_weight"),
            ("type", "type_weight"),
            ("cell", "lattice_weight"),
            ("pair_distance", "pair_distance_weight"),
        )
    )
    # Invalid predicted lattices would silently leave the mean; count them.
    cell_rmsd = torch.cat(cell_rmsd)
    valid = cell_rmsd.isfinite()
    summary["rmsd_predicted_cell"] = cell_rmsd[valid].mean().item()
    summary["invalid_predicted_cells"] = int((~valid).sum())
    return summary


@torch.no_grad()
def decode(model, data, count, seed, out, batch_size=16):
    """Decode a random subset of a split; save reconstructions and targets as tokens."""
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))[
        :count
    ]
    ids = [data.ids[i] for i in indices.tolist()]
    indices = indices.to(data.device)
    decoded = {key: [] for key in ("types", "coords", "cells")}

    for start in range(0, count, batch_size):
        batch = indices[start : start + batch_size]
        coords, logits, cells = model(
            data.types[batch],
            data.coords[batch],
            data.padding[batch],
            data.cells[batch],
        )
        decoded["types"].append(logits.argmax(-1).cpu())
        decoded["coords"].append(coords.cpu())
        decoded["cells"].append(cells.cpu())

    torch.save(
        {"ids": ids, **{key: torch.cat(value) for key, value in decoded.items()}},
        out.with_name(out.name + "_reconstructed.pt"),
    )
    targets = {
        "types": data.types[indices],
        "coords": data.coords[indices],
        "cells": data.cells[indices],
    }
    torch.save(
        {"ids": ids, **{key: value.cpu() for key, value in targets.items()}},
        out.with_name(out.name + "_targets.pt"),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decode", type=int, default=0, metavar="N")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    torch.set_float32_matmul_precision("highest")
    config = yaml.safe_load(Path(args.config).read_text())
    model, _ = load_autoencoder(args.autoencoder, args.device)
    root = prepare(**config["data"])
    args.out.mkdir(parents=True)

    report = {"autoencoder": args.autoencoder, "splits": {}}
    for split in ("train", "val"):
        data = AtomisticData(root, split, args.device)
        report["splits"][split] = split_metrics(
            model, data, config["autoencoder"]["loss"]
        )
        print(split, json.dumps(report["splits"][split]), flush=True)
        if args.decode:
            decode(model, data, args.decode, seed=0, out=args.out / split)

    (args.out / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
