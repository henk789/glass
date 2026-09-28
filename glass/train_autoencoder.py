"""Train the GLASS autoencoder: python -m glass.train_autoencoder --config ... --out ..."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from glass.data import AtomisticData, prepare
from glass.geometry import gram, periodic_delta, predicted_cell_rmsd
from glass.matching import hungarian, periodic_squared_distance
from glass.models import Autoencoder


def reconstruction_loss(
    pred,
    logits,
    pred_cell,
    target,
    types,
    cell,
    padding,
    coord_weight,
    type_weight,
    lattice_weight,
    pair_distance_weight,
    pair_distance_inner,
    pair_distance_cutoff,
):
    """Hungarian-matched reconstruction of a padded crystal by the decoder's slots.

    Every real target atom is assigned to a distinct slot; unassigned slots must
    predict the empty type 0. Coordinates are fractional, compared through the
    nearest periodic image and measured in Cartesian units via the Gram matrix.
    """
    metric = gram(cell)
    atom_counts = (~padding).sum(1)
    total_atoms = atom_counts.sum()
    slots = logits.shape[1]

    with torch.no_grad():
        # Matching cost = the change in the weighted loss if atom i is placed in slot j.
        squared_distance = periodic_squared_distance(target, pred, metric)
        coordinate_cost = (
            coord_weight * squared_distance / (3 * atom_counts[:, None, None])
        )

        # Assigning an atom replaces this slot's empty-class CE with its type CE.
        log_prob = logits.log_softmax(-1).transpose(1, 2)
        atom_log_prob = log_prob.gather(1, types[:, :, None].expand(-1, -1, slots))
        empty_log_prob = log_prob[:, :1]
        type_cost = type_weight * (empty_log_prob - atom_log_prob) / slots

        assignment = hungarian(coordinate_cost + type_cost, atom_counts)

    # Reorder the target into slot order; unmatched slots receive padding (type 0).
    target = target.gather(1, assignment[..., None].expand(-1, -1, 3))
    target_types = types.gather(1, assignment)
    real = ~padding.gather(1, assignment)

    displacement = periodic_delta(target, pred, metric)
    squared_error = torch.einsum("bni,bij,bnj->bn", displacement, metric, displacement)
    atom_error = squared_error * real
    coord_loss = atom_error.sum() / (3 * total_atoms)

    # Cross-entropy averaged over all slots, including empty ones.
    target_logits = logits.gather(-1, target_types[..., None]).squeeze(-1)
    type_loss = (logits.logsumexp(-1) - target_logits).mean()
    cell_loss = F.mse_loss(pred_cell, cell)

    # Relative error of interatomic distances, tapered smoothly to zero between
    # the inner radius and the cutoff so that short-range geometry dominates.
    pairs = (real[:, :, None] & real[:, None, :]).triu(1)
    target_delta = periodic_delta(target[:, :, None], target[:, None], metric)
    predicted_delta = periodic_delta(pred[:, :, None], pred[:, None], metric)
    target_distance = (
        torch.einsum("bnmi,bij,bnmj->bnm", target_delta, metric, target_delta)
        .clamp_min(1e-12)
        .sqrt()
    )
    predicted_distance = (
        torch.einsum("bnmi,bij,bnmj->bnm", predicted_delta, metric, predicted_delta)
        .clamp_min(1e-12)
        .sqrt()
    )

    relative_error = (
        (predicted_distance - target_distance) / target_distance.clamp_min(1e-6)
    ).square()
    taper = (
        (target_distance - pair_distance_inner)
        / (pair_distance_cutoff - pair_distance_inner)
    ).clamp(0, 1)
    pair_weight = 0.5 * (1 + torch.cos(math.pi * taper)) * pairs
    pair_loss = (pair_weight * relative_error).sum() / pair_weight.sum().clamp_min(1)

    loss = (
        coord_weight * coord_loss
        + type_weight * type_loss
        + lattice_weight * cell_loss
        + pair_distance_weight * pair_loss
    )

    # Detach before sqrt: AOT autograd can otherwise evaluate 0/0 at exact recovery.
    rmsd = (atom_error.detach().sum(1) / atom_counts).sqrt()
    accuracy = ((logits.argmax(-1) == target_types) & real).sum() / total_atoms

    return {
        "loss": loss,
        "coord": coord_loss.detach(),
        "type": type_loss.detach(),
        "cell": cell_loss.detach(),
        "pair_distance": pair_loss.detach(),
        "rmsd": rmsd,
        "pair_weight": pair_weight.sum().detach(),
        "accuracy": accuracy,
        "aligned_target": target,
        "real": real,
    }


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compiled(function, device):
    """CUDA graphs through torch.compile; needs Triton, which PyTorch ships on Linux."""
    if device.startswith("cuda") and importlib.util.find_spec("triton"):
        return torch.compile(function, fullgraph=True, mode="reduce-overhead")
    return function


def optimizer_and_schedule(model, cfg, device):
    """AdamW with linear warmup and cosine decay; the LR stays on the GPU for CUDA graphs."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=torch.tensor(cfg["lr"], device=device),
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
        weight_decay=cfg["weight_decay"],
        fused=device.startswith("cuda"),
    )
    warmup, steps = cfg["warmup"], cfg["steps"]

    def multiplier(step):
        if step <= warmup:
            return step / warmup
        progress = (step - warmup) / (steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def log_metrics(path, step, **metrics):
    # Per-structure tensors (e.g. RMSD) are logged as their batch mean.
    values = {
        key: torch.as_tensor(value).float().mean().item()
        for key, value in metrics.items()
    }
    record = {"step": step, **values}
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def load_autoencoder(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = Autoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


@torch.no_grad()
def validate(model, val_data, loss_config, generator):
    types, coords, cells, padding = val_data.batch(min(512, len(val_data)), generator)
    pred, logits, pred_cell = model(types, coords, padding, cells)
    metrics = reconstruction_loss(
        pred, logits, pred_cell, coords, types, cells, padding, **loss_config
    )

    # The reported reconstruction error places atoms in the *predicted* cell.
    predicted_cell = predicted_cell_rmsd(
        pred, pred_cell, metrics.pop("aligned_target"), cells, metrics.pop("real")
    )
    valid = predicted_cell.isfinite()
    metrics["rmsd_predicted_cell"] = predicted_cell[valid].mean()
    metrics["invalid_predicted_cells"] = (~valid).sum()
    return {"val/" + key: value for key, value in metrics.items()}


def train(config, out, device):
    out = Path(out)
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    cfg = config["autoencoder"]["train"]
    model_config = config["autoencoder"]["model"]
    loss_config = config["autoencoder"]["loss"]

    root = prepare(**config["data"])
    train_data = AtomisticData(root, "train", device)
    val_data = AtomisticData(root, "val", device)
    assert model_config["nmax"] == train_data.types.shape[1], "nmax != padding width"

    torch.manual_seed(cfg["seed"])
    model = Autoencoder(**model_config).to(device)
    optimizer, schedule = optimizer_and_schedule(model, cfg, device)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 17)

    def objective(types, coords, cells, padding):
        pred, logits, pred_cell = model(types, coords, padding, cells)
        return reconstruction_loss(
            pred, logits, pred_cell, coords, types, cells, padding, **loss_config
        )

    objective = compiled(objective, device)

    for step in range(1, cfg["steps"] + 1):
        model.train()
        if device.startswith("cuda"):
            torch.compiler.cudagraph_mark_step_begin()

        types, coords, cells, padding = train_data.batch(cfg["batch_size"], generator)
        losses = objective(types, coords, cells, padding)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        schedule.step()

        if step == 1 or step % cfg["log_interval"] == 0 or step == cfg["steps"]:
            train_metrics = {
                "train/" + key: value
                for key, value in losses.items()
                if key not in ("aligned_target", "real")
            }
            model.eval()
            val_metrics = validate(model, val_data, loss_config, generator)
            log_metrics(out / "metrics.jsonl", step, **train_metrics, **val_metrics)

        if step % cfg["checkpoint_interval"] == 0 or step == cfg["steps"]:
            checkpoint = {
                "model": model.state_dict(),
                "model_config": model_config,
                "step": step,
            }
            name = "model.pt" if step == cfg["steps"] else f"checkpoints/step_{step}.pt"
            (out / name).parent.mkdir(exist_ok=True)
            torch.save(checkpoint, out / name)

    return out / "model.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    train(yaml.safe_load(Path(args.config).read_text()), args.out, args.device)


if __name__ == "__main__":
    main()
