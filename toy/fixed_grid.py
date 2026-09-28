"""Fixed-grid toy: how hard is it for a flow to realize one fixed set of N points?

The target is one fixed 2D grid of N sites. Two permutation-equivariant flows
transport Gaussian noise to the grid:
  naive  - each training pair uses a fresh random permutation of the target;
  eqot   - each pair uses the Hungarian (equivariant OT) source-to-target coupling.
The control is the GLASS decoder without a latent ("decoder"): N learned slots,
self-attention, and a Hungarian-matched reconstruction loss.

Train one run:   python -m toy.fixed_grid train --n 24 --seed 0 --method naive
Sample a run:    python -m toy.fixed_grid sample --run runs/toy/n24_seed0_naive
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from glass.matching import hungarian
from glass.models import SelfAttentionLayer
from glass.train_autoencoder import default_device

METHODS = ("naive", "eqot", "decoder")
BATCH_SIZE = 512
POOL_SIZE = 65_536
EVAL_SAMPLES = 1024
EULER_STEPS = 64
WIDTH, HEADS, LAYERS = 64, 4, 3
OPTIMIZER = {"lr": 1e-3, "weight_decay": 1e-6, "betas": (0.9, 0.999), "eps": 1e-8}

# Flows stop at an absolute PI-RMSD in [-1, 1]^2 coordinates; the decoder uses a
# stricter tolerance relative to the minimum grid spacing.
FLOW_STOP_RMSD = 0.01
DECODER_STOP_RMSD_PER_SPACING = 0.01


def eval_interval(step):
    if step < 20_000:
        return 100
    if step < 100_000:
        return 500
    if step < 500_000:
        return 2_000
    return 5_000


def target_grid(n):
    """Near-square grid on [-1, 1]^2, removing surplus sites nearest the origin."""
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)

    xx, yy = np.meshgrid(
        np.linspace(-1, 1, cols, dtype=np.float32),
        np.linspace(-1, 1, rows, dtype=np.float32),
    )
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    if len(points) > n:
        radius = np.sum(points**2, axis=1)
        drop = np.lexsort((points[:, 1], points[:, 0], radius))[: len(points) - n]
        points = np.delete(points, drop, axis=0)
    return torch.from_numpy(points)


def assignment(points, target):
    """For each point, the index of its Hungarian-matched target site."""
    cost = (target[None, :, None] - points[:, None]).square().sum(-1)
    counts = torch.full((len(points),), len(target), device=points.device)
    return hungarian(cost, counts)


def min_spacing(target):
    distance = (target[:, None] - target[None]).norm(dim=-1)
    distance.fill_diagonal_(float("inf"))
    return float(distance.min())


class VelocityField(nn.Module):
    """Permutation-equivariant Transformer on (x, y, t) per particle."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH,
            HEADS,
            2 * WIDTH,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.output = nn.Linear(WIDTH, 2)

    def forward(self, x, t):
        t = t.expand(-1, x.shape[1], 1)
        return self.output(self.encoder(self.input(torch.cat([x, t], dim=-1))))


class SlotDecoder(nn.Module):
    """GLASS decoder reduced to learned slots, self-attention, and a 2D head."""

    def __init__(self, n):
        super().__init__()
        self.slots = nn.Embedding(n, WIDTH)
        self.slot_proj = nn.Linear(WIDTH, WIDTH)
        self.layers = nn.ModuleList(
            SelfAttentionLayer(WIDTH, HEADS) for _ in range(LAYERS)
        )
        self.norm = nn.LayerNorm(WIDTH)
        self.output = nn.Linear(WIDTH, 2)

    def forward(self):
        slots = self.slot_proj(self.slots.weight[None])
        for layer in self.layers:
            slots = layer(slots)
        return self.output(self.norm(slots))


@torch.no_grad()
def sample(model, x):
    """Uniform Euler integration from noise x at t = 0 to t = 1."""
    model.eval()
    dt = 1 / EULER_STEPS
    for i in range(EULER_STEPS):
        t = torch.full((len(x), 1, 1), i * dt, device=x.device)
        x = x + dt * model(x, t)
    return x


@torch.no_grad()
def set_metrics(points, target):
    """PI-RMSD and site occupancy for a batch of generated sets [batch, N, 2]."""
    matched = target[assignment(points, target)]
    rmsd = (points - matched).square().sum(-1).mean(-1).sqrt()

    # A set fails when two points share their nearest target site.
    nearest = (points[:, :, None] - target).square().sum(-1).argmin(-1)
    occupied = torch.zeros(len(points), len(target), device=points.device)
    occupied.scatter_(1, nearest, 1.0)
    unoccupied = 1 - occupied.sum(-1) / len(target)

    return {
        "rmsd": rmsd.mean().item(),
        "collision_rate": (unoccupied > 0).float().mean().item(),
        "unoccupied_site_fraction": unoccupied.mean().item(),
    }


def train_flow(n, seed, method, max_steps, device):
    torch.manual_seed(seed + 1000 * n)
    target = target_grid(n).to(device)
    # A finite, reusable pool of Gaussian sources; eqOT couples each once, in advance.
    rng = np.random.default_rng(seed + 10_000 * n)
    pool_x0 = torch.from_numpy(
        rng.standard_normal((POOL_SIZE, n, 2)).astype(np.float32)
    ).to(device)
    if method == "eqot":
        pool_x1 = torch.cat(
            [target[assignment(x, target)] for x in pool_x0.split(BATCH_SIZE)]
        )

    model = VelocityField().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
    index_rng = torch.Generator(device=device).manual_seed(seed + 20_000 * n)
    time_rng = torch.Generator(device=device).manual_seed(seed + 30_000 * n)
    perm_rng = torch.Generator(device=device).manual_seed(seed + 40_000 * n)
    eval_x0 = torch.randn(
        EVAL_SAMPLES, n, 2, generator=torch.Generator().manual_seed(seed + 50_000 * n)
    ).to(device)
    history = []

    for step in range(1, max_steps + 1):
        model.train()
        idx = torch.randint(
            POOL_SIZE, (BATCH_SIZE,), device=device, generator=index_rng
        )
        x0 = pool_x0[idx]
        if method == "eqot":
            x1 = pool_x1[idx]
        else:
            x1 = target[
                torch.rand(BATCH_SIZE, n, device=device, generator=perm_rng).argsort(1)
            ]

        t = torch.rand(BATCH_SIZE, 1, 1, device=device, generator=time_rng)
        xt = (1 - t) * x0 + t * x1
        loss = (model(xt, t) - (x1 - x0)).square().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval(step) and step != max_steps:
            continue
        history.append({"step": step, **set_metrics(sample(model, eval_x0), target)})
        print(json.dumps(history[-1]), flush=True)
        if history[-1]["rmsd"] <= FLOW_STOP_RMSD:
            break

    return model, history


def train_decoder(n, seed, max_steps, device):
    torch.manual_seed(seed + 1000 * n)
    target = target_grid(n).to(device)
    model = SlotDecoder(n).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
    perm_rng = torch.Generator(device=device).manual_seed(seed + 40_000 * n)
    stop = DECODER_STOP_RMSD_PER_SPACING * min_spacing(target)
    history = []

    for step in range(1, max_steps + 1):
        model.train()
        prediction = model()
        # Present the targets in a random order; Hungarian matching recovers a correspondence.
        shuffled = target[torch.randperm(n, device=device, generator=perm_rng)]
        matched = shuffled[assignment(prediction.detach(), shuffled)]
        loss = (prediction - matched).square().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval(step) and step != max_steps:
            continue
        model.eval()
        with torch.no_grad():
            history.append({"step": step, **set_metrics(model(), target)})
        print(json.dumps(history[-1]), flush=True)
        if history[-1]["rmsd"] <= stop:
            break

    return model, history


def train(n, seed, method, max_steps, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if method == "decoder":
        model, history = train_decoder(n, seed, max_steps, device)
    else:
        model, history = train_flow(n, seed, method, max_steps, device)

    run = {"n": n, "seed": seed, "method": method, "stop_step": history[-1]["step"]}
    torch.save({**run, "model": model.state_dict()}, out / "model.pt")
    (out / "history.json").write_text(json.dumps({**run, "history": history}, indent=2))


def sample_run(run, count, seed, device):
    """Draw sets from a trained run, report metrics, and plot a few against the grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    checkpoint = torch.load(run / "model.pt", map_location=device)
    n, method = checkpoint["n"], checkpoint["method"]
    target = target_grid(n).to(device)

    if method == "decoder":
        model = SlotDecoder(n).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            points = model()
    else:
        model = VelocityField().to(device)
        model.load_state_dict(checkpoint["model"])
        x0 = torch.randn(count, n, 2, generator=torch.Generator().manual_seed(seed)).to(
            device
        )
        points = sample(model, x0)

    metrics = set_metrics(points, target)
    print(json.dumps({"n": n, "method": method, "sets": len(points), **metrics}))
    torch.save(points.cpu(), run / "samples.pt")

    shown = min(4, len(points))
    figure, axes = plt.subplots(1, shown, figsize=(3 * shown, 3), squeeze=False)
    for ax, generated in zip(axes[0], points.cpu()):
        ax.scatter(*target.cpu().T, s=60, color="lightsteelblue", label="target")
        ax.scatter(*generated.T, s=12, color="black", label="generated")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].legend(loc="upper left", fontsize=7)
    figure.suptitle(f"{method}, N = {n}")
    figure.savefig(run / "samples.png", dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--n", type=int, required=True)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--method", choices=METHODS, required=True)
    train_parser.add_argument("--max-steps", type=int, default=1_000_000)
    train_parser.add_argument("--out")
    train_parser.add_argument("--device", default=default_device())

    sample_parser = commands.add_parser("sample")
    sample_parser.add_argument("--run", required=True)
    sample_parser.add_argument("--count", type=int, default=EVAL_SAMPLES)
    sample_parser.add_argument("--seed", type=int, default=80)
    sample_parser.add_argument("--device", default=default_device())

    args = parser.parse_args()
    if args.command == "train":
        out = args.out or f"runs/toy/n{args.n}_seed{args.seed}_{args.method}"
        train(args.n, args.seed, args.method, args.max_steps, out, args.device)
    else:
        sample_run(args.run, args.count, args.seed, args.device)


if __name__ == "__main__":
    main()
