"""Train the latent flow on a frozen autoencoder: python -m glass.train_flow ..."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from glass.data import AtomisticData, prepare
from glass.models import LatentFlow
from glass.train_autoencoder import (
    compiled,
    default_device,
    load_autoencoder,
    log_metrics,
    optimizer_and_schedule,
)


def bf16(device, enabled):
    return (
        torch.autocast("cuda", torch.bfloat16)
        if enabled and device.startswith("cuda")
        else nullcontext()
    )


@torch.no_grad()
def encode(model, data, batch_size, use_bf16):
    latents = []

    for start in range(0, len(data), batch_size):
        batch = slice(start, start + batch_size)
        with bf16(str(data.device), use_bf16):
            z = model.encoder(
                data.types[batch],
                data.coords[batch],
                data.padding[batch],
                data.cells[batch],
            )
        latents.append(z.float())

    return torch.cat(latents)


def train(config, autoencoder, out, device):
    out = Path(out)
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    cfg = config["flow"]["train"]

    root = prepare(**config["data"])
    ae, ae_checkpoint = load_autoencoder(autoencoder, device)
    ae.requires_grad_(False)
    latents = {
        split: encode(
            ae,
            AtomisticData(root, split, device),
            cfg["encode_batch_size"],
            cfg["encode_bf16"],
        )
        for split in ("train", "val")
    }

    # The flow models standardized latents; sampling undoes this with the stored statistics.
    mean = latents["train"].mean(0, keepdim=True)
    std = latents["train"].std(0, keepdim=True).clamp_min(1e-6)
    z_train = (latents["train"] - mean) / std
    z_val = (latents["val"] - mean) / std

    torch.manual_seed(cfg["seed"])
    flow_config = config["flow"]["model"]
    flow = LatentFlow(**flow_config).to(device)
    optimizer, schedule = optimizer_and_schedule(flow, cfg, device)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 29)
    ema = {name: p.detach().clone() for name, p in flow.named_parameters()}

    def objective(x0, x1, t):
        # Linear interpolation path; the target velocity is x1 - x0.
        xt = (1 - t) * x0 + t * x1
        with bf16(device, cfg["bf16"]):
            prediction = flow(xt, t)
        return F.mse_loss(prediction.float(), x1 - x0)

    objective = compiled(objective, device)

    def pair(z):
        count = min(cfg["batch_size"], len(z))
        x1 = z[torch.randint(len(z), (count,), device=device, generator=generator)]
        x0 = torch.randn(x1.shape, device=device, generator=generator)
        t = torch.rand(len(x1), 1, device=device, generator=generator)
        return x0, x1, t

    for step in range(1, cfg["steps"] + 1):
        x0, x1, t = pair(z_train)
        if device.startswith("cuda"):
            torch.compiler.cudagraph_mark_step_begin()
        loss = objective(x0, x1, t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()

        # Warm up the EMA decay so early checkpoints are not dominated by initialization.
        decay = min(cfg["ema_decay"], (1 + step) / (10 + step))
        with torch.no_grad():
            for name, p in flow.named_parameters():
                ema[name].mul_(decay).add_(p, alpha=1 - decay)

        if step == 1 or step % cfg["log_interval"] == 0 or step == cfg["steps"]:
            train_loss = loss.item()
            with torch.no_grad():
                x0, x1, t = pair(z_val)
                with bf16(device, cfg["bf16"]):
                    prediction = flow((1 - t) * x0 + t * x1, t)
                val_loss = F.mse_loss(prediction.float(), x1 - x0)
            log_metrics(
                out / "metrics.jsonl", step, train_loss=train_loss, val_loss=val_loss
            )

        if step % cfg["checkpoint_interval"] == 0 or step == cfg["steps"]:
            # Bundle the AE and latent statistics so a flow checkpoint alone can sample.
            checkpoint = {
                "model": ema,
                "flow_config": flow_config,
                "mean": mean,
                "std": std,
                "step": step,
                "autoencoder": {
                    "model": ae.state_dict(),
                    "model_config": ae_checkpoint["model_config"],
                },
            }
            name = "flow.pt" if step == cfg["steps"] else f"checkpoints/step_{step}.pt"
            (out / name).parent.mkdir(exist_ok=True)
            torch.save(checkpoint, out / name)

    return out / "flow.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    train(
        yaml.safe_load(Path(args.config).read_text()),
        args.autoencoder,
        args.out,
        args.device,
    )


if __name__ == "__main__":
    main()
