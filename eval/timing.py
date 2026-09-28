"""Generation throughput on one GPU: flow integration plus decoding, per 1,000 structures.

    python -m eval.timing --checkpoint flow.pt --out timing.json

Medians over repetitions after a warm-up at each batch size. Timing includes the
device-to-host transfer of the decoded structures, and excludes checkpoint
loading, serialization, and evaluation.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from glass.models import Autoencoder, LatentFlow


@torch.no_grad()
def generate(flow, autoencoder, payload, count, steps, batch_size, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    events = []
    for start in range(0, count, batch_size):
        begin, integrated, decoded = (
            torch.cuda.Event(enable_timing=True) for _ in range(3)
        )
        begin.record()
        z = (
            flow.sample(min(batch_size, count - start), steps, generator)
            * payload["std"]
            + payload["mean"]
        )
        integrated.record()
        coords, logits, cells = autoencoder.decoder(z)
        decoded.record()
        coords.cpu(), logits.argmax(-1).cpu(), cells.cpu()
        events.append((begin, integrated, decoded))

    torch.cuda.synchronize()
    return (
        sum(a.elapsed_time(b) for a, b, _ in events) / 1000,
        sum(b.elapsed_time(c) for _, b, c in events) / 1000,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(256, 512, 1000))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("highest")
    payload = torch.load(args.checkpoint, map_location="cuda")
    flow = LatentFlow(**payload["flow_config"]).cuda().eval()
    flow.load_state_dict(payload["model"])
    autoencoder = Autoencoder(**payload["autoencoder"]["model_config"]).cuda().eval()
    autoencoder.load_state_dict(payload["autoencoder"]["model"])

    results = []
    for batch_size in args.batch_sizes:
        generate(
            flow,
            autoencoder,
            payload,
            min(batch_size, args.count),
            args.steps,
            batch_size,
            0,
        )
        totals, integration, decoding = [], [], []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            flow_seconds, decoder_seconds = generate(
                flow,
                autoencoder,
                payload,
                args.count,
                args.steps,
                batch_size,
                repeat + 1,
            )
            totals.append(time.perf_counter() - started)
            integration.append(flow_seconds)
            decoding.append(decoder_seconds)

        per_1k = 1000 / args.count
        results.append(
            {
                "batch_size": batch_size,
                "median_seconds_per_1k": statistics.median(totals) * per_1k,
                "integration_seconds_per_1k": statistics.median(integration) * per_1k,
                "decoding_seconds_per_1k": statistics.median(decoding) * per_1k,
            }
        )

    report = {
        "checkpoint": args.checkpoint,
        "hardware": torch.cuda.get_device_name(),
        "steps": args.steps,
        "flow_evaluations": 2 * args.steps,
        "results": results,
        "best_seconds_per_1k": min(row["median_seconds_per_1k"] for row in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
