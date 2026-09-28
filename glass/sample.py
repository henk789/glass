"""Sample crystals: python -m glass.sample --config ... --checkpoint flow.pt --out samples/"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from pymatgen.io.cif import CifWriter

from glass.data import structure_from_tokens
from glass.models import Autoencoder, LatentFlow
from glass.train_autoencoder import default_device


@torch.no_grad()
def sample(checkpoint, out, count, steps, seed, batch_size, device):
    payload = torch.load(checkpoint, map_location=device)
    flow = LatentFlow(**payload["flow_config"]).to(device).eval()
    flow.load_state_dict(payload["model"])
    ae = Autoencoder(**payload["autoencoder"]["model_config"]).to(device).eval()
    ae.load_state_dict(payload["autoencoder"]["model"])

    generator = torch.Generator(device=device).manual_seed(seed)
    values = {key: [] for key in ("z", "types", "coords", "cells")}

    # Flow integration and decoding both run in FP32.
    for start in range(0, count, batch_size):
        normalized = flow.sample(min(batch_size, count - start), steps, generator)
        z = normalized * payload["std"] + payload["mean"]
        coords, logits, cells = ae.decoder(z)

        values["z"].append(z.cpu())
        values["types"].append(logits.argmax(-1).cpu())
        values["coords"].append(coords.cpu())
        values["cells"].append(cells.cpu())

    samples = {key: torch.cat(chunks) for key, chunks in values.items()}
    samples["ids"] = [f"sample_{i + 1:06d}" for i in range(count)]
    out = Path(out)
    (out / "cifs").mkdir(parents=True)
    torch.save(samples, out / "samples.pt")

    # Degenerate generated lattices cannot be written as CIFs; count them as invalid.
    invalid = 0
    tokens = zip(samples["ids"], samples["types"], samples["coords"], samples["cells"])
    for identifier, types, coords, cell in tokens:
        try:
            structure = structure_from_tokens(types, coords, cell)
        except (ValueError, np.linalg.LinAlgError):
            invalid += 1
            continue
        CifWriter(structure).write_file(out / "cifs" / f"{identifier}.cif")

    print(f"Wrote {count - invalid} CIFs to {out / 'cifs'}; {invalid} invalid lattices")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    sample(args.checkpoint, args.out, **config["sample"], device=args.device)


if __name__ == "__main__":
    main()
