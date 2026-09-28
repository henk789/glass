"""Install the pinned external evaluators, each in its own environment, under cache/.

    python -m eval.setup <genbench | crystalite | mofchecker | mofid | esen>

The upstream protocols need mutually incompatible dependency versions (for example
different pymatgen releases), so none of them shares GLASS's environment.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / "cache"
REPOSITORIES = {
    "crystalite": (
        "https://github.com/joshrosie/crystalite",
        "3b2d4eacf3f0b17a04851b7bed1fcedc9733cda9",
    ),
    "genbench": (
        "https://github.com/LeMaterial/LeMat-GenBench",
        "58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6",
    ),
    "mofid": (
        "https://github.com/snurr-group/mofid",
        "36873683083c7bc62c1da2062df2833adc35b48a",
    ),
}
CRYSTALITE_DATA = (
    "joshrosie/crystalite-datasets",
    "3e88267c9b07a23a2b069bd94f29a351be858c21",
)
PHASE_DIAGRAM = (
    "https://ndownloader.figshare.com/files/48241624",
    "60d19d691fa1d338aa496a40a9641bef",
)
ESEN_MODEL = (
    "facebook/OMAT24",
    "esen_30m_oam.pt",
    "8a5a78c7ba7b250a17e85fe85943c4608499d895",
)


def run(*command, **options):
    subprocess.run([str(part) for part in command], check=True, **options)


def upstream(name):
    """Check out an upstream repository at its pinned revision."""
    url, revision = REPOSITORIES[name]
    root = CACHE / "upstream" / name
    if not root.exists():
        run("git", "clone", "--no-checkout", url, root)
    run("git", "checkout", "--detach", revision, cwd=root)
    return root


def setup_genbench():
    root = upstream("genbench")
    # Upstream passes the model name both positionally and as a keyword.
    patch = Path(__file__).with_name("genbench_registry.patch")
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if applied.returncode:
        run("git", "apply", patch, cwd=root)
    run("uv", "sync", "--locked", cwd=root)


def setup_crystalite():
    root = upstream("crystalite")
    run("uv", "sync", "--locked", cwd=root)
    assets = CACHE / "crystalite"

    # Crystalite's MP20 CSVs: training structures for novelty, validation for distances.
    from huggingface_hub import hf_hub_download

    repo, revision = CRYSTALITE_DATA
    for split in ("train", "val", "test"):
        hf_hub_download(
            repo,
            f"mp20/raw/{split}.csv",
            repo_type="dataset",
            revision=revision,
            local_dir=assets,
        )

    # Materials Project phase diagram for energies above hull.
    url, md5 = PHASE_DIAGRAM
    archive = assets / "thermo/2023-02-07-ppd-mp.pkl.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as source, archive.open("wb") as target:
            shutil.copyfileobj(source, target)
    assert hashlib.md5(archive.read_bytes()).hexdigest() == md5, (
        "Phase diagram checksum"
    )
    run("gzip", "-dkf", archive)

    # NequIP-OAM-L, compiled for the GPU that will run the evaluation.
    model = assets / "nequip/NequIP-OAM-L-ase.nequip.pt2"
    model.parent.mkdir(parents=True, exist_ok=True)
    run(
        root / ".venv/bin/nequip-compile",
        "nequip.net:mir-group/NequIP-OAM-L:0.1",
        model,
        "--mode",
        "aotinductor",
        "--device",
        "cuda",
        "--target",
        "ase",
    )


def setup_mofchecker():
    root = CACHE / "environments/mofchecker"
    python = root / "bin/python"
    if not python.exists():
        run("uv", "venv", "--python", "3.11", root)
    install = ["uv", "pip", "install", "--python", python]

    # pyeqeq still imports pkg_resources, which setuptools 81 removed.
    run(*install, "setuptools==80.9.0", "wheel", "pybind11>=2.12")
    run(
        *install,
        "--no-build-isolation-package",
        "pyeqeq",
        "mofchecker==0.9.6",
        "pymatgen==2024.6.10",
        "monty==2024.7.30",  # The older CifWriter relies on implicit text mode.
        "ase==3.26.0",
        "spglib==2.7.0",
        "numpy==1.26.4",
        "pyyaml",
        "torch==2.13.0",
    )
    run(*install, "--no-deps", "-e", PROJECT)


def setup_mofid():
    root = upstream("mofid")
    if not (root / "bin/sbu").exists():
        # MOFid bundles an Open Babel whose CMake files CMake 4 no longer accepts.
        tools = CACHE / "environments/cmake"
        if not tools.exists():
            run("uv", "venv", tools)
        run("uv", "pip", "install", "--python", tools / "bin/python", "cmake>=3.20,<4")
        environment = {
            **os.environ,
            "PATH": f"{tools / 'bin'}{os.pathsep}{os.environ['PATH']}",
        }
        run("make", "init", cwd=root, env=environment)
    run("python", "set_paths.py", cwd=root)


def setup_esen():
    root = CACHE / "environments/esen"
    python = root / "bin/python"
    if not python.exists():
        run("uv", "venv", "--python", "3.12", root)
    run(
        "uv",
        "pip",
        "sync",
        "--python",
        python,
        Path(__file__).with_name("esen-oam-d3.txt"),
    )
    run("uv", "pip", "install", "--python", python, "--no-deps", "-e", PROJECT)

    # The OMAT24 checkpoints are gated: accept their license on Hugging Face first.
    from huggingface_hub import hf_hub_download

    repo, filename, revision = ESEN_MODEL
    hf_hub_download(repo, filename, revision=revision, local_dir=CACHE / "models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "evaluator", choices=("genbench", "crystalite", "mofchecker", "mofid", "esen")
    )
    globals()["setup_" + parser.parse_args().evaluator]()
