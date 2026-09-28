"""LeMat-GenBench: the full benchmark, a fast single-point variant, or validity + novelty.

    python -m eval.genbench --input samples.pt --out report.json --mode prod

prod  - the benchmark's production recipe, including its internal MLIP relaxation.
fast  - the same metrics on the supplied geometry (single-point energies only).
vn    - validity filtering and novelty only; used for checkpoint sweeps.

Rates in the summary use all requested candidates as the denominator. The upstream
code runs in its own environment (python -m eval.setup genbench).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROD_CONFIG = "comprehensive_multi_mlip_hull"
UMA_MODEL = "uma-s-1p1"
UPSTREAM = Path(__file__).resolve().parents[1] / "cache/upstream/genbench"


def requested_summary(scores, requested, mode):
    """Rebase counts, including the joint SUN/MSUN counts, on all requested candidates."""
    counts = {
        "valid": scores["validity"]["overall_validity_count"],
        "unique": scores["uniqueness"]["unique_structures_count"],
        "novel": scores["novelty"]["novel_structures_count"],
        "stable": scores["stability"]["stable_count"],
        "metastable": scores["stability"]["metastable_count"],
        "SUN": scores["sun"]["sun_count"],
        "MSUN": scores["sun"]["msun_count"],
    }
    summary = {
        "mode": mode,
        "requested": requested,
        "counts": counts,
        "percent_requested": {key: 100 * n / requested for key, n in counts.items()},
        "mean_e_above_hull": scores["stability"]["mean_e_above_hull"],
    }
    if mode == "prod":
        summary["mean_relaxation_RMSE"] = scores["stability"]["mean_relaxation_RMSE"]
    return summary


def evaluate(source, out, expected_count, mode, workers=1, device="cuda"):
    from eval.common import export  # GLASS environment only; not in the upstream one

    source, out = Path(source).resolve(), Path(out).resolve()
    work = out.with_suffix("")
    export(source, work / "cifs")

    records = json.loads((work / "cifs/manifest.json").read_text())["records"]
    assert len(records) == expected_count, (
        f"Expected {expected_count}, got {len(records)}"
    )

    environment = dict(os.environ)
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        str(Path(__file__).resolve()),
        "--upstream",
        "--mode",
        mode,
        "--cifs",
        str(work / "cifs"),
        "--requested",
        str(expected_count),
        "--workers",
        str(workers),
        "--device",
        device,
        "--result",
        str(out),
        "--name",
        str(work).replace("/", "_"),
    ]
    subprocess.run(command, cwd=UPSTREAM, env=environment, check=True)

    if mode != "vn":
        scores = json.loads(out.with_suffix(".scores.json").read_text())
        summary = requested_summary(scores, expected_count, mode)
        out.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    return out


def run_upstream(options):
    """Runs inside the pinned LeMat-GenBench checkout and environment."""
    from pymatgen.core import Structure

    sys.path.insert(0, str(Path.cwd() / "scripts"))
    import run_benchmarks as upstream

    config = upstream.load_benchmark_config(PROD_CONFIG)
    config["fingerprint_method"] = "structure-matcher"
    config["novelty_settings"]["n_jobs"] = options.workers
    config["preprocessor_config"]["relax_structures"] = options.mode == "prod"
    mlips = config["preprocessor_config"]["mlip_configs"]
    mlips["uma"]["model_name"] = UMA_MODEL
    for settings in mlips.values():
        settings["device"] = options.device

    # Embeddings are kept in memory; skip upstream's extra dump into its checkout.
    upstream.save_embeddings_from_structures = lambda *args, **kwargs: None

    # Upstream hard-codes relaxation in this constructor; fast mode disables only it.
    if options.mode == "fast":
        preprocessor = upstream.MultiMLIPStabilityPreprocessor

        def single_point(**settings):
            return preprocessor(**{**settings, "relax_structures": False})

        upstream.MultiMLIPStabilityPreprocessor = single_point

    structures, load_errors = [], 0
    for cif in upstream.load_cif_files(str(options.cifs)):
        try:
            structures.append(Structure.from_file(cif))
        except Exception as error:  # noqa: BLE001 - unreadable CIFs stay in the denominator
            load_errors += 1
            upstream.logger.warning(f"Failed to load {cif}: {error}")

    validity, valid_structures, filtering = (
        upstream.run_validity_preprocessing_and_filtering(structures, config)
    )

    if options.mode == "vn":
        novelty = upstream.run_remaining_benchmarks(
            valid_structures, ["novelty"], config
        )
        valid = validity.final_scores["overall_validity_count"]
        valid_novel = novelty["novelty"].final_scores["novel_structures_count"]
        report = {
            "requested": options.requested,
            "loaded": len(structures),
            "load_errors": load_errors,
            "valid_count": valid,
            "valid_novel_count": valid_novel,
            "valid_rate": valid / options.requested,
            "valid_novel_rate": valid_novel / options.requested,
            "validity_filtering": filtering,
            "novelty": novelty["novelty"].final_scores,
        }
        options.result.write_text(
            json.dumps(report, indent=2, default=lambda v: v.item()) + "\n"
        )
        return

    families = [
        "distribution",
        "diversity",
        "novelty",
        "uniqueness",
        "hhi",
        "sun",
        "stability",
    ]
    preprocessors = upstream.create_preprocessor_config(families, "structure-matcher")
    preprocessors["validity"] = False
    processed, _ = upstream.run_remaining_preprocessors(
        valid_structures, preprocessors, config, options.name
    )
    remaining = upstream.run_remaining_benchmarks(processed, families, config)

    scores = {
        name: result.final_scores
        for name, result in {"validity": validity, **remaining}.items()
    }
    scores_json = (
        json.dumps(scores, indent=2, default=lambda value: value.item()) + "\n"
    )
    options.result.with_suffix(".scores.json").write_text(scores_json)
    options.result.with_suffix(".config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    if options.mode == "prod":
        native = upstream.save_results(
            validity, remaining, filtering, options.name, PROD_CONFIG, len(structures)
        )
        shutil.copyfile(native, options.result)
    else:
        options.result.write_text(scores_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", dest="source")
    parser.add_argument("--out")
    parser.add_argument("--mode", choices=("prod", "fast", "vn"), required=True)
    parser.add_argument("--expected-count", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=1, help="CPU novelty workers")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    # Internal arguments of the upstream stage.
    parser.add_argument("--upstream", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cifs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--requested", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--name", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.upstream:
        run_upstream(args)
    else:
        print(
            evaluate(
                args.source,
                args.out,
                args.expected_count,
                args.mode,
                args.workers,
                args.device,
            )
        )
