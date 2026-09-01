"""Run the CUR-109 uncapped learned-frontier falsification experiment."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "pure": {
        "structure_temperature_bpp": 0.0,
        "structure_temperature_final_bpp": None,
        "predictive_logit_soft_clip": 0.0,
        "proposal_distillation_weight": 1.0,
    },
    "stabilized": {
        "structure_temperature_bpp": 0.05,
        "structure_temperature_final_bpp": 0.02,
        "predictive_logit_soft_clip": 12.0,
        "proposal_distillation_weight": 1.0,
    },
    "no_distillation": {
        "structure_temperature_bpp": 0.0,
        "structure_temperature_final_bpp": None,
        "predictive_logit_soft_clip": 0.0,
        "proposal_distillation_weight": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[108])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cur109-self-stabilizing-frontier"))
    return parser.parse_args()


def command_for(args: argparse.Namespace, variant: str, seed: int, output: Path) -> list[str]:
    config = VARIANTS[variant]
    command = [
        sys.executable,
        "train_object_permanence_quadtree_v2.py",
        "--updates", str(args.updates),
        "--eval-families", "1",
        "--seed", str(seed),
        "--device", args.device,
        "--objective", "recursive_rgb_innovation_bits",
        "--environment-mode", "causal_pinch_three_step_wild",
        "--tree-allocation-mode", "learned_frontier",
        "--candidate-selection", "learned_frontier",
        # Deliberately omit --candidate-max-nodes: no execution ceiling.
        "--minimum-prediction-depth", "0",
        "--candidate-exploration-paths", "1",
        "--candidate-exploration", "0",
        "--max-observation-frames", "6",
        "--prefetch-workers", "1",
        "--learner-threads", "1",
        "--eval-every", str(args.updates + 1),
        "--checkpoint-every", str(args.checkpoint_every),
        "--skip-final-eval",
        "--report-every", "1",
        "--structure-temperature-bpp", str(config["structure_temperature_bpp"]),
        "--predictive-logit-soft-clip", str(config["predictive_logit_soft_clip"]),
        "--proposal-distillation-weight", str(config["proposal_distillation_weight"]),
        "--output-dir", str(output),
    ]
    final_temperature = config["structure_temperature_final_bpp"]
    if final_temperature is not None:
        command.extend(("--structure-temperature-final-bpp", str(final_temperature)))
    return command


def summarize(output: Path) -> dict:
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    horizons = (1, 4, 8)

    def horizon_mean(row: dict, suffix: str) -> float:
        return statistics.mean(row[f"h{horizon}_{suffix}"] for horizon in horizons)

    return {
        "updates": len(rows),
        "loss_first": rows[0]["total_loss"],
        "loss_last": rows[-1]["total_loss"],
        "candidate_nodes_max": max(row["candidate_nodes"] for row in rows),
        "candidate_nodes_mean": statistics.mean(row["candidate_nodes"] for row in rows),
        "depth8_fraction_max": max(row["candidate_depth8_fraction"] for row in rows),
        "expected_nodes_mean": statistics.mean(horizon_mean(row, "posterior_expected_nodes") for row in rows),
        "posterior_entropy_mean": statistics.mean(horizon_mean(row, "posterior_entropy_bits") for row in rows),
        "posterior_saturation_mean": statistics.mean(horizon_mean(row, "posterior_saturation_fraction") for row in rows),
        "proposal_bits_mean": statistics.mean(horizon_mean(row, "proposal_distillation_bits") for row in rows),
        "gradient_norm_max": max(row["gradient_norm"] for row in rows),
        "updates_per_second": json.loads((output / "profile.json").read_text())["end_to_end_updates_per_second"],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for seed in args.seeds:
        for variant in args.variants:
            output = args.output_dir / f"{variant}-seed-{seed}"
            command = command_for(args, variant, seed, output)
            subprocess.run(command, check=True)
            results[f"{variant}-seed-{seed}"] = summarize(output)
    summary = args.output_dir / "comparison.json"
    summary.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
