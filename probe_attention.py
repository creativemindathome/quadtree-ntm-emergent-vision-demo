"""Evaluate episode-memory read ablations on a trained quadtree checkpoint."""

from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from ntm.quadtree_world_model_v2 import QuadtreeWorldModelV2, V2MemoryState, V2ModelConfig
from tasks.rgb_quadtree import RGBQuadtreeConfig
from train_object_permanence_quadtree_v2 import (
    encode_observations,
    prediction_loss,
    prepare_example,
    selected_frame_indices,
)


@contextmanager
def episode_read_mode(model: QuadtreeWorldModelV2, mode: str):
    """Temporarily replace only the episode-memory read policy."""
    original = model._read_episode

    if mode == "full":
        yield
        return

    def replacement(self, query_input, episode_payloads, prediction):
        if mode == "no_episode_read":
            return query_input.new_zeros(query_input.shape[0], self.config.payload_dim), None
        if mode != "uniform_slot_read":
            raise ValueError(mode)
        gate_layer = (
            self.prediction_episode_gate if prediction
            else self.observation_episode_gate
        )
        values = self.episode_read_value(episode_payloads)
        context = values.mean(dim=0, keepdim=True).expand(query_input.shape[0], -1)
        gate = torch.sigmoid(gate_layer(query_input))
        weights = query_input.new_full(
            (query_input.shape[0], episode_payloads.shape[0]),
            1.0 / episode_payloads.shape[0],
        )
        return gate * context, weights

    model._read_episode = MethodType(replacement, model)
    try:
        yield
    finally:
        model._read_episode = original


def mean(rows):
    return sum(rows) / max(1, len(rows))


@torch.no_grad()
def run(checkpoint: Path, families: int, seed: int, output: Path) -> dict:
    run_dir = checkpoint.parent
    config = json.loads((run_dir / "config.json").read_text())
    args = Namespace(**config["args"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = QuadtreeWorldModelV2(V2ModelConfig(**config["model_config"])).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    tree_config = RGBQuadtreeConfig(**config["tree_config"])
    rows = [
        prepare_example(20_000 + index, 10_000, seed, args.environment_mode)
        for index in range(families)
    ]
    conditions = ("full", "no_episode_read", "uniform_slot_read", "zero_before_prediction")
    result = {}
    for condition in conditions:
        collected = {}
        losses = []
        mode = condition if condition in ("no_episode_read", "uniform_slot_read") else "full"
        with episode_read_mode(model, mode):
            for prepared in rows:
                observations = prepared["observations"]
                indices = selected_frame_indices(
                    observations.shape[0], args.history_stride, args.max_observation_frames,
                )
                state, _, samples, _, _ = encode_observations(
                    model, observations, tree_config, indices,
                )
                if condition == "zero_before_prediction":
                    state = V2MemoryState(
                        spatial_addresses=state.spatial_addresses,
                        spatial_payloads=state.spatial_payloads,
                        episode_payloads=torch.zeros_like(state.episode_payloads),
                    )
                loss, metrics, _ = prediction_loss(
                    model, state, samples[-1]["heap_indices"], prepared["targets"],
                    args.candidate_max_nodes, args.memory_cost,
                    args.bce_weight, args.iou_weight, args.brier_weight, args.mass_weight,
                    args.pyramid_support_weight, args.target_depth_weight,
                    args.target_depth_goal, args.node_budget, args.budget_penalty,
                    slot_targets=prepared.get("slot_targets"),
                    center_targets=prepared.get("center_targets"),
                    target_visibility=prepared.get("target_visibility"),
                    uncertainty_mask=prepared.get("uncertainty_mask"),
                    candidate_expansion_levels=args.candidate_expansion_levels,
                    union_loss_weight=args.union_loss_weight,
                    slot_loss_weight=args.slot_loss_weight,
                    slot_center_weight=args.slot_center_weight,
                    horizon_weights=(args.h1_weight, args.h4_weight, args.h8_weight),
                )
                losses.append(float(loss.cpu()))
                for name, value in metrics.items():
                    collected.setdefault(name, []).append(float(value))
        result[condition] = {
            "families": families,
            "prediction_loss": mean(losses),
            "h1_hard_iou": mean(collected["h1_hard_iou"]),
            "h4_hard_iou": mean(collected["h4_hard_iou"]),
            "h8_hard_iou": mean(collected["h8_hard_iou"]),
            "h1_expected_nodes": mean(collected["h1_expected_nodes"]),
            "h1_attention_depth_lift": mean(collected["h1_attention_depth_lift"]),
            "h1_mass_ratio": mean(collected["h1_mass_ratio"]),
        }
    baseline = result["full"]
    for values in result.values():
        values["loss_delta_vs_full"] = values["prediction_loss"] - baseline["prediction_loss"]
        values["h1_iou_delta_vs_full"] = values["h1_hard_iou"] - baseline["h1_hard_iou"]
    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "device": str(device),
        "seed": seed,
        "conditions": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--families", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2603)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.families, args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
