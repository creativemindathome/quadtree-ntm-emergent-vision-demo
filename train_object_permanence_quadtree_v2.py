"""Train the unchanged QuadtreeWorldModelV2 on the RGB permanence arena.

Capacity is increased exclusively through ``V2ModelConfig``.  RGB adaptation,
tree budgeting, curriculum, prefetch, logging, and ablations live outside the
model so its exact-pointer memory and recurrent semantics remain unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ntm.quadtree_memory import address_to_bounds
from ntm.quadtree_world_model_v2 import (
    QuadtreeWorldModelV2,
    V2MemoryState,
    V2ModelConfig,
    balanced_pixel_bce,
    depth_class_balanced_split_bce,
    soft_hierarchical_rasterize,
)
from tasks.object_permanence_arena import (
    ObjectPermanenceConfig,
    build_object_permanence_training_examples,
    generate_object_permanence_family,
)
from tasks.rgb_quadtree import RGBQuadtreeConfig, build_rgb_quadtree_sample
from tasks.visual_domains import VisualDomainConfig


MODEL_CONFIG = V2ModelConfig(
    canvas_size=256,
    image_size=192,
    max_depth=8,
    sensor_features=13,
    payload_dim=96,
    episode_slots=16,
    path_dim=48,
    hidden_dim=192,
    prediction_slots=4,
    episode_attention=True,
    controller_norm=True,
    prediction_split_bias=1.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--eval-families", type=int, default=4)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--history-stride", type=int, default=4)
    parser.add_argument("--max-observation-frames", type=int, default=12)
    parser.add_argument("--tree-active-depth", type=int, default=4)
    parser.add_argument("--tree-max-nodes", type=int, default=None,
                        help="legacy adapter cap; unused by learned_frontier")
    parser.add_argument("--frontier-split-threshold", type=float, default=0.5)
    parser.add_argument(
        "--frontier-exploration", type=float, default=0.0,
        help="training-only probability of exploring a below-threshold frontier node",
    )
    parser.add_argument(
        "--tree-allocation-mode",
        choices=("complete", "variance_budgeted", "learned_frontier"),
        default="learned_frontier",
        help="learned_frontier lets the model request regions directly from full RGB",
    )
    parser.add_argument(
        "--candidate-max-nodes", type=int, default=None,
        help=(
            "optional emergency execution ceiling; omitted means recursive "
            "parent SPLIT decisions alone determine predictive support"
        ),
    )
    parser.add_argument(
        "--minimum-prediction-depth", type=int, default=0,
        help="structural global floor; depth 1 means four quadrant blobs",
    )
    parser.add_argument("--candidate-expansion-levels", type=int, default=1)
    parser.add_argument(
        "--candidate-selection", choices=("uniform", "learned_frontier"),
        default="uniform",
        help="learned_frontier recursively proposes target-independent children from split logits",
    )
    parser.add_argument("--candidate-split-threshold", type=float, default=0.5)
    parser.add_argument(
        "--candidate-exploration", type=float, default=0.0,
        help="training-only probability of expanding a below-threshold predictive leaf",
    )
    parser.add_argument(
        "--candidate-exploration-paths", type=int, default=0,
        help="complete-sibling random paths that expose depth-8 compression gradients",
    )
    parser.add_argument("--observation-loss-weight", type=float, default=0.05)
    parser.add_argument("--split-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-cost", type=float, default=2.5e-5)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--iou-weight", type=float, default=0.3)
    parser.add_argument("--brier-weight", type=float, default=0.2)
    parser.add_argument("--mass-weight", type=float, default=0.05)
    parser.add_argument("--pyramid-support-weight", type=float, default=0.0)
    parser.add_argument("--target-depth-weight", type=float, default=0.0)
    parser.add_argument("--target-depth-goal", type=float, default=5.0)
    parser.add_argument("--allocation-warmup-fraction", type=float, default=0.15)
    parser.add_argument("--allocation-transition-fraction", type=float, default=0.35)
    parser.add_argument("--target-depth-final-scale", type=float, default=0.15)
    parser.add_argument("--union-loss-weight", type=float, default=1.0)
    parser.add_argument("--slot-loss-weight", type=float, default=0.0)
    parser.add_argument("--slot-center-weight", type=float, default=0.0)
    parser.add_argument("--slot-diversity-weight", type=float, default=0.0)
    parser.add_argument("--h1-weight", type=float, default=1.0)
    parser.add_argument("--h4-weight", type=float, default=1.0)
    parser.add_argument("--h8-weight", type=float, default=1.0)
    parser.add_argument(
        "--horizon-curriculum", action="store_true",
        help="shift objective mass from short to long prediction across north-star stages",
    )
    parser.add_argument("--node-budget", type=float, default=64.0)
    parser.add_argument("--budget-penalty", type=float, default=0.02)
    parser.add_argument(
        "--objective", choices=(
            "legacy", "predictive_bits", "recursive_rgb_bits",
            "recursive_rgb_innovation_bits",
        ),
        default="predictive_bits",
    )
    parser.add_argument("--address-budget-bits", type=float, default=256.0)
    parser.add_argument(
        "--structure-temperature-bpp", type=float, default=0.02,
        help="softmin temperature in local bits/bit for innovation-tree attention",
    )
    parser.add_argument(
        "--structure-temperature-final-bpp", type=float, default=None,
        help="optional final local-bpp temperature, linearly annealed over training",
    )
    parser.add_argument(
        "--predictive-logit-soft-clip", type=float, default=12.0,
        help="smooth loss-only bound for RGB bit logits; <=0 disables it",
    )
    parser.add_argument(
        "--proposal-distillation-weight", type=float, default=1.0,
        help="weight on posterior-matched traversal cross-entropy in bits per reached decision",
    )
    parser.add_argument("--dual-learning-rate", type=float, default=1e-3)
    parser.add_argument("--dual-price-initial", type=float, default=0.0)
    parser.add_argument("--dual-price-max", type=float, default=100.0)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--learner-threads", type=int, default=1)
    parser.add_argument("--examples-per-family", type=int, default=4)
    parser.add_argument(
        "--environment-mode",
        choices=(
            "staged", "mvp", "hard_mvp", "north_star",
            "dense_causal_relay", "dense_causal_relay_ood", "causal_pinch",
            "causal_pinch_three_step", "causal_pinch_three_step_diverse",
            "causal_pinch_three_step_wild",
        ),
        default="staged",
    )
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument(
        "--skip-final-eval", action="store_true",
        help="save the final checkpoint without the seven-condition evaluation",
    )
    parser.add_argument("--periodic-eval-families", type=int, default=2)
    parser.add_argument("--output-dir", default="runs/object-permanence-quadtree-v3")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        # This exact-pointer implementation is currently scalar/Python heavy;
        # local profiling shows CPU is dramatically faster than Apple MPS.
        return torch.device("cpu")
    return torch.device(requested)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def scheduled_learning_rate(update: int, args: argparse.Namespace) -> float:
    """Linear warmup followed by cosine decay, indexed before each update."""
    if args.warmup_updates > 0 and update < args.warmup_updates:
        return args.learning_rate * float(update + 1) / args.warmup_updates
    decay_updates = max(1, args.updates - args.warmup_updates)
    progress = min(1.0, max(0.0, (update - args.warmup_updates) / decay_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (
        args.learning_rate - args.min_learning_rate
    ) * cosine


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _compact_training_sample(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Retain only tensors consumed by the learner across process transfer."""
    return {
        key: sample[key]
        for key in ("memory", "heap_indices", "split_targets", "depths", "image")
    }


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def curriculum_config(
    update: int, updates: int, seed: int, environment_mode: str = "staged",
) -> Tuple[str, ObjectPermanenceConfig]:
    fraction = update / max(1, updates)
    common = dict(
        image_size=192,
        episode_length=56,
        object_radius=4.0,
        speed_range=(1.6, 2.2),
        occluder_width=18,
        target_horizons=(1, 4, 8),
        query_start=16,
        query_stride=8,
        behavior_branch_count=3,
        seed=seed,
    )
    if environment_mode == "mvp":
        return "mvp_occlusion", ObjectPermanenceConfig(
            tracked_count=2, distractor_count=2, occluder_count=1,
            ambiguous_tracked_appearance=True, hidden_behavior_enabled=False, **common,
        )
    if environment_mode == "hard_mvp":
        hard = dict(common)
        hard.update(
            episode_length=72,
            object_radius=3.0,
            speed_range=(1.9, 2.8),
            occluder_width=22,
            occluder_height_fraction=0.82,
            query_start=20,
            query_stride=10,
            behavior_branch_count=5,
            target_cue_frames=1,
        )
        return "hard_mvp_hidden_behavior", ObjectPermanenceConfig(
            tracked_count=4, distractor_count=6, occluder_count=2,
            ambiguous_tracked_appearance=True, hidden_behavior_enabled=True, **hard,
        )
    if environment_mode == "north_star":
        if fraction < 0.10:
            return "identity_warmup", ObjectPermanenceConfig(
                image_size=192, episode_length=56,
                tracked_count=2, distractor_count=1, occluder_count=1,
                object_radius=5.0, speed_range=(1.6, 2.2),
                occluder_width=12, occluder_height_fraction=0.35,
                target_horizons=(1, 4, 8), query_start=12, query_stride=4,
                target_cue_frames=2, behavior_branch_count=3,
                ambiguous_tracked_appearance=True, hidden_behavior_enabled=False,
                seed=seed,
            )
        if fraction < 0.30:
            return "crossing_occlusion", ObjectPermanenceConfig(
                image_size=192, episode_length=64,
                tracked_count=2, distractor_count=3, occluder_count=1,
                object_radius=4.0, speed_range=(1.8, 2.4),
                occluder_width=18, occluder_height_fraction=0.72,
                target_horizons=(1, 4, 8), query_start=12, query_stride=4,
                target_cue_frames=2, behavior_branch_count=3,
                ambiguous_tracked_appearance=True, hidden_behavior_enabled=False,
                seed=seed,
            )
        if fraction < 0.55:
            return "temporal_crossing", ObjectPermanenceConfig(
                image_size=192, episode_length=72,
                tracked_count=4, distractor_count=4, occluder_count=2,
                object_radius=4.0, speed_range=(1.8, 2.6),
                occluder_width=18, occluder_height_fraction=0.72,
                target_horizons=(1, 4, 8), query_start=16, query_stride=4,
                target_cue_frames=2, behavior_branch_count=3,
                ambiguous_tracked_appearance=True, hidden_behavior_enabled=False,
                seed=seed,
            )
        if fraction < 0.80:
            return "hidden_behavior", ObjectPermanenceConfig(
                image_size=192, episode_length=80,
                tracked_count=4, distractor_count=6, occluder_count=2,
                object_radius=3.5, speed_range=(1.9, 2.8),
                occluder_width=22, occluder_height_fraction=0.82,
                target_horizons=(1, 4, 8), query_start=20, query_stride=4,
                query_end=44, target_cue_frames=1, behavior_branch_count=6,
                ambiguous_tracked_appearance=True, hidden_behavior_enabled=True,
                seed=seed,
            )
        return "hard_mixture", ObjectPermanenceConfig(
            image_size=192, episode_length=96,
            tracked_count=4, distractor_count=6, occluder_count=2,
            object_radius=3.0, speed_range=(1.9, 2.8),
            occluder_width=22, occluder_height_fraction=0.82,
            target_horizons=(1, 4, 8), query_start=20, query_stride=4,
            query_end=64, target_cue_frames=1, behavior_branch_count=6,
            ambiguous_tracked_appearance=True, hidden_behavior_enabled=True,
            seed=seed,
        )
    if environment_mode in ("dense_causal_relay", "dense_causal_relay_ood"):
        if environment_mode == "dense_causal_relay_ood":
            chain_length, distractors, episode_length = 12, 72, 144
            stage = "dense_relay_ood_l12"
        elif fraction < 0.34:
            chain_length, distractors, episode_length = 4, 24, 80
            stage = "dense_relay_l4"
        elif fraction < 0.67:
            chain_length, distractors, episode_length = 6, 36, 96
            stage = "dense_relay_l6"
        else:
            chain_length, distractors, episode_length = 8, 48, 112
            stage = "dense_relay_l8"
        return stage, ObjectPermanenceConfig(
            image_size=192,
            episode_length=episode_length,
            tracked_count=2,
            distractor_count=distractors,
            occluder_count=2,
            object_radius=3.0,
            speed_range=(1.5, 2.5),
            occluder_width=16,
            occluder_height_fraction=0.66,
            target_horizons=(1, 4, 8),
            query_start=16,
            query_stride=6,
            query_end=episode_length - 12,
            target_cue_frames=2,
            behavior_branch_count=2,
            ambiguous_tracked_appearance=True,
            hidden_behavior_enabled=False,
            motion_layout="dense_relay",
            relay_chain_length=chain_length,
            moving_shape_count=4,
            seed=seed,
        )
    if environment_mode in (
        "causal_pinch", "causal_pinch_three_step", "causal_pinch_three_step_diverse",
        "causal_pinch_three_step_wild",
    ):
        three_step = environment_mode != "causal_pinch"
        diverse_continuation = environment_mode == "causal_pinch_three_step_diverse"
        wild = environment_mode == "causal_pinch_three_step_wild"
        if wild:
            # Coprime cycles prevent bundled updates from aliasing the same
            # environment configuration. These are environment parameters,
            # not labels exposed to the model.
            phase = lambda modulus, multiplier: ((update * multiplier + seed) % modulus) / (modulus - 1)
            density = phase(17, 11)
            motion = phase(19, 13)
            geometry = phase(23, 17)
            timing = phase(29, 19)
            material = phase(31, 23)
            stage = "pinch3_wild"
            distractors = 5 + round(7 * density)
            shape_count = 4
            angle_bins = 7 + round(13 * geometry)
            same_shape = min(distractors - 2, 2 + round((distractors - 4) * density))
            offset_scale = 0.35 + 1.25 * geometry
            size_range = (0.55 + 0.20 * geometry, 1.35 + 0.30 * geometry)
            speed_range = (0.65 + 0.75 * motion, 2.6 + 1.8 * motion)
            occluders = 1 + round(2 * density)
            object_radius = 2.5 + 3.5 * geometry
            occluder_width = 8 + round(22 * phase(37, 29))
            occluder_height = 0.25 + 0.50 * phase(41, 31)
            contact_frame = 22 + round(12 * timing)
            close_frames = 3 + round(9 * phase(43, 37))
            transport_speed = 0.6 + 2.9 * motion
            friction_levels = (
                0.08 + 0.32 * material,
                0.58 + 0.40 * phase(47, 41),
            )
            episode_length, query_end = 96, 76
        elif diverse_continuation:
            # Continue from a model trained on pinch3_outcome without abruptly
            # destroying its learned allocation policy. Each rung adds causal
            # aliases before the final maximum-diversity distribution.
            if fraction < 1.0 / 3.0:
                stage = "pinch3_diverse_bridge"
                distractors, shape_count, angle_bins = 6, 4, 8
                same_shape, offset_scale, size_range = 4, 0.8, (0.8, 1.2)
                speed_range, occluders = (1.3, 2.3), 1
            elif fraction < 2.0 / 3.0:
                stage = "pinch3_diverse_expand"
                distractors, shape_count, angle_bins = 8, 4, 10
                same_shape, offset_scale, size_range = 6, 0.95, (0.75, 1.3)
                speed_range, occluders = (1.25, 2.45), 2
            else:
                stage = "pinch3_diverse_max"
                distractors, shape_count, angle_bins = 10, 4, 12
                same_shape, offset_scale, size_range = 8, 1.1, (0.7, 1.35)
                speed_range, occluders = (1.2, 2.6), 2
        elif three_step:
            if fraction < 1.0 / 3.0:
                stage = "pinch3_identify"
                distractors, shape_count, angle_bins = 3, 2, 2
                same_shape, offset_scale, size_range = 1, 0.0, (1.0, 1.0)
                speed_range, occluders = (1.3, 1.7), 1
            elif fraction < 2.0 / 3.0:
                stage = "pinch3_contact"
                distractors, shape_count, angle_bins = 4, 4, 4
                same_shape, offset_scale, size_range = 2, 0.0, (0.9, 1.1)
                speed_range, occluders = (1.3, 2.0), 1
            else:
                stage = "pinch3_outcome"
                distractors, shape_count, angle_bins = 6, 4, 8
                same_shape, offset_scale, size_range = 4, 0.8, (0.8, 1.2)
                speed_range, occluders = (1.3, 2.3), 1
        elif fraction < 0.15:
            stage = "pinch_p0_geometry"
            distractors, shape_count, angle_bins = 3, 2, 2
            same_shape, offset_scale, size_range = 1, 0.0, (1.0, 1.0)
            speed_range, occluders = (1.3, 1.7), 1
        elif fraction < 0.40:
            stage = "pinch_p1_material"
            distractors, shape_count, angle_bins = 4, 4, 4
            same_shape, offset_scale, size_range = 2, 0.0, (0.9, 1.1)
            speed_range, occluders = (1.3, 2.0), 1
        elif fraction < 0.70:
            stage = "pinch_p2_torque"
            distractors, shape_count, angle_bins = 6, 4, 8
            same_shape, offset_scale, size_range = 4, 0.8, (0.8, 1.2)
            speed_range, occluders = (1.3, 2.3), 1
        else:
            stage = "pinch_p3_dense_causal"
            distractors, shape_count, angle_bins = 10, 4, 12
            same_shape, offset_scale, size_range = 8, 1.1, (0.7, 1.35)
            speed_range, occluders = (1.2, 2.6), 2
        if not wild:
            object_radius = 4.0
            occluder_width = 14
            occluder_height = 0.35
            contact_frame = 28
            close_frames = 8
            transport_speed = 1.5
            friction_levels = (0.25, 0.85)
            episode_length, query_end = 80, 60
        return stage, ObjectPermanenceConfig(
            image_size=192,
            episode_length=episode_length,
            tracked_count=1,
            distractor_count=distractors,
            occluder_count=occluders,
            object_radius=object_radius,
            speed_range=speed_range,
            occluder_width=occluder_width,
            occluder_height_fraction=occluder_height,
            target_horizons=(1, 4, 8),
            query_start=16,
            query_stride=4,
            query_end=query_end,
            target_cue_frames=2,
            behavior_branch_count=2,
            ambiguous_tracked_appearance=False,
            hidden_behavior_enabled=False,
            motion_layout="causal_pinch",
            moving_shape_count=4,
            pinch_contact_frame=contact_frame,
            pinch_close_frames=close_frames,
            pinch_transport_speed=transport_speed,
            pinch_friction_levels=friction_levels,
            pinch_angle_bins=angle_bins,
            pinch_shape_count=shape_count,
            pinch_same_shape_distractors=same_shape,
            pinch_offset_scale=offset_scale,
            pinch_size_range=size_range,
            seed=seed,
        )
    if fraction < 0.2:
        return "motion", ObjectPermanenceConfig(
            tracked_count=2, distractor_count=2, occluder_count=1,
            ambiguous_tracked_appearance=False, hidden_behavior_enabled=False, **common,
        )
    if fraction < 0.5:
        return "occlusion", ObjectPermanenceConfig(
            tracked_count=4, distractor_count=4, occluder_count=2,
            ambiguous_tracked_appearance=True, hidden_behavior_enabled=False, **common,
        )
    return "hidden_behavior", ObjectPermanenceConfig(
        tracked_count=4, distractor_count=6, occluder_count=2,
        ambiguous_tracked_appearance=True, hidden_behavior_enabled=True, **common,
    )


def _aggregate_slots(slot_probabilities: torch.Tensor) -> torch.Tensor:
    """Union independent per-slot occupancy into the model's one raster head."""
    return 1.0 - torch.prod(1.0 - slot_probabilities.clamp(0.0, 1.0), dim=1)


def horizon_weights_for_stage(
    stage: str, args: argparse.Namespace,
) -> Tuple[float, float, float]:
    """Curriculum over objectives, not topology or split decisions."""
    if not args.horizon_curriculum:
        return args.h1_weight, args.h4_weight, args.h8_weight
    schedule = {
        "identity_warmup": (1.0, 0.35, 0.10),
        "crossing_occlusion": (1.0, 0.55, 0.25),
        "temporal_crossing": (0.80, 0.80, 0.50),
        "hidden_behavior": (0.50, 0.80, 0.80),
        "hard_mixture": (0.35, 0.75, 1.00),
        "pinch_p0_geometry": (1.00, 0.45, 0.20),
        "pinch_p1_material": (0.90, 0.70, 0.40),
        "pinch_p2_torque": (0.65, 0.85, 0.75),
        "pinch_p3_dense_causal": (0.40, 0.80, 1.00),
        "pinch3_identify": (1.00, 0.35, 0.10),
        "pinch3_contact": (0.85, 0.70, 0.35),
        "pinch3_outcome": (0.55, 0.85, 0.85),
        "pinch3_diverse_bridge": (0.45, 0.80, 1.00),
        "pinch3_diverse_expand": (0.45, 0.80, 1.00),
        "pinch3_diverse_max": (0.45, 0.80, 1.00),
    }
    return schedule.get(stage, (args.h1_weight, args.h4_weight, args.h8_weight))


def allocation_objective_scales(
    fraction: float, args: argparse.Namespace,
) -> Tuple[float, float]:
    """Learn spatial support first, then compress it without hard tree targets."""
    start = args.allocation_warmup_fraction
    end = args.allocation_transition_fraction
    if fraction <= start:
        return 1.0, 0.0
    if fraction >= end:
        return args.target_depth_final_scale, 1.0
    progress = (fraction - start) / max(end - start, 1e-8)
    smooth = 0.5 - 0.5 * math.cos(math.pi * progress)
    target_scale = 1.0 + smooth * (args.target_depth_final_scale - 1.0)
    return target_scale, smooth


def _causal_metadata(family, arena_config: ObjectPermanenceConfig) -> Dict[str, object]:
    physical = family.physical_episode
    if not physical.contact_events or physical.grasp_program is None:
        return {}
    event = physical.contact_events[0]
    return {
        "causal_outcome": event.outcome,
        "causal_friction": event.friction,
        "causal_contact_offset": event.center_of_mass_offset,
        "causal_approach_angle": physical.grasp_program.approach_angle,
        "causal_target_shape": physical.manifest.object_specs[event.target_object_index].shape,
        "causal_same_shape_distractors": arena_config.pinch_same_shape_distractors,
    }


def prepare_example(
    update: int, updates: int, seed: int, environment_mode: str = "staged",
) -> Dict[str, object]:
    start = time.perf_counter()
    stage, arena_config = curriculum_config(update, updates, seed, environment_mode)
    family = generate_object_permanence_family(
        arena_config,
        VisualDomainConfig(seed=seed + 17),
        family_index=update,
        appearance_indices=(update % 3,),
    )
    causal_metadata = _causal_metadata(family, arena_config)
    examples = build_object_permanence_training_examples(family)
    example = examples[update % len(examples)]
    return {
        "stage": stage,
        "arena_config": asdict(arena_config),
        "observations": example.observations,
        "future_rgb": example.future_rgb,
        "targets": _aggregate_slots(example.target_occupancy),
        "slot_targets": example.target_occupancy,
        "center_targets": example.target_center_density,
        "target_visibility": example.evaluator_target_visibility,
        "uncertainty_mask": example.evaluator_uncertainty_mask,
        "uncertainty_rate": float(example.evaluator_uncertainty_mask.float().mean()),
        "family_id": example.family_id,
        "query_frame": example.query_frame,
        "curriculum_fraction": min(1.0, max(0.0, update / max(updates, 1))),
        "data_seconds": time.perf_counter() - start,
        **causal_metadata,
    }


def prepare_sparse_example(
    update: int,
    updates: int,
    seed: int,
    tree_config_payload: Dict[str, object],
    history_stride: int,
    max_observation_frames: int,
    environment_mode: str = "staged",
) -> Dict[str, object]:
    """Worker-side raw generation followed by compact sparse conversion.

    Only sparse rows, targets, and one small query preview cross the process
    boundary.  The full raw episode is created and discarded inside the worker.
    """
    return prepare_sparse_bundle(
        update, updates, seed, tree_config_payload,
        history_stride, max_observation_frames, 1, environment_mode,
    )[0]


def prepare_sparse_bundle(
    start_update: int,
    updates: int,
    seed: int,
    tree_config_payload: Dict[str, object],
    history_stride: int,
    max_observation_frames: int,
    examples_per_family: int,
    environment_mode: str = "staged",
) -> list[Dict[str, object]]:
    """Generate one raw family and use several distinct causal query prefixes."""
    worker_start = time.perf_counter()
    torch.set_num_threads(1)
    stage, arena_config = curriculum_config(
        start_update, updates, seed, environment_mode,
    )
    tick = time.perf_counter()
    # ``start_update`` advances by ``examples_per_family`` in the producer.
    # Using it directly as the family index aliases every modulo-controlled
    # physical factor whenever the bundle width shares that modulus (the
    # default width of four previously produced only low-friction slips).
    physical_family_index = start_update // examples_per_family
    family = generate_object_permanence_family(
        arena_config,
        VisualDomainConfig(seed=seed + 17),
        family_index=physical_family_index,
        appearance_indices=(start_update % 3,),
    )
    causal_metadata = _causal_metadata(family, arena_config)
    examples = build_object_permanence_training_examples(family)
    if environment_mode in (
        "causal_pinch_three_step", "causal_pinch_three_step_diverse",
        "causal_pinch_three_step_wild",
    ):
        close_frame = arena_config.pinch_contact_frame + arena_config.pinch_close_frames
        examples = [example for example in examples if example.query_frame >= close_frame]
    raw_environment_family_seconds = time.perf_counter() - tick
    count = min(examples_per_family, len(examples), updates - start_update)
    chosen = [examples[(start_update + offset) % len(examples)] for offset in range(count)]
    tree_config = RGBQuadtreeConfig(**tree_config_payload)
    prepared_rows = []
    sample_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    for example_offset, example in enumerate(chosen):
        observations = example.observations
        if environment_mode in (
            "causal_pinch_three_step", "causal_pinch_three_step_diverse",
            "causal_pinch_three_step_wild",
        ):
            indices = (
                min(1, observations.shape[0] - 1),
                min(arena_config.pinch_contact_frame, observations.shape[0] - 1),
                observations.shape[0] - 1,
            )
        else:
            indices = selected_frame_indices(
                observations.shape[0], history_stride, max_observation_frames,
            )
        tick = time.perf_counter()
        if tree_config.allocation_mode == "learned_frontier":
            # Allocation depends on the current model, so workers transfer only
            # the selected raw frames. No image-statistics heuristic chooses a
            # topology here.
            samples = []
            selected_observations = observations[list(indices)].contiguous()
        else:
            samples = []
            for index in indices:
                if index not in sample_cache:
                    sample_cache[index] = _compact_training_sample(
                        build_rgb_quadtree_sample(observations[index], tree_config)
                    )
                samples.append(sample_cache[index])
            selected_observations = None
        tree_seconds = time.perf_counter() - tick
        prepared_rows.append({
            "stage": stage,
            "arena_config": asdict(arena_config),
            "targets": _aggregate_slots(example.target_occupancy),
            "future_rgb": example.future_rgb,
            "slot_targets": example.target_occupancy,
            "center_targets": example.target_center_density,
            "target_visibility": example.evaluator_target_visibility,
            "uncertainty_mask": example.evaluator_uncertainty_mask,
            "uncertainty_rate": float(example.evaluator_uncertainty_mask.float().mean()),
            "family_id": example.family_id,
            "physical_family_index": physical_family_index,
            "query_frame": example.query_frame,
            "curriculum_fraction": min(
                1.0,
                max(0.0, (start_update + example_offset) / max(updates, 1)),
            ),
            "raw_environment_seconds": raw_environment_family_seconds / count,
            "raw_environment_family_seconds": raw_environment_family_seconds,
            "samples": samples,
            "selected_observations": selected_observations,
            "worker_tree_seconds": tree_seconds,
            "frame_indices": indices,
            "family_reuse_count": count,
            "family_unique_tree_frames": 0,
            **causal_metadata,
        })
    unique_tree_frames = len(sample_cache)
    amortized_total = (time.perf_counter() - worker_start) / count
    for prepared in prepared_rows:
        prepared["worker_total_seconds"] = amortized_total
        prepared["family_unique_tree_frames"] = unique_tree_frames
        prepared["payload_bytes"] = _tensor_bytes(prepared)
    return prepared_rows


def selected_frame_indices(length: int, stride: int, maximum: int) -> Tuple[int, ...]:
    important = {0, min(1, length - 1), length - 1}
    important.update(range(0, length, stride))
    important.update(range(max(0, length - 3), length))
    ordered = sorted(important)
    if len(ordered) > maximum:
        # Always preserve cue frames and the most recent causal evidence.
        middle = ordered[2:-3]
        keep_middle = max(0, maximum - 5)
        if keep_middle and middle:
            positions = torch.linspace(0, len(middle) - 1, keep_middle).round().long()
            middle = [middle[int(position)] for position in positions]
        else:
            middle = []
        ordered = sorted(set(ordered[:2] + middle + ordered[-3:]))
    return tuple(ordered)


def encode_observations(
    model: QuadtreeWorldModelV2,
    observations: torch.Tensor,
    tree_config: RGBQuadtreeConfig,
    frame_indices: Sequence[int],
    *,
    reset_episode: bool = False,
    frontier_split_threshold: float = 0.5,
) -> Tuple[V2MemoryState, list, list, float, float]:
    state: Optional[V2MemoryState] = None
    outputs = []
    samples = []
    tree_seconds = 0.0
    forward_seconds = 0.0
    for index in frame_indices:
        tick = time.perf_counter()
        if tree_config.allocation_mode == "learned_frontier":
            sample = build_learned_frontier_sample(
                model, observations[index], state, tree_config,
                frontier_split_threshold,
            )
        else:
            sample = build_rgb_quadtree_sample(observations[index], tree_config)
        tree_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        output = model.observe(sample, state)
        forward_seconds += time.perf_counter() - tick
        state = output.state
        outputs.append(output)
        samples.append(sample)
    if state is None:
        raise RuntimeError("no observation frames were selected")
    if reset_episode:
        state = state.reset_episode()
    return state, outputs, samples, tree_seconds, forward_seconds


def encode_sparse_samples(
    model: QuadtreeWorldModelV2,
    samples: Sequence[Dict[str, torch.Tensor]],
    *,
    reset_episode: bool = False,
) -> Tuple[V2MemoryState, list, float]:
    state: Optional[V2MemoryState] = None
    outputs = []
    tick = time.perf_counter()
    for sample in samples:
        output = model.observe(sample, state)
        state = output.state
        outputs.append(output)
    if state is None:
        raise RuntimeError("no sparse observations were supplied")
    if reset_episode:
        state = state.reset_episode()
    return state, outputs, time.perf_counter() - tick


def build_learned_frontier_sample(
    model: QuadtreeWorldModelV2,
    image: torch.Tensor,
    state: Optional[V2MemoryState],
    tree_config: RGBQuadtreeConfig,
    split_threshold: float,
    exploration_probability: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Let the model traverse the full RGB source from root to max depth.

    Intermediate passes only choose the next frontier. The final pass in
    ``encode_learned_frontier`` commits one recurrent memory update, avoiding
    accidental multiple writes for a single video frame.
    """
    selected = {0}
    frontier = [0]
    for depth in range(tree_config.max_depth):
        sample = build_rgb_quadtree_sample(image, tree_config, selected)
        with torch.no_grad():
            proposal = model.observe(sample, state)
            probabilities = torch.sigmoid(proposal.split_logits).detach().cpu()
        row_for_address = {
            int(address): row
            for row, address in enumerate(sample["heap_indices"].detach().cpu().tolist())
        }
        split = [
            address for address in frontier
            if (
                float(probabilities[row_for_address[address]]) >= split_threshold
                or (
                    exploration_probability > 0.0
                    and bool(torch.rand(()) < exploration_probability)
                )
            )
        ]
        if not split:
            break
        frontier = []
        for address in split:
            children = [4 * address + offset for offset in (1, 2, 3, 4)]
            selected.update(children)
            frontier.extend(children)
    return build_rgb_quadtree_sample(image, tree_config, selected)


def encode_learned_frontier(
    model: QuadtreeWorldModelV2,
    observations: torch.Tensor,
    tree_config: RGBQuadtreeConfig,
    split_threshold: float,
    exploration_probability: float = 0.0,
) -> Tuple[V2MemoryState, list, list, float]:
    state: Optional[V2MemoryState] = None
    outputs = []
    samples = []
    start = time.perf_counter()
    for image in observations:
        sample = build_learned_frontier_sample(
            model, image, state, tree_config, split_threshold,
            exploration_probability,
        )
        output = model.observe(sample, state)
        state = output.state
        outputs.append(output)
        samples.append(sample)
    if state is None:
        raise RuntimeError("no observation frames were supplied")
    return state, outputs, samples, time.perf_counter() - start


def _rgb_bit_counts(
    image: torch.Tensor,
    addresses: torch.Tensor,
    canvas_size: int,
    reference: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact sufficient statistics for 24 independent RGB bit codes per node."""
    quantized = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    shifts = torch.arange(8, device=image.device, dtype=torch.int64)
    bits = ((quantized[:, None] >> shifts[None, :, None, None]) & 1)
    if reference is not None:
        reference_u8 = (reference.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
        reference_bits = ((reference_u8[:, None] >> shifts[None, :, None, None]) & 1)
        bits = bits ^ reference_bits
    bits = bits.to(image.dtype)
    flat = bits.reshape(24, image.shape[-2], image.shape[-1])
    integral = F.pad(flat.cumsum(-2).cumsum(-1), (1, 0, 1, 0))
    bounds = [address_to_bounds(int(a), canvas_size) for a in addresses.detach().cpu().tolist()]
    x0 = torch.tensor([b[1] for b in bounds], device=image.device).clamp_max(image.shape[-1])
    y0 = torch.tensor([b[2] for b in bounds], device=image.device).clamp_max(image.shape[-2])
    x1 = torch.tensor([b[1] + b[3] for b in bounds], device=image.device).clamp_max(image.shape[-1])
    y1 = torch.tensor([b[2] + b[3] for b in bounds], device=image.device).clamp_max(image.shape[-2])
    ones = (
        integral[:, y1, x1] - integral[:, y0, x1]
        - integral[:, y1, x0] + integral[:, y0, x0]
    ).transpose(0, 1)
    pixels = ((x1 - x0) * (y1 - y0)).to(image.dtype)
    return ones.reshape(-1, 3, 8), pixels


def recursive_rgb_bit_loss(
    addresses: torch.Tensor,
    split_logits: torch.Tensor,
    rgb_bit_logits: torch.Tensor,
    target_rgb: torch.Tensor,
    canvas_size: int,
    max_depth: int,
    reference_rgb: Optional[torch.Tensor] = None,
    structure_temperature_bpp: float = 0.0,
    minimum_depth: int = 0,
    predictive_logit_soft_clip: float = 0.0,
    decode_prediction: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Marginalize every valid STOP/SPLIT subtree in the supplied frontier."""
    ones, pixels = _rgb_bit_counts(
        target_rgb, addresses, canvas_size, reference=reference_rgb,
    )
    if predictive_logit_soft_clip > 0.0:
        clip = rgb_bit_logits.new_tensor(predictive_logit_soft_clip)
        loss_rgb_bit_logits = clip * torch.tanh(rgb_bit_logits / clip)
    else:
        loss_rgb_bit_logits = rgb_bit_logits
    log_content = (
        ones * F.logsigmoid(loss_rgb_bit_logits)
        + (pixels[:, None, None] - ones) * F.logsigmoid(-loss_rgb_bit_logits)
    ).sum((1, 2))
    work_addresses = addresses.to(split_logits.device)
    depths = torch.tensor(
        [address_to_bounds(int(a), canvas_size)[0] for a in addresses.detach().cpu().tolist()],
        device=split_logits.device,
    )
    sorted_addresses, sorted_rows = torch.sort(work_addresses)
    log_z = log_content.new_zeros(log_content.shape)
    local_bpp = log_content.new_zeros(log_content.shape)
    posterior_split = split_logits.new_zeros(split_logits.shape)
    decision_mask = torch.zeros_like(split_logits, dtype=torch.bool)
    evidence_gap_bpp = split_logits.new_zeros(split_logits.shape)
    stop_flag_bits = split_logits.new_zeros(split_logits.shape)
    split_flag_bits = split_logits.new_zeros(split_logits.shape)
    offsets = work_addresses.new_tensor((1, 2, 3, 4))
    temperature = split_logits.new_tensor(structure_temperature_bpp)
    for depth in range(max_depth, -1, -1):
        rows = torch.nonzero(depths == depth, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        denominator = (pixels[rows] * 24.0).clamp_min(1.0)
        level_log_z = log_content[rows]
        level_local_bpp = -level_log_z / (math.log(2.0) * denominator)
        if depth < max_depth:
            child_addresses = 4 * work_addresses[rows, None] + offsets[None, :]
            positions = torch.searchsorted(sorted_addresses, child_addresses)
            safe_positions = positions.clamp_max(max(work_addresses.numel() - 1, 0))
            children_exist = (
                (positions < work_addresses.numel())
                & (sorted_addresses[safe_positions] == child_addresses)
            )
            can_split = children_exist.all(dim=1)
            if bool(can_split.any()):
                split_rows = rows[can_split]
                child_rows = sorted_rows[safe_positions[can_split]]
                decision_mask[split_rows] = True
                stop = F.logsigmoid(-split_logits[split_rows]) + log_content[split_rows]
                split = F.logsigmoid(split_logits[split_rows]) + log_z[child_rows].sum(1)
                stop_flags = -F.logsigmoid(-split_logits[split_rows]) / math.log(2.0)
                split_flags = -F.logsigmoid(split_logits[split_rows]) / math.log(2.0)
                stop_flag_bits[split_rows] = stop_flags
                split_flag_bits[split_rows] = split_flags
                split_denominator = denominator[can_split]
                stop_cost = -stop / (math.log(2.0) * split_denominator)
                forced = depths[split_rows] < minimum_depth
                if structure_temperature_bpp > 0.0:
                    child_content = (
                        pixels[child_rows] * 24.0 * local_bpp[child_rows]
                    ).sum(1)
                    learned_split_cost = (split_flags + child_content) / split_denominator
                    forced_split_cost = child_content / split_denominator
                    split_cost = torch.where(forced, forced_split_cost, learned_split_cost)
                    split_local_bpp = -temperature * torch.logsumexp(
                        torch.stack((-stop_cost / temperature, -split_cost / temperature)), dim=0,
                    )
                    split_local_bpp = torch.where(forced, split_cost, split_local_bpp)
                    split_log_z = -split_local_bpp * split_denominator * math.log(2.0)
                    split_posterior = torch.sigmoid(
                        ((stop_cost - split_cost) / temperature).detach()
                    )
                else:
                    forced_log_z = log_z[child_rows].sum(1)
                    split_log_z = torch.where(forced, forced_log_z, torch.logaddexp(stop, split))
                    split_local_bpp = -split_log_z / (math.log(2.0) * split_denominator)
                    split_cost = -split / (math.log(2.0) * split_denominator)
                    split_posterior = torch.sigmoid((split - stop).detach())
                split_posterior = torch.where(forced, torch.ones_like(split_posterior), split_posterior)
                evidence_gap_bpp[split_rows] = (stop_cost - split_cost).detach()
                posterior_split[split_rows] = split_posterior
                level_log_z = level_log_z.index_copy(0, torch.nonzero(can_split).flatten(), split_log_z)
                level_local_bpp = level_local_bpp.index_copy(
                    0, torch.nonzero(can_split).flatten(), split_local_bpp,
                )
        log_z = log_z.index_copy(0, rows, level_log_z)
        local_bpp = local_bpp.index_copy(0, rows, level_local_bpp)
    root_rows = torch.nonzero(work_addresses == 0, as_tuple=False).flatten()
    if root_rows.numel() != 1:
        raise ValueError("recursive tree support must contain root address 0")
    root = int(root_rows.item())
    n_valid_pixels = target_rgb.shape[-2] * target_rgb.shape[-1] * 24
    loss_bpp = (
        local_bpp[root]
        if structure_temperature_bpp > 0.0
        else -log_z[root] / (math.log(2.0) * n_valid_pixels)
    )

    # Reach and reconstruction are diagnostics, never part of the optimized
    # code length. Keep them outside autograd; training can skip the expensive
    # per-node raster entirely and decode only during evaluation.
    with torch.no_grad():
        reach = split_logits.new_zeros(split_logits.shape)
        reach[root] = 1.0
        for depth in range(max_depth):
            rows = torch.nonzero((depths == depth) & decision_mask, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            child_addresses = 4 * work_addresses[rows, None] + offsets[None, :]
            positions = torch.searchsorted(sorted_addresses, child_addresses)
            child_rows = sorted_rows[positions]
            reach[child_rows.flatten()] = (
                reach[rows, None] * posterior_split[rows, None]
            ).expand(-1, 4).flatten()
        stop_mass = reach * (1.0 - posterior_split)
        expected_nodes = reach.sum()
        depth_mass = (stop_mass * depths).sum()
    rgb_prediction = target_rgb.new_zeros((3, target_rgb.shape[-2], target_rgb.shape[-1]))
    if decode_prediction:
        bit_values = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=split_logits.device)
        node_rgb = (torch.sigmoid(rgb_bit_logits.detach()) * bit_values).sum(-1) / 255.0
        for row, address in enumerate(addresses.detach().cpu().tolist()):
            _, x, y, size = address_to_bounds(int(address), canvas_size)
            x1, y1 = min(x + size, target_rgb.shape[-1]), min(y + size, target_rgb.shape[-2])
            if x < x1 and y < y1:
                rgb_prediction[:, y:y1, x:x1] += stop_mass[row] * node_rgb[row, :, None, None]
    if decode_prediction and reference_rgb is not None:
        # Decode expected XOR bits against the observed frame. This makes a
        # zero-innovation prediction equal persistence rather than black.
        reference_u8 = (reference_rgb.clamp(0, 1) * 255).round().to(torch.int64)
        shifts = torch.arange(8, device=reference_rgb.device, dtype=torch.int64)
        reference_bits = ((reference_u8[:, None] >> shifts[None, :, None, None]) & 1).to(reference_rgb.dtype)
        flip_probability = rgb_prediction.new_zeros((3, 8, *target_rgb.shape[-2:]))
        for row, address in enumerate(addresses.detach().cpu().tolist()):
            _, x, y, size = address_to_bounds(int(address), canvas_size)
            x1, y1 = min(x + size, target_rgb.shape[-1]), min(y + size, target_rgb.shape[-2])
            if x < x1 and y < y1:
                flip_probability[:, :, y:y1, x:x1] += (
                    stop_mass[row] * torch.sigmoid(rgb_bit_logits[row].detach())[:, :, None, None]
                )
        future_bit_probability = (
            reference_bits * (1.0 - flip_probability)
            + (1.0 - reference_bits) * flip_probability
        )
        rgb_prediction = (future_bit_probability * bit_values[None, :, None, None]).sum(1) / 255.0
    reached_decisions = decision_mask & (reach > 0.0)
    if bool(reached_decisions.any()):
        decision_reach = reach[reached_decisions]
        decision_probability = posterior_split[reached_decisions].clamp(1e-7, 1.0 - 1e-7)
        weight = decision_reach / decision_reach.sum().clamp_min(1e-8)
        posterior_entropy = -(
            decision_probability * torch.log2(decision_probability)
            + (1.0 - decision_probability) * torch.log2(1.0 - decision_probability)
        )
        gaps = evidence_gap_bpp[reached_decisions]
        gap_quantiles = torch.quantile(gaps.detach(), gaps.new_tensor([0.1, 0.5, 0.9]))
        weighted_split_probability = (weight * decision_probability).sum()
        weighted_posterior_entropy = (weight * posterior_entropy).sum()
        posterior_saturation = (
            (decision_probability < 0.01) | (decision_probability > 0.99)
        ).to(weight.dtype)
        weighted_saturation = (weight * posterior_saturation).sum()
        weighted_stop_flag_bits = (weight * stop_flag_bits[reached_decisions]).sum()
        weighted_split_flag_bits = (weight * split_flag_bits[reached_decisions]).sum()
    else:
        zero = loss_bpp.detach() * 0.0
        gap_quantiles = torch.stack((zero, zero, zero))
        weighted_split_probability = zero
        weighted_posterior_entropy = zero
        weighted_saturation = zero
        weighted_stop_flag_bits = zero
        weighted_split_flag_bits = zero
    if predictive_logit_soft_clip > 0.0:
        logit_clip_fraction = (
            rgb_bit_logits.detach().abs() > predictive_logit_soft_clip
        ).to(rgb_bit_logits.dtype).mean()
    else:
        logit_clip_fraction = rgb_bit_logits.detach().new_zeros(())
    diagnostics = {
        "rgb_bpp": float(loss_bpp.detach().cpu()),
        "posterior_expected_nodes": float(expected_nodes.detach().cpu()),
        "posterior_mean_stop_depth": float(depth_mass.detach().cpu()),
        "posterior_split_mean": float(posterior_split.mean().detach().cpu()),
        "reachable_split_probability": float(weighted_split_probability.detach().cpu()),
        "posterior_entropy_bits": float(weighted_posterior_entropy.detach().cpu()),
        "posterior_saturation_fraction": float(weighted_saturation.detach().cpu()),
        "evidence_gap_bpp_q10": float(gap_quantiles[0].cpu()),
        "evidence_gap_bpp_q50": float(gap_quantiles[1].cpu()),
        "evidence_gap_bpp_q90": float(gap_quantiles[2].cpu()),
        "stop_flag_bits": float(weighted_stop_flag_bits.detach().cpu()),
        "split_flag_bits": float(weighted_split_flag_bits.detach().cpu()),
        "predictive_logit_abs_max": float(rgb_bit_logits.detach().abs().max().cpu()),
        "predictive_logit_soft_clip_fraction": float(logit_clip_fraction.cpu()),
        "predictive_logit_soft_clip": float(predictive_logit_soft_clip),
        "prediction_spatial_std": float(
            rgb_prediction.flatten(1).std(dim=1).mean().detach().cpu()
        ) if decode_prediction else 0.0,
        "innovation_rate": float(
            (ones[root].sum() / (pixels[root] * 24).clamp_min(1.0)).detach().cpu()
        ) if reference_rgb is not None else 0.0,
        "structure_temperature_bpp": float(structure_temperature_bpp),
        "_posterior_split_target": posterior_split.detach(),
        "_posterior_reach": reach.detach(),
    }
    return loss_bpp, diagnostics, rgb_prediction


@torch.no_grad()
def learned_prediction_frontier(
    model: QuadtreeWorldModelV2,
    state: V2MemoryState,
    current_addresses: torch.Tensor,
    *,
    max_nodes: Optional[int],
    split_threshold: float,
    exploration_probability: float = 0.0,
    exploration_paths: int = 0,
) -> torch.Tensor:
    """Recursively expose children selected only by the learned compression prior.

    The realized future target is never consulted. Complete sibling groups keep
    the final STOP/SPLIT marginal exact on the proposed support. A node cap
    bounds compute, while stochastic training-only exploration prevents a
    prematurely stopped parent from permanently hiding all deeper addresses.
    """
    if not 0.0 <= split_threshold <= 1.0:
        raise ValueError("candidate split threshold must lie in [0, 1]")
    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("candidate exploration must lie in [0, 1]")
    if exploration_paths < 0:
        raise ValueError("candidate exploration paths must be non-negative")
    selected = set(int(value) for value in current_addresses.detach().cpu().tolist())
    selected.update(range(5))  # Root plus a target-independent global depth-1 floor.
    if max_nodes is not None and len(selected) > max_nodes:
        raise ValueError("initial predictive support exceeds candidate max nodes")
    # Support exploration is distinct from allocation: these complete-sibling
    # paths expose compression gradients at every depth, but posterior reach is
    # still the product of learned parent SPLIT probabilities.
    for path_index in range(exploration_paths):
        address = 1 + path_index % 4
        for _ in range(1, model.config.max_depth):
            children = [4 * address + offset for offset in (1, 2, 3, 4)]
            missing = [child for child in children if child not in selected]
            if max_nodes is not None and len(selected) + len(missing) > max_nodes:
                break
            selected.update(children)
            address = children[int(torch.randint(0, 4, ()).item())]
    frontier = [address for address in selected if address != 0]
    device = current_addresses.device
    for _ in range(1, model.config.max_depth):
        ordered = sorted(
            selected,
            key=lambda address: (
                address_to_bounds(address, model.config.canvas_size)[0], address,
            ),
        )
        candidates = torch.tensor(ordered, dtype=torch.long, device=device)
        # The coding objective averages horizon losses, so proposal reachability
        # must use the same aggregation. A max here lets one unstable horizon
        # materialize the complete tree for every horizon.
        scores = torch.stack([
            torch.sigmoid(model.predict(state, candidates, horizon).split_logits)
            for horizon in model.supported_horizons
        ]).mean(0).detach().cpu()
        row_for_address = {address: row for row, address in enumerate(ordered)}
        proposals = []
        for address in frontier:
            depth = address_to_bounds(address, model.config.canvas_size)[0]
            if depth >= model.config.max_depth:
                continue
            probability = float(scores[row_for_address[address]])
            if exploration_paths > 0 or exploration_probability > 0.0:
                # Uncapped ancestral traversal. Reachability is sampled from
                # the learned SPLIT probability, with a small exploration
                # floor; no independent local threshold controls training
                # compute and every child still requires its parent SPLIT.
                effective_probability = 1.0 - (
                    (1.0 - probability) * (1.0 - exploration_probability)
                )
                split = bool(torch.rand(()) < effective_probability)
                explored = split and bool(probability < split_threshold)
            else:
                # Deterministic evaluation view only.
                split = probability >= split_threshold
                explored = False
            if split:
                proposals.append((probability, explored, address))
        available = (
            len(proposals)
            if max_nodes is None
            else max(0, (max_nodes - len(selected)) // 4)
        )
        if available == 0 or not proposals:
            break
        proposals.sort(key=lambda item: (-int(item[1]), -item[0], item[2]))
        frontier = []
        for _, _, address in proposals[:available]:
            children = [4 * address + offset for offset in (1, 2, 3, 4)]
            selected.update(children)
            frontier.extend(children)
    ordered = sorted(
        selected,
        key=lambda address: (
            address_to_bounds(address, model.config.canvas_size)[0], address,
        ),
    )
    return torch.tensor(ordered, dtype=torch.long, device=device)


def prediction_loss(
    model: QuadtreeWorldModelV2,
    state: V2MemoryState,
    addresses: torch.Tensor,
    targets: torch.Tensor,
    candidate_max_nodes: Optional[int],
    memory_cost: float,
    bce_weight: float = 1.0,
    iou_weight: float = 0.0,
    brier_weight: float = 0.0,
    mass_weight: float = 0.0,
    pyramid_support_weight: float = 0.0,
    target_depth_weight: float = 0.0,
    target_depth_goal: float = 5.0,
    node_budget: float = 64.0,
    budget_penalty: float = 0.0,
    slot_targets: Optional[torch.Tensor] = None,
    center_targets: Optional[torch.Tensor] = None,
    target_visibility: Optional[torch.Tensor] = None,
    uncertainty_mask: Optional[torch.Tensor] = None,
    candidate_expansion_levels: int = 1,
    union_loss_weight: float = 1.0,
    slot_loss_weight: float = 0.0,
    slot_center_weight: float = 0.0,
    horizon_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
    objective: str = "legacy",
    address_budget_bits: float = 256.0,
    dual_price: float = 0.0,
    future_rgb: Optional[torch.Tensor] = None,
    current_rgb: Optional[torch.Tensor] = None,
    structure_temperature_bpp: float = 0.0,
    candidate_selection: str = "uniform",
    candidate_split_threshold: float = 0.5,
    candidate_exploration: float = 0.0,
    candidate_exploration_paths: int = 0,
    minimum_prediction_depth: int = 0,
    proposal_distillation_weight: float = 1.0,
    predictive_logit_soft_clip: float = 0.0,
    decode_rgb_prediction: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[int, Dict[str, torch.Tensor]]]:
    objective_mode = objective
    candidate_tick = time.perf_counter()
    if candidate_selection == "learned_frontier":
        candidates = learned_prediction_frontier(
            model, state, addresses,
            max_nodes=candidate_max_nodes,
            split_threshold=candidate_split_threshold,
            exploration_probability=candidate_exploration,
            exploration_paths=candidate_exploration_paths,
        )
    else:
        effective_max_nodes = candidate_max_nodes
        if effective_max_nodes is None:
            effective_max_nodes = (4 ** (model.config.max_depth + 1) - 1) // 3
        candidates = model.prediction_candidates(
            addresses, expansion_levels=candidate_expansion_levels,
            max_nodes=effective_max_nodes, global_depth=1,
        )
    candidate_proposal_seconds = time.perf_counter() - candidate_tick
    losses = []
    loss_weights = []
    metrics = {
        "candidate_nodes": float(candidates.numel()),
        "candidate_max_depth": float(max(
            address_to_bounds(int(address), model.config.canvas_size)[0]
            for address in candidates.detach().cpu().tolist()
        )),
        "candidate_proposal_seconds": candidate_proposal_seconds,
    }
    predictions = {}
    candidate_depths = torch.tensor(
        [address_to_bounds(int(address), model.config.canvas_size)[0]
         for address in candidates.detach().cpu().tolist()],
        dtype=torch.float32,
        device=candidates.device,
    )
    metrics["candidate_depth8_fraction"] = float(
        (candidate_depths == model.config.max_depth).to(torch.float32).mean().detach().cpu()
    )
    if objective_mode in ("recursive_rgb_bits", "recursive_rgb_innovation_bits"):
        if future_rgb is None:
            raise ValueError("recursive_rgb_bits requires future RGB frames")
        if objective_mode == "recursive_rgb_innovation_bits" and current_rgb is None:
            raise ValueError("recursive_rgb_innovation_bits requires current RGB")
        predict_seconds = 0.0
        compression_seconds = 0.0
        for horizon_index, horizon in enumerate(model.supported_horizons):
            tick = time.perf_counter()
            output = model.predict(state, candidates, horizon=horizon)
            predict_seconds += time.perf_counter() - tick
            if output.rgb_bit_logits is None:
                raise RuntimeError("model does not expose RGB bit logits")
            tick = time.perf_counter()
            loss_bpp, diagnostics, rgb_prediction = recursive_rgb_bit_loss(
                candidates,
                output.split_logits,
                output.rgb_bit_logits,
                future_rgb[horizon_index].to(output.split_logits.device),
                model.config.canvas_size,
                model.config.max_depth,
                reference_rgb=(
                    current_rgb.to(output.split_logits.device)
                    if objective_mode == "recursive_rgb_innovation_bits" and current_rgb is not None
                    else None
                ),
                structure_temperature_bpp=(
                    structure_temperature_bpp
                    if objective_mode == "recursive_rgb_innovation_bits" else 0.0
                ),
                minimum_depth=minimum_prediction_depth,
                predictive_logit_soft_clip=predictive_logit_soft_clip,
                decode_prediction=decode_rgb_prediction,
            )
            posterior_target = diagnostics.pop("_posterior_split_target")
            posterior_reach = diagnostics.pop("_posterior_reach")
            address_rows = {
                int(address): row
                for row, address in enumerate(candidates.detach().cpu().tolist())
            }
            eligible = torch.tensor([
                minimum_prediction_depth <= int(candidate_depths[row]) < model.config.max_depth
                and all(4 * int(address) + offset in address_rows for offset in (1, 2, 3, 4))
                for row, address in enumerate(candidates.detach().cpu().tolist())
            ], dtype=torch.bool, device=output.split_logits.device)
            if bool(eligible.any()):
                decision_bits = F.binary_cross_entropy_with_logits(
                    output.split_logits[eligible], posterior_target[eligible], reduction="none",
                ) / math.log(2.0)
                decision_reach = posterior_reach[eligible]
                distillation_bits = (
                    (decision_reach * decision_bits).sum()
                    / decision_reach.sum().clamp_min(1.0)
                )
            else:
                distillation_bits = output.split_logits.sum() * 0.0
            compression_seconds += time.perf_counter() - tick
            losses.append(
                (loss_bpp + proposal_distillation_weight * distillation_bits)
                * horizon_weights[horizon_index]
            )
            loss_weights.append(horizon_weights[horizon_index])
            diagnostics["proposal_distillation_bits"] = float(distillation_bits.detach().cpu())
            metrics.update({f"h{horizon}_{key}": value for key, value in diagnostics.items()})
            predictions[horizon] = {"rgb": rgb_prediction.detach().cpu()}
        metrics["candidate_predict_seconds"] = predict_seconds
        metrics["compression_loss_seconds"] = compression_seconds
        return torch.stack(losses).sum() / max(sum(loss_weights), 1e-8), metrics, predictions
    for horizon_index, horizon in enumerate(model.supported_horizons):
        output = model.predict(state, candidates, horizon=horizon)
        target = targets[horizon_index].to(output.frame_logits.device).unsqueeze(0)
        pixel = balanced_pixel_bce(output.frame_logits, target)
        probability = output.frame_probabilities
        intersection = (probability * target).sum()
        union = (probability + target - probability * target).sum()
        soft_iou = intersection / union.clamp_min(1e-6)
        brier = (probability - target).square().mean()
        predicted_mass = probability.sum()
        target_mass = target.sum()
        yy, xx = torch.meshgrid(
            torch.arange(model.config.image_size, device=probability.device),
            torch.arange(model.config.image_size, device=probability.device),
            indexing="ij",
        )
        depth_values = (
            candidate_depths.to(output.frame_logits.device) / model.config.max_depth
        ).clamp(1e-4, 1.0 - 1e-4)
        depth_logits = torch.logit(depth_values)
        attention_depth, _ = soft_hierarchical_rasterize(
            candidates.to(output.frame_logits.device), depth_logits, output.hierarchy,
            image_size=model.config.image_size,
            canvas_size=model.config.canvas_size,
        )
        attention_depth = attention_depth * model.config.max_depth
        foreground = target >= 0.05
        background = ~foreground
        target_depth = (
            attention_depth[foreground].mean()
            if bool(foreground.any()) else attention_depth.mean()
        )
        background_depth = (
            attention_depth[background].mean()
            if bool(background.any()) else attention_depth.mean()
        )
        target_depth_shortfall = F.relu(
            target_depth_goal - target_depth,
        ) / model.config.max_depth
        mass_log_error = torch.abs(torch.log(
            (predicted_mass + 1.0) / (target_mass + 1.0)
        ))
        pyramid_support_losses = []
        for scale in (4, 8, 16):
            coarse_probability = F.avg_pool2d(probability, scale, scale)
            coarse_support = F.max_pool2d(target, scale, scale)
            outside_mass = (coarse_probability * (1.0 - coarse_support)).sum()
            pyramid_support_losses.append(
                outside_mass / coarse_probability.sum().clamp_min(1e-6)
            )
        pyramid_support_loss = torch.stack(pyramid_support_losses).mean()
        budget_overflow = F.relu(output.expected_nodes - node_budget) / node_budget
        address_bits = (
            output.hierarchy.reach
            * output.hierarchy.can_split.to(output.hierarchy.reach.dtype)
        ).sum()
        future_bits = F.binary_cross_entropy_with_logits(
            output.frame_logits, target, reduction="sum",
        ) / math.log(2.0)
        future_bpp = future_bits / target.numel()
        address_bpp = address_bits / target.numel()
        if objective_mode == "predictive_bits":
            union_objective = future_bpp + dual_price * (
                address_bits - address_budget_bits
            ) / target.numel()
        else:
            union_objective = (
                bce_weight * pixel
                + iou_weight * (1.0 - soft_iou)
                + brier_weight * brier
                + mass_weight * mass_log_error
                + pyramid_support_weight * pyramid_support_loss
                + target_depth_weight * target_depth_shortfall
                + memory_cost * output.expected_nodes
                + budget_penalty * budget_overflow.square()
            )
        slot_objective = probability.sum() * 0.0
        slot_center_objective = probability.sum() * 0.0
        slot_iou = probability.sum() * 0.0
        slot_mass_ratio = probability.sum() * 0.0
        slot_center_error = probability.sum() * 0.0
        slot_identity_accuracy = probability.sum() * 0.0
        slot_spatial_bits = probability.sum() * 0.0
        occluded_slot_iou = probability.sum() * 0.0
        visible_slot_iou = probability.sum() * 0.0
        uncertain_slot_iou = probability.sum() * 0.0
        occluded_slot_count = 0
        visible_slot_count = 0
        uncertain_slot_count = 0
        if output.slot_frame_probabilities is not None and slot_targets is not None:
            predicted_slots = output.slot_frame_probabilities[:, 0]
            predicted_slot_logits = output.slot_frame_logits[:, 0]
            raw_slot_targets = slot_targets[horizon_index].to(probability.device)
            raw_center_targets = (
                center_targets[horizon_index].to(probability.device)
                if center_targets is not None else None
            )
            valid_slots = min(raw_slot_targets.shape[0], predicted_slots.shape[0])
            slot_losses = []
            slot_center_losses = []
            slot_ious = []
            slot_mass_ratios = []
            slot_center_errors = []
            slot_spatial_code_lengths = []
            predicted_centers = []
            target_centers = []
            for slot in range(valid_slots):
                slot_probability = predicted_slots[slot]
                slot_target = raw_slot_targets[slot]
                slot_logit = predicted_slot_logits[slot].unsqueeze(0)
                slot_target_batch = slot_target.unsqueeze(0)
                target_density = slot_target / slot_target.sum().clamp_min(1e-8)
                spatial_log_probability = F.log_softmax(
                    predicted_slot_logits[slot].reshape(-1), dim=0,
                ).reshape_as(slot_target)
                slot_spatial_code_lengths.append(
                    -(target_density * spatial_log_probability).sum() / math.log(2.0)
                )
                slot_pixel = balanced_pixel_bce(slot_logit, slot_target_batch)
                slot_intersection = (slot_probability * slot_target).sum()
                slot_union = (
                    slot_probability + slot_target - slot_probability * slot_target
                ).sum()
                this_iou = slot_intersection / slot_union.clamp_min(1e-6)
                slot_predicted_mass = slot_probability.sum()
                slot_target_mass = slot_target.sum()
                slot_mass_log_error = torch.abs(torch.log(
                    (slot_predicted_mass + 1.0) / (slot_target_mass + 1.0)
                ))
                slot_losses.append(
                    bce_weight * slot_pixel
                    + iou_weight * (1.0 - this_iou)
                    + brier_weight * (slot_probability - slot_target).square().mean()
                    + mass_weight * slot_mass_log_error
                )
                slot_ious.append(this_iou)
                slot_mass_ratios.append(
                    slot_predicted_mass / slot_target_mass.clamp_min(1e-6)
                )
                if raw_center_targets is not None:
                    density = raw_center_targets[slot]
                    density = density / density.sum().clamp_min(1e-8)
                    predicted_density = slot_probability / slot_predicted_mass.clamp_min(1e-8)
                    center_kl = (
                        density * (
                            torch.log(density.clamp_min(1e-8))
                            - torch.log(predicted_density.clamp_min(1e-8))
                        )
                    ).sum() / math.log(float(density.numel()))
                    slot_center_losses.append(center_kl)
                    pred_xy = torch.stack((
                        (predicted_density * xx).sum(),
                        (predicted_density * yy).sum(),
                    ))
                    target_xy = torch.stack((
                        (density * xx).sum(), (density * yy).sum(),
                    ))
                    slot_center_errors.append(torch.linalg.vector_norm(pred_xy - target_xy))
                    predicted_centers.append(pred_xy)
                    target_centers.append(target_xy)
            if slot_losses:
                slot_objective = torch.stack(slot_losses).mean()
                slot_iou = torch.stack(slot_ious).mean()
                slot_mass_ratio = torch.stack(slot_mass_ratios).mean()
                slot_spatial_bits = torch.stack(slot_spatial_code_lengths).mean()
            if slot_center_losses:
                slot_center_objective = torch.stack(slot_center_losses).mean()
                slot_center_error = torch.stack(slot_center_errors).mean()
                pairwise_center_error = torch.cdist(
                    torch.stack(predicted_centers), torch.stack(target_centers),
                )
                nearest_target = pairwise_center_error.argmin(dim=1)
                slot_identity_accuracy = (
                    nearest_target == torch.arange(
                        valid_slots, device=nearest_target.device,
                    )
                ).float().mean()
            if slot_ious:
                slot_iou_values = torch.stack(slot_ious)
                if target_visibility is not None:
                    visibility = target_visibility[horizon_index, :valid_slots].to(
                        device=probability.device,
                    )
                    occluded = visibility < 0.5
                    visible = ~occluded
                    occluded_slot_count = int(occluded.sum().item())
                    visible_slot_count = int(visible.sum().item())
                    if occluded_slot_count:
                        occluded_slot_iou = slot_iou_values[occluded].mean()
                    if visible_slot_count:
                        visible_slot_iou = slot_iou_values[visible].mean()
                if uncertainty_mask is not None:
                    uncertain = uncertainty_mask[horizon_index, :valid_slots].to(
                        device=probability.device, dtype=torch.bool,
                    )
                    uncertain_slot_count = int(uncertain.sum().item())
                    if uncertain_slot_count:
                        uncertain_slot_iou = slot_iou_values[uncertain].mean()
            if predicted_slots.shape[0] > valid_slots:
                slot_objective = slot_objective + 0.1 * predicted_slots[valid_slots:].mean()
        if objective_mode == "predictive_bits" and slot_spatial_code_lengths:
            horizon_objective = slot_spatial_bits + dual_price * (
                address_bits - address_budget_bits
            ) / max(address_budget_bits, 1.0)
        else:
            horizon_objective = (
                union_loss_weight * union_objective
                + slot_loss_weight * slot_objective
                + slot_center_weight * slot_center_objective
            )
        losses.append(horizon_objective * horizon_weights[horizon_index])
        loss_weights.append(horizon_weights[horizon_index])

        hard_prediction = probability >= 0.5
        hard_target = target >= 0.5
        true_positive = (hard_prediction & hard_target).sum().float()
        precision = true_positive / hard_prediction.sum().clamp_min(1)
        recall = true_positive / hard_target.sum().clamp_min(1)
        hard_union = (hard_prediction | hard_target).sum().clamp_min(1)
        hard_iou = true_positive / hard_union
        pred_center = torch.stack((
            (probability[0] * xx).sum() / predicted_mass.clamp_min(1e-6),
            (probability[0] * yy).sum() / predicted_mass.clamp_min(1e-6),
        ))
        true_center = torch.stack((
            (target[0] * xx).sum() / target_mass.clamp_min(1e-6),
            (target[0] * yy).sum() / target_mass.clamp_min(1e-6),
        ))
        center_error = torch.linalg.vector_norm(pred_center - true_center)

        split_probability = output.hierarchy.split_probability[output.hierarchy.can_split]
        split_entropy = (
            -(split_probability * torch.log(split_probability.clamp_min(1e-6))
              + (1.0 - split_probability) * torch.log((1.0 - split_probability).clamp_min(1e-6))).mean()
            if split_probability.numel() else probability.sum() * 0.0
        )

        metrics[f"h{horizon}_pixel_loss"] = float(pixel.detach().cpu())
        metrics[f"h{horizon}_objective"] = float(horizon_objective.detach().cpu())
        metrics[f"h{horizon}_soft_iou"] = float(soft_iou.detach().cpu())
        metrics[f"h{horizon}_hard_iou"] = float(hard_iou.detach().cpu())
        metrics[f"h{horizon}_precision"] = float(precision.detach().cpu())
        metrics[f"h{horizon}_recall"] = float(recall.detach().cpu())
        metrics[f"h{horizon}_brier"] = float(brier.detach().cpu())
        metrics[f"h{horizon}_mass_ratio"] = float((predicted_mass / target_mass.clamp_min(1e-6)).detach().cpu())
        metrics[f"h{horizon}_mass_log_error"] = float(mass_log_error.detach().cpu())
        metrics[f"h{horizon}_pyramid_support_loss"] = float(
            pyramid_support_loss.detach().cpu()
        )
        metrics[f"h{horizon}_center_error_pixels"] = float(center_error.detach().cpu())
        metrics[f"h{horizon}_attention_target_depth"] = float(target_depth.detach().cpu())
        metrics[f"h{horizon}_attention_background_depth"] = float(background_depth.detach().cpu())
        metrics[f"h{horizon}_attention_depth_lift"] = float((target_depth - background_depth).detach().cpu())
        metrics[f"h{horizon}_target_depth_shortfall"] = float(
            target_depth_shortfall.detach().cpu()
        )
        metrics[f"h{horizon}_split_entropy"] = float(split_entropy.detach().cpu())
        metrics[f"h{horizon}_expected_nodes"] = float(output.expected_nodes.detach().cpu())
        metrics[f"h{horizon}_future_bits"] = float(future_bits.detach().cpu())
        metrics[f"h{horizon}_future_bpp"] = float(future_bpp.detach().cpu())
        metrics[f"h{horizon}_address_bits"] = float(address_bits.detach().cpu())
        metrics[f"h{horizon}_address_bpp"] = float(address_bpp.detach().cpu())
        metrics[f"h{horizon}_slot_spatial_bits"] = float(
            slot_spatial_bits.detach().cpu()
        )
        metrics[f"h{horizon}_budget_overflow"] = float(budget_overflow.detach().cpu())
        metrics[f"h{horizon}_slot_objective"] = float(slot_objective.detach().cpu())
        metrics[f"h{horizon}_slot_soft_iou"] = float(slot_iou.detach().cpu())
        metrics[f"h{horizon}_slot_mass_ratio"] = float(slot_mass_ratio.detach().cpu())
        metrics[f"h{horizon}_slot_center_kl"] = float(slot_center_objective.detach().cpu())
        metrics[f"h{horizon}_slot_center_error_pixels"] = float(
            slot_center_error.detach().cpu()
        )
        metrics[f"h{horizon}_slot_identity_accuracy"] = float(
            slot_identity_accuracy.detach().cpu()
        )
        metrics[f"h{horizon}_occluded_slot_iou"] = float(
            occluded_slot_iou.detach().cpu()
        )
        metrics[f"h{horizon}_visible_slot_iou"] = float(
            visible_slot_iou.detach().cpu()
        )
        metrics[f"h{horizon}_uncertain_slot_iou"] = float(
            uncertain_slot_iou.detach().cpu()
        )
        metrics[f"h{horizon}_occluded_slot_count"] = float(occluded_slot_count)
        metrics[f"h{horizon}_visible_slot_count"] = float(visible_slot_count)
        metrics[f"h{horizon}_uncertain_slot_count"] = float(uncertain_slot_count)
        if output.episode_attention_weights is not None:
            episode_weights = output.episode_attention_weights
            episode_entropy = -(
                episode_weights
                * torch.log(episode_weights.clamp_min(1e-8))
            ).sum(dim=-1).mean() / math.log(float(episode_weights.shape[-1]))
            metrics[f"h{horizon}_episode_attention_entropy"] = float(
                episode_entropy.detach().cpu()
            )
            metrics[f"h{horizon}_episode_attention_peak"] = float(
                episode_weights.max(dim=-1).values.mean().detach().cpu()
            )
        for depth in range(model.config.max_depth + 1):
            at_depth = candidate_depths == depth
            metrics[f"h{horizon}_stop_mass_depth_{depth}"] = float(
                output.hierarchy.stop[at_depth].sum().detach().cpu()
            )
        predictions[horizon] = {
            "probability": output.frame_probabilities.detach().cpu(),
            "slot_probabilities": (
                output.slot_frame_probabilities.detach().cpu()
                if output.slot_frame_probabilities is not None else None
            ),
            "attention_depth": attention_depth.detach().cpu(),
        }
    active_slots = state.episode_payloads[:max(1, model.config.prediction_slots)]
    metrics["episode_slot_std"] = float(active_slots.std(dim=0).mean().detach().cpu())
    if active_slots.shape[0] > 1:
        normalized_slots = F.normalize(active_slots, dim=-1)
        similarity = normalized_slots @ normalized_slots.transpose(0, 1)
        off_diagonal = ~torch.eye(
            active_slots.shape[0], dtype=torch.bool, device=active_slots.device,
        )
        metrics["episode_slot_cosine"] = float(
            similarity[off_diagonal].mean().detach().cpu()
        )
    return torch.stack(losses).sum() / max(sum(loss_weights), 1e-8), metrics, predictions


def train_update(
    model: QuadtreeWorldModelV2,
    optimizer: torch.optim.Optimizer,
    prepared: Dict[str, object],
    tree_config: RGBQuadtreeConfig,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[int, Dict[str, torch.Tensor]]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    samples = prepared["samples"]
    if tree_config.allocation_mode == "learned_frontier":
        state, outputs, samples, forward_seconds = encode_learned_frontier(
            model,
            prepared["selected_observations"],
            tree_config,
            args.frontier_split_threshold,
            args.frontier_exploration,
        )
    else:
        state, outputs, forward_seconds = encode_sparse_samples(model, samples)
    tree_seconds = float(prepared["worker_tree_seconds"])
    tick = time.perf_counter()
    horizon_weights = horizon_weights_for_stage(str(prepared["stage"]), args)
    curriculum_fraction = float(prepared.get("curriculum_fraction", 1.0))
    target_depth_scale, sparsity_scale = allocation_objective_scales(
        curriculum_fraction, args,
    )
    final_temperature = (
        args.structure_temperature_bpp
        if args.structure_temperature_final_bpp is None
        else args.structure_temperature_final_bpp
    )
    structure_temperature_bpp = (
        args.structure_temperature_bpp
        + curriculum_fraction * (final_temperature - args.structure_temperature_bpp)
    )
    future_loss, metrics, predictions = prediction_loss(
        model, state, samples[-1]["heap_indices"], prepared["targets"],
        args.candidate_max_nodes, args.memory_cost * sparsity_scale,
        args.bce_weight, args.iou_weight, args.brier_weight, args.mass_weight,
        args.pyramid_support_weight,
        args.target_depth_weight * target_depth_scale, args.target_depth_goal,
        args.node_budget, args.budget_penalty * sparsity_scale,
        slot_targets=prepared.get("slot_targets"),
        center_targets=prepared.get("center_targets"),
        target_visibility=prepared.get("target_visibility"),
        uncertainty_mask=prepared.get("uncertainty_mask"),
        candidate_expansion_levels=args.candidate_expansion_levels,
        union_loss_weight=args.union_loss_weight,
        slot_loss_weight=args.slot_loss_weight,
        slot_center_weight=args.slot_center_weight,
        horizon_weights=horizon_weights,
        objective=args.objective,
        address_budget_bits=args.address_budget_bits,
        dual_price=args._dual_price,
        future_rgb=prepared.get("future_rgb"),
        current_rgb=(
            prepared["selected_observations"][-1]
            if prepared.get("selected_observations") is not None else None
        ),
        structure_temperature_bpp=structure_temperature_bpp,
        candidate_selection=args.candidate_selection,
        candidate_split_threshold=args.candidate_split_threshold,
        candidate_exploration=args.candidate_exploration,
        candidate_exploration_paths=args.candidate_exploration_paths,
        minimum_prediction_depth=args.minimum_prediction_depth,
        proposal_distillation_weight=args.proposal_distillation_weight,
        predictive_logit_soft_clip=args.predictive_logit_soft_clip,
        decode_rgb_prediction=False,
    )
    forward_seconds += time.perf_counter() - tick

    observation_losses = []
    split_losses = []
    for output, sample in zip(outputs[-3:], samples[-3:]):
        observation_losses.append(balanced_pixel_bce(
            output.frame_logits, sample["image"].to(output.frame_logits.device),
        ))
        split_losses.append(depth_class_balanced_split_bce(
            output.split_logits,
            sample["split_targets"].to(output.split_logits.device),
            sample["depths"].to(output.split_logits.device),
            model.config.max_depth,
        ))
    observation_loss = torch.stack(observation_losses).mean()
    split_loss = torch.stack(split_losses).mean()
    if model.config.prediction_slots:
        active_slot_count = model.config.prediction_slots
        effective_slots = state.episode_payloads[:active_slot_count] + model.episode_slot_embedding[:active_slot_count]
        normalized_slots = F.normalize(effective_slots, dim=-1)
        slot_gram = normalized_slots @ normalized_slots.transpose(0, 1)
        off_diagonal = ~torch.eye(active_slot_count, dtype=torch.bool, device=slot_gram.device)
        slot_diversity_loss = slot_gram[off_diagonal].square().mean() if bool(off_diagonal.any()) else slot_gram.sum() * 0.0
    else:
        slot_diversity_loss = future_loss * 0.0
    if args.objective in (
        "predictive_bits", "recursive_rgb_bits", "recursive_rgb_innovation_bits",
    ):
        loss = future_loss
    else:
        loss = (
            future_loss
            + args.observation_loss_weight * observation_loss
            + args.split_loss_weight * split_loss
            + args.slot_diversity_weight * slot_diversity_loss
        )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite training objective")
    tick = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - tick
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
    tick = time.perf_counter()
    optimizer.step()
    optimizer_seconds = time.perf_counter() - tick
    if args.objective == "predictive_bits":
        mean_address_bits = sum(
            metrics[f"h{horizon}_address_bits"]
            for horizon in model.supported_horizons
        ) / len(model.supported_horizons)
        args._dual_price = min(
            args.dual_price_max,
            max(0.0, args._dual_price + args.dual_learning_rate * (
                mean_address_bits / max(args.address_budget_bits, 1.0) - 1.0
            )),
        )

    rows = [sample["memory"].shape[0] for sample in samples]
    metrics.update({
        "total_loss": float(loss.detach().cpu()),
        "prediction_loss": float(future_loss.detach().cpu()),
        "observation_loss": float(observation_loss.detach().cpu()),
        "split_loss": float(split_loss.detach().cpu()),
        "slot_diversity_loss": float(slot_diversity_loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "gradient_clipped": float(gradient_norm > args.gradient_clip),
        "dual_price": float(args._dual_price),
        "tree_seconds": tree_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "observation_frames": len(samples),
        "mean_rows_per_frame": _mean(rows),
        "max_rows_per_frame": max(rows),
        "persistent_memory_rows": int(state.spatial_addresses.numel()),
        "h1_loss_weight": horizon_weights[0],
        "h4_loss_weight": horizon_weights[1],
        "h8_loss_weight": horizon_weights[2],
        "target_depth_scale": target_depth_scale,
        "sparsity_scale": sparsity_scale,
    })
    return metrics, predictions


@torch.no_grad()
def evaluate_ablations(
    model: QuadtreeWorldModelV2,
    prepared_rows: Sequence[Dict[str, object]],
    tree_config: RGBQuadtreeConfig,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    model.eval()
    conditions = (
        "full", "reset_episode", "recent_only", "no_cue", "grayscale",
        "shuffled_history", "reversed_history",
    )
    scores: Dict[str, list] = {condition: [] for condition in conditions}
    condition_times: Dict[str, list] = {condition: [] for condition in conditions}
    condition_metrics: Dict[str, Dict[str, list]] = {condition: {} for condition in conditions}
    qualitative = {}
    for prepared_index, prepared in enumerate(prepared_rows):
        original = prepared["observations"]
        base_indices = selected_frame_indices(
            original.shape[0], args.history_stride, args.max_observation_frames,
        )
        for condition in conditions:
            condition_start = time.perf_counter()
            observations = original.clone() if condition in (
                "no_cue", "grayscale", "shuffled_history", "reversed_history",
            ) else original
            indices = base_indices
            if condition == "recent_only":
                indices = tuple(range(max(0, original.shape[0] - 3), original.shape[0]))
            elif condition == "no_cue":
                observations[:2] = 0.0
            elif condition == "grayscale":
                gray = 0.2126 * observations[:, 0:1] + 0.7152 * observations[:, 1:2] + 0.0722 * observations[:, 2:3]
                observations = gray.repeat(1, 3, 1, 1)
            elif condition == "shuffled_history" and len(indices) > 5:
                middle = list(indices[2:-3])
                generator = random.Random(args.seed + prepared_index)
                generator.shuffle(middle)
                indices = tuple(indices[:2]) + tuple(middle) + tuple(indices[-3:])
            elif condition == "reversed_history" and len(indices) > 3:
                # Preserve identity cue order and the causal query frame, while
                # reversing the motion evidence between them.
                indices = tuple(indices[:2]) + tuple(reversed(indices[2:-1])) + tuple(indices[-1:])
            state, _, samples, _, _ = encode_observations(
                model, observations, tree_config, indices,
                reset_episode=condition == "reset_episode",
                frontier_split_threshold=args.frontier_split_threshold,
            )
            loss, horizon_metrics, predictions = prediction_loss(
                model, state, samples[-1]["heap_indices"], prepared["targets"],
                args.candidate_max_nodes, args.memory_cost,
                args.bce_weight, args.iou_weight, args.brier_weight, args.mass_weight,
                args.pyramid_support_weight,
                args.target_depth_weight, args.target_depth_goal,
                args.node_budget, args.budget_penalty,
                slot_targets=prepared.get("slot_targets"),
                center_targets=prepared.get("center_targets"),
                target_visibility=prepared.get("target_visibility"),
                uncertainty_mask=prepared.get("uncertainty_mask"),
                candidate_expansion_levels=args.candidate_expansion_levels,
                union_loss_weight=args.union_loss_weight,
                slot_loss_weight=args.slot_loss_weight,
                slot_center_weight=args.slot_center_weight,
                horizon_weights=(args.h1_weight, args.h4_weight, args.h8_weight),
                objective=args.objective,
                address_budget_bits=args.address_budget_bits,
                dual_price=args._dual_price,
                future_rgb=prepared.get("future_rgb"),
                current_rgb=original[-1],
                structure_temperature_bpp=(
                    args.structure_temperature_bpp
                    if args.structure_temperature_final_bpp is None
                    else args.structure_temperature_final_bpp
                ),
                candidate_selection=args.candidate_selection,
                candidate_split_threshold=args.candidate_split_threshold,
                candidate_exploration=0.0,
                candidate_exploration_paths=0,
                minimum_prediction_depth=args.minimum_prediction_depth,
                proposal_distillation_weight=args.proposal_distillation_weight,
                predictive_logit_soft_clip=args.predictive_logit_soft_clip,
            )
            scores[condition].append(float(loss.cpu()))
            for name, value in horizon_metrics.items():
                condition_metrics[condition].setdefault(name, []).append(float(value))
            condition_times[condition].append(time.perf_counter() - condition_start)
            if prepared_index == 0 and condition == "full":
                qualitative = {
                    "query_rgb": original[-1].cpu(),
                    "targets": prepared["targets"].cpu(),
                    "slot_targets": prepared["slot_targets"].cpu(),
                    "target_visibility": prepared["target_visibility"].cpu(),
                    "uncertainty_mask": prepared["uncertainty_mask"].cpu(),
                    "predictions": torch.stack([
                        predictions[h]["rgb"] for h in model.supported_horizons
                    ]) if args.objective in ("recursive_rgb_bits", "recursive_rgb_innovation_bits") else torch.cat([
                        predictions[h]["probability"] for h in model.supported_horizons
                    ]),
                    "future_rgb": prepared.get("future_rgb"),
                }
                if args.objective not in ("recursive_rgb_bits", "recursive_rgb_innovation_bits"):
                    qualitative["attention_depth"] = torch.cat([
                        predictions[h]["attention_depth"] for h in model.supported_horizons
                    ])
                    if predictions[model.supported_horizons[0]]["slot_probabilities"] is not None:
                        qualitative["slot_predictions"] = torch.stack([
                            predictions[h]["slot_probabilities"][:, 0]
                            for h in model.supported_horizons
                        ])
    summary = {
        condition: {
            "prediction_loss": _mean(rows),
            "families": len(rows),
            "mean_wall_seconds": _mean(condition_times[condition]),
            "metrics": {
                name: _mean(values)
                for name, values in condition_metrics[condition].items()
            },
        }
        for condition, rows in scores.items()
    }
    full = summary["full"]["prediction_loss"]
    for condition in conditions[1:]:
        summary[condition]["delta_vs_full"] = summary[condition]["prediction_loss"] - full
    return summary, qualitative


def hardware_manifest(device: torch.device) -> Dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def main() -> None:
    args = parse_args()
    args._dual_price = args.dual_price_initial
    if args.smoke:
        args.updates = min(args.updates, 2)
        args.eval_families = min(args.eval_families, 1)
        if args.tree_max_nodes is not None:
            args.tree_max_nodes = min(args.tree_max_nodes, 21)
        if args.tree_allocation_mode != "learned_frontier" and args.candidate_max_nodes is not None:
            args.candidate_max_nodes = min(args.candidate_max_nodes, 85)
        args.max_observation_frames = min(args.max_observation_frames, 6)
        args.prefetch_workers = 1
        args.tree_active_depth = min(args.tree_active_depth, 2)
    if args.prefetch_workers < 1:
        raise ValueError("prefetch-workers must be positive")
    if args.learner_threads < 1:
        raise ValueError("learner-threads must be positive")
    if args.examples_per_family < 1:
        raise ValueError("examples-per-family must be positive")
    if args.candidate_max_nodes is not None and args.candidate_max_nodes < 5:
        raise ValueError("candidate-max-nodes must be at least five when supplied")
    if args.proposal_distillation_weight < 0.0:
        raise ValueError("proposal-distillation-weight must be non-negative")
    if args.structure_temperature_bpp < 0.0:
        raise ValueError("structure-temperature-bpp must be non-negative")
    if (
        args.structure_temperature_final_bpp is not None
        and args.structure_temperature_final_bpp < 0.0
    ):
        raise ValueError("structure-temperature-final-bpp must be non-negative")
    if args.predictive_logit_soft_clip < 0.0:
        raise ValueError("predictive-logit-soft-clip must be non-negative")
    if not 0 <= args.minimum_prediction_depth <= MODEL_CONFIG.max_depth:
        raise ValueError("minimum-prediction-depth must lie within the model tree")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("min-learning-rate must lie in [0, learning-rate]")
    torch.set_num_threads(args.learner_threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_log = output / "train.jsonl"
    event_log = output / "events.jsonl"
    eval_log = output / "eval.jsonl"
    periodic_eval_log = output / "periodic_eval.jsonl"
    if train_log.exists():
        train_log.unlink()
    if event_log.exists():
        event_log.unlink()
    if eval_log.exists():
        eval_log.unlink()
    if periodic_eval_log.exists():
        periodic_eval_log.unlink()

    model_config = (
        replace(
            MODEL_CONFIG,
            sensor_features=3,
            episode_slots=0,
            prediction_slots=0,
            episode_attention=False,
            rgb_bit_output=True,
        )
        if args.objective in ("recursive_rgb_bits", "recursive_rgb_innovation_bits")
        else MODEL_CONFIG
    )
    model = QuadtreeWorldModelV2(model_config).to(device)
    if args.objective in ("recursive_rgb_bits", "recursive_rgb_innovation_bits"):
        # One context model prices structure in both sensory allocation and
        # future coding. This is the differentiable bridge by which shorter
        # future codes teach the next observation which RGB regions to refine.
        model.prediction_split = model.split_head
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    resumed_update = 0
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" not in checkpoint:
            raise ValueError("resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        resumed_update = int(checkpoint.get("update", 0))
        args._dual_price = float(checkpoint.get("dual_price", args._dual_price))
    tree_config = RGBQuadtreeConfig(
        active_depth=(
            model_config.max_depth
            if args.tree_allocation_mode == "learned_frontier"
            else args.tree_active_depth
        ),
        max_nodes=args.tree_max_nodes,
        allocation_mode=args.tree_allocation_mode,
    )
    config_payload = {
        "experiment_version": "object-permanence-quadtree-v3",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_class": "QuadtreeWorldModelV2",
        "model_source_changed": True,
        "model_config": asdict(model_config),
        "model_parameters": parameter_count(model),
        "tree_config": asdict(tree_config),
        "hardware": hardware_manifest(device),
        "resume": {
            "checkpoint": str(args.resume_from.resolve()) if args.resume_from else None,
            "parent_update": resumed_update,
        },
    }
    _write_json(output / "config.json", config_payload)

    periodic_eval_rows = []
    if args.eval_every > 0 and args.periodic_eval_families > 0:
        periodic_eval_rows = [
            prepare_example(
                args.updates + 10_000 + index,
                args.updates,
                args.seed,
                args.environment_mode,
            )
            for index in range(args.periodic_eval_families)
        ]

    wall_start = time.perf_counter()
    _append_jsonl(event_log, {"event": "run_start", "wall_seconds": 0.0})
    last_predictions = {}
    training_rows = []
    worker_args = (
        args.updates,
        args.seed,
        asdict(tree_config),
        args.history_stride,
        args.max_observation_frames,
        args.examples_per_family,
        args.environment_mode,
    )
    bundle_count = math.ceil(args.updates / args.examples_per_family)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.prefetch_workers,
        mp_context=context,
    ) as executor:
        futures = {
            bundle_index: executor.submit(
                prepare_sparse_bundle,
                bundle_index * args.examples_per_family,
                *worker_args,
            )
            for bundle_index in range(min(args.prefetch_workers, bundle_count))
        }
        update = 0
        for bundle_index in range(bundle_count):
            tick = time.perf_counter()
            prepared_bundle = futures.pop(bundle_index).result()
            bundle_wait_seconds = time.perf_counter() - tick
            next_bundle = bundle_index + args.prefetch_workers
            if next_bundle < bundle_count:
                futures[next_bundle] = executor.submit(
                    prepare_sparse_bundle,
                    next_bundle * args.examples_per_family,
                    *worker_args,
                )
            for bundle_offset, prepared in enumerate(prepared_bundle):
                update_start = time.perf_counter()
                learner_wait_seconds = bundle_wait_seconds if bundle_offset == 0 else 0.0
                current_learning_rate = scheduled_learning_rate(update, args)
                for group in optimizer.param_groups:
                    group["lr"] = current_learning_rate
                tick = time.perf_counter()
                metrics, last_predictions = train_update(model, optimizer, prepared, tree_config, args)
                step_seconds = time.perf_counter() - tick
                row = {
                    "update": update + 1,
                    "stage": prepared["stage"],
                    "family_id": prepared["family_id"],
                    "family_reuse_count": prepared["family_reuse_count"],
                    "family_unique_tree_frames": prepared["family_unique_tree_frames"],
                    "query_frame": prepared["query_frame"],
                    "uncertainty_rate": prepared["uncertainty_rate"],
                    "raw_environment_seconds": prepared["raw_environment_seconds"],
                    "raw_environment_family_seconds": prepared["raw_environment_family_seconds"],
                    "worker_tree_seconds": prepared["worker_tree_seconds"],
                    "worker_total_seconds": prepared["worker_total_seconds"],
                    "worker_payload_bytes": prepared["payload_bytes"],
                    "learner_wait_seconds": learner_wait_seconds,
                    "step_seconds": step_seconds,
                    "update_wall_seconds": time.perf_counter() - update_start + learner_wait_seconds,
                    "examples_per_second": 1.0 / max(step_seconds, 1e-9),
                    "learning_rate": current_learning_rate,
                    **{
                        key: prepared[key] for key in (
                            "causal_outcome", "causal_friction", "causal_contact_offset",
                            "causal_approach_angle", "causal_target_shape",
                            "causal_same_shape_distractors",
                        ) if key in prepared
                    },
                    **metrics,
                }
                training_rows.append(row)
                tick = time.perf_counter()
                _append_jsonl(train_log, row)
                log_write_seconds = time.perf_counter() - tick
                _append_jsonl(event_log, {
                    "event": "update_complete",
                    "update": update + 1,
                    "bundle_index": bundle_index,
                    "bundle_offset": bundle_offset,
                    "wall_seconds": time.perf_counter() - wall_start,
                    "learner_wait_seconds": learner_wait_seconds,
                    "learner_step_seconds": step_seconds,
                    "log_write_seconds": log_write_seconds,
                })
                if (update + 1) % args.report_every == 0:
                    print(json.dumps(row, sort_keys=True), flush=True)
                update += 1
                if args.checkpoint_every > 0 and update % args.checkpoint_every == 0:
                    tick = time.perf_counter()
                    checkpoint_payload = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "dual_price": args._dual_price,
                        "config": config_payload,
                    }
                    torch.save(checkpoint_payload, output / f"checkpoint-{update:06d}.pt")
                    torch.save(checkpoint_payload, output / "checkpoint-latest.pt")
                    _append_jsonl(event_log, {
                        "event": "periodic_checkpoint",
                        "update": update,
                        "wall_seconds": time.perf_counter() - wall_start,
                        "duration_seconds": time.perf_counter() - tick,
                    })
                if (
                    args.eval_every > 0
                    and periodic_eval_rows
                    and update % args.eval_every == 0
                ):
                    tick = time.perf_counter()
                    periodic_ablation, _ = evaluate_ablations(
                        model, periodic_eval_rows, tree_config, args,
                    )
                    duration = time.perf_counter() - tick
                    for condition, values in periodic_ablation.items():
                        _append_jsonl(periodic_eval_log, {
                            "update": update,
                            "condition": condition,
                            "evaluation_seconds": duration,
                            **values,
                        })
                    _append_jsonl(event_log, {
                        "event": "periodic_evaluation",
                        "update": update,
                        "wall_seconds": time.perf_counter() - wall_start,
                        "duration_seconds": duration,
                    })

    training_wall_seconds = time.perf_counter() - wall_start
    if args.skip_final_eval:
        ablations = {}
        eval_environment_seconds = 0.0
        eval_model_seconds = 0.0
        artifact_write_seconds = 0.0
    else:
        tick = time.perf_counter()
        eval_rows = [
            prepare_example(
                args.updates + i, args.updates, args.seed, args.environment_mode,
            )
            for i in range(args.eval_families)
        ]
        eval_environment_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        ablations, qualitative = evaluate_ablations(model, eval_rows, tree_config, args)
        eval_model_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        _write_json(output / "ablations.json", ablations)
        for condition, values in ablations.items():
            _append_jsonl(eval_log, {"condition": condition, **values})
        torch.save(qualitative, output / "qualitative.pt")
        artifact_write_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "update": resumed_update + args.updates, "config": config_payload,
        "dual_price": args._dual_price,
    }, output / "checkpoint.pt")
    checkpoint_seconds = time.perf_counter() - tick
    phase_names = (
        "learner_wait_seconds", "tree_seconds", "forward_seconds",
        "backward_seconds", "optimizer_seconds", "update_wall_seconds",
    )
    profile = {
        "updates": args.updates,
        "prefetch_workers": args.prefetch_workers,
        "learner_threads": args.learner_threads,
        "training_wall_seconds": training_wall_seconds,
        "end_to_end_updates_per_second": args.updates / max(training_wall_seconds, 1e-9),
        "mean_raw_environment_seconds": _mean(row["raw_environment_seconds"] for row in training_rows),
        "mean_worker_total_seconds": _mean(row["worker_total_seconds"] for row in training_rows),
        "mean_worker_payload_bytes": _mean(row["worker_payload_bytes"] for row in training_rows),
        "examples_per_physical_family": args.examples_per_family,
        "unique_physical_families": len({row["family_id"] for row in training_rows}),
        "eval_environment_seconds": eval_environment_seconds,
        "eval_model_seconds": eval_model_seconds,
        "artifact_write_seconds": artifact_write_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "mean_phases": {
            name: _mean(row[name] for row in training_rows) for name in phase_names
        },
    }
    profile["producer_stall_fraction"] = (
        profile["mean_phases"]["learner_wait_seconds"]
        / max(profile["mean_phases"]["update_wall_seconds"], 1e-9)
    )
    _write_json(output / "profile.json", profile)
    _append_jsonl(event_log, {
        "event": "evaluation_and_checkpoint_complete",
        "wall_seconds": time.perf_counter() - wall_start,
        **{key: profile[key] for key in (
            "eval_environment_seconds", "eval_model_seconds",
            "artifact_write_seconds", "checkpoint_seconds",
        )},
    })
    summary = {
        "model_class": "QuadtreeWorldModelV2",
        "model_parameters": parameter_count(model),
        "default_model_parameters": parameter_count(QuadtreeWorldModelV2()),
        "parameter_scale": parameter_count(model) / parameter_count(QuadtreeWorldModelV2()),
        "wall_seconds": time.perf_counter() - wall_start,
        "profile": profile,
        "updates": args.updates,
        "ablations": ablations,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
