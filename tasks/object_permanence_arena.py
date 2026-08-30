"""Deterministic multi-object tracking and occlusion environment.

The model receives RGB frames only.  Persistent identity, amodal masks, visible
masks, trajectories, depth order, and future targets are evaluator/training
state.  Moving disks do not interact physically in v1; opaque static rectangles
create exact, replayable occlusion without changing trajectories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from tasks.visual_domains import (
    VisualDomainConfig,
    VisualManifest,
    make_visual_manifest,
    render_rgb_layers,
)


_STREAM_NAMES = ("geometry", "motion", "occlusion", "hidden_behavior")


@dataclass(frozen=True)
class ObjectPermanenceConfig:
    image_size: int = 192
    episode_length: int = 96
    tracked_count: int = 4
    distractor_count: int = 6
    occluder_count: int = 2
    object_radius: float = 4.0
    speed_range: Tuple[float, float] = (1.5, 2.4)
    occluder_width: int = 18
    occluder_height_fraction: float = 0.72
    target_horizons: Tuple[int, ...] = (1, 4, 8, 16, 24)
    query_start: int = 20
    query_stride: int = 7
    query_end: Optional[int] = None
    center_density_sigma: float = 2.5
    ambiguous_tracked_appearance: bool = True
    target_cue_frames: int = 2
    behavior_branch_count: int = 5
    hidden_turn_speed: float = 1.1
    hidden_behavior_enabled: bool = True
    motion_layout: str = "lanes"
    relay_chain_length: int = 0
    moving_shape_count: int = 1
    pinch_contact_frame: int = 28
    pinch_close_frames: int = 8
    pinch_transport_speed: float = 1.5
    pinch_friction_levels: Tuple[float, float] = (0.25, 0.85)
    pinch_angle_bins: int = 3
    pinch_shape_count: int = 4
    pinch_same_shape_distractors: int = 1
    pinch_offset_scale: float = 0.8
    pinch_size_range: Tuple[float, float] = (0.9, 1.1)
    seed: int = 701

    def __post_init__(self) -> None:
        object.__setattr__(self, "speed_range", tuple(float(x) for x in self.speed_range))
        object.__setattr__(self, "target_horizons", tuple(self.target_horizons))
        if self.image_size < 96:
            raise ValueError("image_size must be at least 96")
        if self.episode_length < 48:
            raise ValueError("episode_length must be at least 48")
        for name in ("tracked_count", "distractor_count", "occluder_count"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1.0 <= self.object_radius < self.image_size / 12:
            raise ValueError("object_radius is outside the supported range")
        if len(self.speed_range) != 2 or self.speed_range[0] <= 0 or self.speed_range[0] > self.speed_range[1]:
            raise ValueError("speed_range must contain ordered positive values")
        if not 4 <= self.occluder_width < self.image_size // 3:
            raise ValueError("occluder_width is outside the supported range")
        if not 0.25 <= self.occluder_height_fraction <= 0.9:
            raise ValueError("occluder_height_fraction must lie in [0.25, 0.9]")
        if not self.target_horizons or any(
            not isinstance(h, int) or isinstance(h, bool) or h <= 0
            for h in self.target_horizons
        ):
            raise ValueError("target_horizons must contain positive integers")
        if tuple(sorted(set(self.target_horizons))) != self.target_horizons:
            raise ValueError("target_horizons must be unique and increasing")
        if (
            not isinstance(self.query_start, int)
            or isinstance(self.query_start, bool)
            or not isinstance(self.query_stride, int)
            or isinstance(self.query_stride, bool)
            or self.query_start < 0
            or self.query_stride < 1
        ):
            raise ValueError("query_start and query_stride are invalid")
        if self.query_start + max(self.target_horizons) >= self.episode_length:
            raise ValueError("query_start plus largest horizon must fit")
        if self.query_end is not None and (
            self.query_end <= self.query_start
            or self.query_end + max(self.target_horizons) > self.episode_length
        ):
            raise ValueError("query_end must follow query_start and leave room for horizons")
        if self.center_density_sigma <= 0:
            raise ValueError("center_density_sigma must be positive")
        if not 1 <= self.target_cue_frames < self.query_start:
            raise ValueError("target_cue_frames must lie before the first query")
        if not isinstance(self.behavior_branch_count, int) or not 2 <= self.behavior_branch_count <= 8:
            raise ValueError("behavior_branch_count must lie in [2, 8]")
        if not math.isfinite(self.hidden_turn_speed) or self.hidden_turn_speed <= 0:
            raise ValueError("hidden_turn_speed must be finite and positive")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if self.motion_layout not in ("lanes", "dense_relay", "causal_pinch"):
            raise ValueError("motion_layout must be 'lanes', 'dense_relay', or 'causal_pinch'")
        if not isinstance(self.relay_chain_length, int) or self.relay_chain_length < 0:
            raise ValueError("relay_chain_length must be a non-negative integer")
        if self.motion_layout == "dense_relay":
            if self.relay_chain_length < 2:
                raise ValueError("dense_relay requires at least two relay steps")
            if self.distractor_count < 2 * self.tracked_count * self.relay_chain_length:
                raise ValueError("dense_relay requires actual and decoy relays per step")
        if not isinstance(self.moving_shape_count, int) or not 1 <= self.moving_shape_count <= 4:
            raise ValueError("moving_shape_count must lie in [1, 4]")
        if self.motion_layout == "causal_pinch":
            if self.tracked_count != 1:
                raise ValueError("causal_pinch currently requires exactly one tracked object")
            if self.distractor_count < 3:
                raise ValueError("causal_pinch requires a distractor and two gripper fingers")
            if not self.query_start < self.pinch_contact_frame < self.episode_length - 8:
                raise ValueError("pinch_contact_frame must follow the first query and leave rollout room")
            if self.pinch_close_frames < 2:
                raise ValueError("pinch_close_frames must be at least two")
            if self.pinch_transport_speed <= 0:
                raise ValueError("pinch_transport_speed must be positive")
            if len(self.pinch_friction_levels) != 2 or not (
                0.0 < self.pinch_friction_levels[0] < self.pinch_friction_levels[1] <= 1.0
            ):
                raise ValueError("pinch_friction_levels must contain ordered values in (0, 1]")
            if self.pinch_angle_bins < 2:
                raise ValueError("pinch_angle_bins must be at least two")
            if not 1 <= self.pinch_shape_count <= 4:
                raise ValueError("pinch_shape_count must lie in [1, 4]")
            if not 0 <= self.pinch_same_shape_distractors <= self.distractor_count - 2:
                raise ValueError("pinch_same_shape_distractors exceeds available distractors")
            if self.pinch_offset_scale < 0:
                raise ValueError("pinch_offset_scale must be non-negative")
            if len(self.pinch_size_range) != 2 or not (
                0.5 <= self.pinch_size_range[0] <= self.pinch_size_range[1] <= 1.8
            ):
                raise ValueError("pinch_size_range must be ordered inside [0.5, 1.8]")

    @property
    def moving_count(self) -> int:
        return self.tracked_count + self.distractor_count

    @property
    def object_count(self) -> int:
        return self.moving_count + self.occluder_count


@dataclass(frozen=True)
class SceneObjectSpec:
    object_id: str
    role: str
    shape: str
    size: Tuple[float, float]
    depth: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_id": self.object_id,
            "role": self.role,
            "shape": self.shape,
            "size": list(self.size),
            "depth": self.depth,
        }


@dataclass(frozen=True)
class BehaviorChangeEvent:
    object_index: int
    object_id: str
    trigger_frame: int
    reveal_frame: int
    mode: str
    pre_velocity: Tuple[float, float]
    post_velocity: Tuple[float, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_index": self.object_index,
            "object_id": self.object_id,
            "trigger_frame": self.trigger_frame,
            "reveal_frame": self.reveal_frame,
            "mode": self.mode,
            "pre_velocity": list(self.pre_velocity),
            "post_velocity": list(self.post_velocity),
        }


@dataclass(frozen=True)
class RelayEvent:
    tracked_index: int
    relay_object_index: int
    frame: int
    operator_bit: int
    position: Tuple[float, float]
    alternate_relay_object_index: int
    alternate_operator_bit: int
    alternate_position: Tuple[float, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "tracked_index": self.tracked_index,
            "relay_object_index": self.relay_object_index,
            "frame": self.frame,
            "operator_bit": self.operator_bit,
            "position": list(self.position),
            "alternate_relay_object_index": self.alternate_relay_object_index,
            "alternate_operator_bit": self.alternate_operator_bit,
            "alternate_position": list(self.alternate_position),
        }


@dataclass(frozen=True)
class GraspProgram:
    target_object_index: int
    finger_indices: Tuple[int, int]
    approach_angle: float
    contact_offset: float
    initial_aperture: float
    final_aperture: float
    approach_start: int
    contact_frame: int
    close_frame: int
    transport_velocity: Tuple[float, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContactEvent:
    frame: int
    target_object_index: int
    finger_indices: Tuple[int, int]
    contact_points: Tuple[Tuple[float, float], Tuple[float, float]]
    friction: float
    center_of_mass_offset: float
    outcome: str
    post_velocity: Tuple[float, float]
    post_angular_velocity: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectPermanenceManifest:
    family_id: str
    family_index: int
    root_seed: int
    rng_seeds: Mapping[str, int]
    object_specs: Tuple[SceneObjectSpec, ...]
    depth_order: Tuple[int, ...]
    target_cue_frames: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "rng_seeds", MappingProxyType(dict(self.rng_seeds)))
        object.__setattr__(self, "object_specs", tuple(self.object_specs))
        object.__setattr__(self, "depth_order", tuple(self.depth_order))

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "family_id": self.family_id,
            "family_index": self.family_index,
            "root_seed": self.root_seed,
            "rng_seeds": dict(self.rng_seeds),
            "object_specs": [spec.to_dict() for spec in self.object_specs],
            "depth_order": list(self.depth_order),
            "target_cue_frames": self.target_cue_frames,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "ObjectPermanenceManifest":
        return cls(
            family_id=str(payload["family_id"]),
            family_index=int(payload["family_index"]),
            root_seed=int(payload["root_seed"]),
            rng_seeds={str(k): int(v) for k, v in dict(payload["rng_seeds"]).items()},
            object_specs=tuple(SceneObjectSpec(
                object_id=str(spec["object_id"]),
                role=str(spec["role"]),
                shape=str(spec["shape"]),
                size=tuple(float(v) for v in spec["size"]),
                depth=int(spec["depth"]),
            ) for spec in payload["object_specs"]),
            depth_order=tuple(int(v) for v in payload["depth_order"]),
            target_cue_frames=int(payload["target_cue_frames"]),
        )

    def __reduce__(self):
        return (ObjectPermanenceManifest.from_json_dict, (self.to_json_dict(),))


@dataclass(frozen=True)
class ObjectPermanencePhysicalEpisode:
    centers: torch.Tensor
    velocities: torch.Tensor
    alive: torch.Tensor
    amodal_masks: torch.Tensor
    object_names: Tuple[str, ...]
    manifest: ObjectPermanenceManifest
    tracked_indices: Tuple[int, ...]
    query_frames: torch.Tensor
    horizons: Tuple[int, ...]
    target_occupancy: torch.Tensor
    target_center_density: torch.Tensor
    target_marginal_occupancy: torch.Tensor
    target_marginal_center_density: torch.Tensor
    behavior_variant: int
    behavior_events: Tuple[BehaviorChangeEvent, ...]
    behavior_branch_centers: torch.Tensor
    behavior_branch_trigger_frames: torch.Tensor
    behavior_branch_reveal_frames: torch.Tensor
    object_radius: float
    center_density_sigma: float
    relay_events: Tuple[RelayEvent, ...] = ()
    angles: Optional[torch.Tensor] = None
    angular_velocities: Optional[torch.Tensor] = None
    grasp_program: Optional[GraspProgram] = None
    contact_events: Tuple[ContactEvent, ...] = ()
    surface_friction: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ObjectPermanenceEpisode:
    frames: torch.Tensor
    visible_masks: torch.Tensor
    physical_episode: ObjectPermanencePhysicalEpisode
    visual_manifest: VisualManifest
    target_visibility: torch.Tensor

    @property
    def observations(self) -> torch.Tensor:
        return self.frames

    def model_input(self) -> torch.Tensor:
        return self.frames

    @property
    def evaluator_metadata(self) -> Dict[str, object]:
        physical = self.physical_episode
        return {
            "amodal_masks": physical.amodal_masks,
            "visible_masks": self.visible_masks,
            "centers": physical.centers,
            "velocities": physical.velocities,
            "alive": physical.alive,
            "object_specs": physical.manifest.object_specs,
            "depth_order": physical.manifest.depth_order,
            "tracked_indices": physical.tracked_indices,
            "physical_manifest": physical.manifest,
            "visual_manifest": self.visual_manifest,
            "behavior_variant": physical.behavior_variant,
            "behavior_events": physical.behavior_events,
            "angles": physical.angles,
            "angular_velocities": physical.angular_velocities,
            "grasp_program": physical.grasp_program,
            "contact_events": physical.contact_events,
        }


@dataclass(frozen=True)
class ObjectPermanenceFamily:
    physical_episode: ObjectPermanencePhysicalEpisode
    episodes: Mapping[int, ObjectPermanenceEpisode]

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", dict(self.episodes))

    @property
    def family_id(self) -> str:
        return self.physical_episode.manifest.family_id


@dataclass(frozen=True)
class ObjectPermanenceTrainingExample:
    observations: torch.Tensor
    future_rgb: torch.Tensor
    query_frame: int
    horizons: Tuple[int, ...]
    target_occupancy: torch.Tensor
    target_center_density: torch.Tensor
    family_id: str
    visual_family_id: str
    evaluator_target_visibility: torch.Tensor
    evaluator_uncertainty_mask: torch.Tensor

    def model_input(self) -> torch.Tensor:
        return self.observations


def _derived_seed(root_seed: int, family_index: int, stream_name: str) -> int:
    payload = f"object-permanence-v1|{root_seed}|{family_index}|{stream_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 2) + 1


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def _reflect(position: float, velocity: float, step: int, low: float, high: float) -> Tuple[float, float]:
    span = high - low
    phase = (position - low) + velocity * step
    folded = phase % (2.0 * span)
    if folded <= span:
        return low + folded, velocity
    return high - (folded - span), -velocity


def _irregular_bounce(
    start: Tuple[float, float],
    velocity: Tuple[float, float],
    frames: int,
    low: float,
    high: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic nonperiodic billiards with angled rebounds and sparse kicks."""

    positions = torch.zeros((frames, 2), dtype=torch.float32)
    velocities = torch.zeros_like(positions)
    x, y = start
    vx, vy = velocity
    kick_period = int(torch.randint(11, 24, (), generator=generator))
    kick_phase = int(torch.randint(0, kick_period, (), generator=generator))
    for frame in range(frames):
        positions[frame] = torch.tensor((x, y))
        velocities[frame] = torch.tensor((vx, vy))
        if frame > 0 and frame % kick_period == kick_phase:
            angle = _uniform(generator, -0.48, 0.48)
            cosine, sine = math.cos(angle), math.sin(angle)
            vx, vy = cosine * vx - sine * vy, sine * vx + cosine * vy
        next_x, next_y = x + vx, y + vy
        hit_x = next_x < low or next_x > high
        hit_y = next_y < low or next_y > high
        if hit_x:
            vx = -vx
        if hit_y:
            vy = -vy
        if hit_x or hit_y:
            angle = _uniform(generator, -0.30, 0.30)
            cosine, sine = math.cos(angle), math.sin(angle)
            vx, vy = cosine * vx - sine * vy, sine * vx + cosine * vy
        x = min(high, max(low, x + vx))
        y = min(high, max(low, y + vy))
    return positions, velocities


def _disk_masks(centers: torch.Tensor, radius: float, side: int) -> torch.Tensor:
    values = torch.arange(side, dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    dx = xx[None, None] - centers[:, :, 0, None, None]
    dy = yy[None, None] - centers[:, :, 1, None, None]
    return dx.square() + dy.square() <= radius * radius


def _moving_masks(
    centers: torch.Tensor,
    radius: float,
    side: int,
    shape_count: int,
) -> torch.Tensor:
    """Render a balanced disk/square/diamond/triangle shape mixture."""

    if shape_count == 1:
        return _disk_masks(centers, radius, side)
    values = torch.arange(side, dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    rows = []
    for object_index in range(centers.shape[1]):
        dx = xx[None] - centers[:, object_index, 0, None, None]
        dy = yy[None] - centers[:, object_index, 1, None, None]
        shape = object_index % shape_count
        if shape == 0:
            mask = dx.square() + dy.square() <= radius * radius
        elif shape == 1:
            mask = torch.maximum(dx.abs(), dy.abs()) <= radius
        elif shape == 2:
            mask = dx.abs() + dy.abs() <= radius * 1.35
        else:
            normalized_y = dy / radius
            half_width = ((normalized_y + 1.0) * 0.5 * radius).clamp(0.0, radius)
            mask = (normalized_y >= -1.0) & (normalized_y <= 1.0) & (dx.abs() <= half_width)
        rows.append(mask)
    return torch.stack(rows, dim=1)


def _density(center: torch.Tensor, side: int, sigma: float) -> torch.Tensor:
    values = torch.arange(side, dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    density = torch.exp(
        -((xx - center[0]).square() + (yy - center[1]).square()) / (2.0 * sigma * sigma)
    )
    return density / density.sum().clamp_min(1e-12)


def _make_manifest(config: ObjectPermanenceConfig, family_index: int) -> ObjectPermanenceManifest:
    seeds = {
        name: _derived_seed(config.seed, family_index, name)
        for name in _STREAM_NAMES
    }
    specs = []
    shapes = ("disk", "square", "diamond", "triangle")
    pinch_shapes = ("oriented_rectangle", "capsule", "handled_body", "notched_body")
    for index in range(config.tracked_count):
        shape = (
            pinch_shapes[family_index % config.pinch_shape_count]
            if config.motion_layout == "causal_pinch"
            else shapes[index % config.moving_shape_count]
        )
        specs.append(SceneObjectSpec(
            f"tracked_{index}", "tracked", shape,
            ((config.object_radius * 2.2, config.object_radius * 1.25)
             if config.motion_layout == "causal_pinch"
             else (config.object_radius, config.object_radius)), index,
        ))
    for index in range(config.distractor_count):
        moving_index = config.tracked_count + index
        is_finger = (
            config.motion_layout == "causal_pinch"
            and moving_index >= config.moving_count - 2
        )
        specs.append(SceneObjectSpec(
            (f"gripper_finger_{moving_index - (config.moving_count - 2)}"
             if is_finger else f"distractor_{index}"),
            "gripper" if is_finger else "distractor",
            "finger" if is_finger else shapes[moving_index % config.moving_shape_count],
            (config.object_radius, config.object_radius), config.tracked_count + index,
        ))
    height = config.image_size * config.occluder_height_fraction
    for index in range(config.occluder_count):
        specs.append(SceneObjectSpec(
            f"occluder_{index}", "occluder", "rectangle",
            (float(config.occluder_width), float(height)), config.moving_count + index,
        ))
    fingerprint = json.dumps(
        {"config": asdict(config), "family_index": family_index},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    family_id = "opa-{}".format(hashlib.sha256(fingerprint).hexdigest()[:16])
    return ObjectPermanenceManifest(
        family_id=family_id,
        family_index=family_index,
        root_seed=config.seed,
        rng_seeds=seeds,
        object_specs=tuple(specs),
        depth_order=tuple(range(config.object_count)),
        target_cue_frames=config.target_cue_frames,
    )


def _oriented_graspable_mask(
    center: torch.Tensor,
    angle: float,
    half_width: float,
    half_height: float,
    side: int,
    shape: str,
) -> torch.Tensor:
    """Rasterize one rigid, oriented grasp object with optional part geometry."""

    values = torch.arange(side, dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    dx, dy = xx - center[0], yy - center[1]
    cosine, sine = math.cos(angle), math.sin(angle)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    rectangle = (local_x.abs() <= half_width) & (local_y.abs() <= half_height)
    if shape == "capsule":
        segment_x = local_x.clamp(-half_width + half_height, half_width - half_height)
        return (local_x - segment_x).square() + local_y.square() <= half_height**2
    if shape == "handled_body":
        handle_center = half_width + half_height * 0.65
        handle_radius = half_height * 0.72
        handle = (local_x - handle_center).square() + local_y.square() <= handle_radius**2
        handle_hole = (local_x - handle_center).square() + local_y.square() < (handle_radius * 0.45)**2
        return rectangle | (handle & ~handle_hole)
    if shape == "notched_body":
        notch = (
            (local_x > half_width * 0.35)
            & (local_y.abs() < half_height * 0.45)
        )
        return rectangle & ~notch
    return rectangle


def _generate_causal_pinch_physics(
    config: ObjectPermanenceConfig,
    family_index: int,
) -> ObjectPermanencePhysicalEpisode:
    """Generate a deterministic rigid pinch with attach, slip, or rotation."""

    manifest = _make_manifest(config, family_index)
    geometry_rng = _generator(manifest.rng_seeds["geometry"])
    motion_rng = _generator(manifest.rng_seeds["motion"])
    frames, count, side = config.episode_length, config.object_count, config.image_size
    centers = torch.zeros((frames, count, 2), dtype=torch.float32)
    velocities = torch.zeros_like(centers)
    angles = torch.zeros((frames, count), dtype=torch.float32)
    angular_velocities = torch.zeros_like(angles)
    alive = torch.ones((frames, count), dtype=torch.bool)
    masks = torch.zeros((frames, count, side, side), dtype=torch.bool)

    target_index = 0
    distractor_indices = tuple(range(config.tracked_count, config.moving_count - 2))
    finger_indices = (config.moving_count - 2, config.moving_count - 1)
    target_center = torch.tensor((
        side * 0.50 + _uniform(geometry_rng, -8.0, 8.0),
        side * 0.56 + _uniform(geometry_rng, -6.0, 6.0),
    ))
    approach_angle = (
        (family_index % config.pinch_angle_bins) * math.pi / config.pinch_angle_bins
    )
    object_angle = approach_angle + (math.pi / 2.0 if family_index % 2 else math.pi / 5.0)
    friction = config.pinch_friction_levels[family_index % 2]
    offset_levels = (
        0.0,
        config.object_radius * config.pinch_offset_scale,
        -config.object_radius * config.pinch_offset_scale,
    )
    contact_offset = offset_levels[(family_index // 2) % len(offset_levels)]
    size_phase = ((family_index * 7) % 11) / 10.0
    size_scale = (
        config.pinch_size_range[0]
        + size_phase * (config.pinch_size_range[1] - config.pinch_size_range[0])
    )
    half_width = config.object_radius * 2.2 * size_scale
    half_height = config.object_radius * 1.25 * size_scale
    initial_aperture = half_height * 5.0
    final_aperture = half_height * 1.35
    contact_aperture = half_height * 2.0 + config.object_radius * 0.9
    contact_frame = config.pinch_contact_frame
    close_frame = contact_frame + config.pinch_close_frames
    normal = torch.tensor((math.cos(approach_angle), math.sin(approach_angle)))
    tangent = torch.tensor((-math.sin(approach_angle), math.cos(approach_angle)))
    grasp_center = target_center + tangent * contact_offset
    transport_velocity = tangent * (-config.pinch_transport_speed)

    if friction >= 0.5 and abs(contact_offset) < config.object_radius * 0.4:
        outcome = "attach"
        object_velocity = transport_velocity
        post_angular_velocity = 0.0
    elif friction >= 0.5:
        outcome = "rotate"
        object_velocity = transport_velocity * 0.65
        post_angular_velocity = math.copysign(0.10, contact_offset)
    else:
        outcome = "slip"
        object_velocity = transport_velocity * 0.20 + normal * math.copysign(0.75, contact_offset or 1.0)
        post_angular_velocity = math.copysign(0.025, contact_offset or 1.0)

    for frame in range(frames):
        post_steps = max(0, frame - close_frame)
        if frame <= close_frame:
            center = target_center
        else:
            center = target_center + object_velocity * post_steps
        centers[frame, target_index] = center
        velocities[frame, target_index] = object_velocity if frame > close_frame else 0.0
        angle = object_angle + post_angular_velocity * post_steps
        angles[frame, target_index] = angle
        angular_velocities[frame, target_index] = post_angular_velocity if frame > close_frame else 0.0
        masks[frame, target_index] = _oriented_graspable_mask(
            center, angle, half_width, half_height, side,
            manifest.object_specs[target_index].shape,
        )

        if frame <= contact_frame:
            alpha = max(0.0, frame / max(1, contact_frame))
            aperture = initial_aperture + alpha * (contact_aperture - initial_aperture)
        elif frame <= close_frame:
            alpha = (frame - contact_frame) / max(1, config.pinch_close_frames)
            aperture = contact_aperture + alpha * (final_aperture - contact_aperture)
        else:
            aperture = final_aperture
        transported_grasp = grasp_center + transport_velocity * post_steps
        for sign, finger_index in zip((-1.0, 1.0), finger_indices):
            finger_center = transported_grasp + normal * (sign * aperture / 2.0)
            centers[frame, finger_index] = finger_center
            if frame:
                velocities[frame, finger_index] = finger_center - centers[frame - 1, finger_index]
            masks[frame, finger_index] = _disk_masks(
                finger_center[None, None], config.object_radius * 0.8, side,
            )[0, 0]

    low, high = config.object_radius + 2.0, side - config.object_radius - 3.0
    surface_friction = torch.full((count,), float("nan"), dtype=torch.float32)
    surface_friction[target_index] = friction
    target_shape = manifest.object_specs[target_index].shape
    for distractor_rank, distractor_index in enumerate(distractor_indices):
        start = (_uniform(geometry_rng, low, high), _uniform(geometry_rng, low, high))
        angle = _uniform(motion_rng, -math.pi, math.pi)
        speed = _uniform(motion_rng, *config.speed_range)
        trajectory, trajectory_velocity = _irregular_bounce(
            start, (speed * math.cos(angle), speed * math.sin(angle)),
            frames, low, high, motion_rng,
        )
        centers[:, distractor_index] = trajectory
        velocities[:, distractor_index] = trajectory_velocity
        decoy_friction = config.pinch_friction_levels[(family_index + distractor_rank + 1) % 2]
        surface_friction[distractor_index] = decoy_friction
        if distractor_rank < config.pinch_same_shape_distractors:
            decoy_angle = object_angle + (distractor_rank + 1) * math.pi / 7.0
            for frame in range(frames):
                angles[frame, distractor_index] = decoy_angle + frame * 0.012 * (-1 if distractor_rank % 2 else 1)
                angular_velocities[frame, distractor_index] = 0.012 * (-1 if distractor_rank % 2 else 1)
                masks[frame, distractor_index] = _oriented_graspable_mask(
                    trajectory[frame], float(angles[frame, distractor_index]),
                    half_width, half_height, side, target_shape,
                )
        else:
            masks[:, distractor_index] = _moving_masks(
                trajectory[:, None], config.object_radius, side, config.moving_shape_count,
            )[:, 0]

    # Keep the inherited occlusion primitive as a peripheral nuisance in this
    # first grasp rung. Contact geometry itself must remain visible; later
    # curricula can deliberately move an occluder across the interaction.
    occluder_height = round(side * config.occluder_height_fraction)
    occluders = tuple(
        (
            round(side * (0.82 + 0.05 * index)), side // 2,
            side // 2 - occluder_height // 2,
            side // 2 - occluder_height // 2 + occluder_height,
            round(side * (0.82 + 0.05 * index)) - config.occluder_width // 2,
            round(side * (0.82 + 0.05 * index)) - config.occluder_width // 2 + config.occluder_width,
        )
        for index in range(config.occluder_count)
    )
    for index, (center_x, center_y, top, bottom, x0, x1) in enumerate(occluders):
        object_index = config.moving_count + index
        centers[:, object_index] = torch.tensor((center_x, center_y), dtype=torch.float32)
        masks[:, object_index, top:bottom, x0:x1] = True

    query_frames = torch.arange(
        config.query_start,
        config.query_end or config.episode_length - max(config.target_horizons),
        config.query_stride,
        dtype=torch.int64,
    )
    tracked = (target_index,)
    occupancy = torch.stack([
        torch.stack([masks[query + horizon, list(tracked)] for horizon in config.target_horizons])
        for query in query_frames.tolist()
    ])
    density = torch.stack([
        torch.stack([torch.stack([
            _density(centers[query + horizon, index], side, config.center_density_sigma)
            for index in tracked
        ]) for horizon in config.target_horizons])
        for query in query_frames.tolist()
    ])
    program = GraspProgram(
        target_object_index=target_index,
        finger_indices=finger_indices,
        approach_angle=approach_angle,
        contact_offset=contact_offset,
        initial_aperture=initial_aperture,
        final_aperture=final_aperture,
        approach_start=0,
        contact_frame=contact_frame,
        close_frame=close_frame,
        transport_velocity=tuple(float(v) for v in transport_velocity),
    )
    contact_points = tuple(
        tuple(float(v) for v in grasp_center + normal * (sign * half_height))
        for sign in (-1.0, 1.0)
    )
    contact = ContactEvent(
        frame=contact_frame,
        target_object_index=target_index,
        finger_indices=finger_indices,
        contact_points=contact_points,
        friction=friction,
        center_of_mass_offset=contact_offset,
        outcome=outcome,
        post_velocity=tuple(float(v) for v in object_velocity),
        post_angular_velocity=post_angular_velocity,
    )
    behavior = BehaviorChangeEvent(
        object_index=target_index,
        object_id=manifest.object_specs[target_index].object_id,
        trigger_frame=close_frame,
        reveal_frame=close_frame + 1,
        mode=f"pinch_{outcome}",
        pre_velocity=(0.0, 0.0),
        post_velocity=tuple(float(v) for v in object_velocity),
    )
    return ObjectPermanencePhysicalEpisode(
        centers=centers, velocities=velocities, alive=alive, amodal_masks=masks,
        object_names=tuple(spec.object_id for spec in manifest.object_specs),
        manifest=manifest, tracked_indices=tracked, query_frames=query_frames,
        horizons=config.target_horizons, target_occupancy=occupancy,
        target_center_density=density, target_marginal_occupancy=occupancy.float(),
        target_marginal_center_density=density, behavior_variant=0,
        behavior_events=(behavior,),
        behavior_branch_centers=centers[None, :, list(tracked)],
        behavior_branch_trigger_frames=torch.tensor([[close_frame]], dtype=torch.int64),
        behavior_branch_reveal_frames=torch.tensor([[close_frame + 1]], dtype=torch.int64),
        object_radius=config.object_radius, center_density_sigma=config.center_density_sigma,
        angles=angles, angular_velocities=angular_velocities,
        grasp_program=program, contact_events=(contact,),
        surface_friction=surface_friction,
    )


def _occluder_geometry(
    config: ObjectPermanenceConfig,
    manifest: ObjectPermanenceManifest,
) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
    rng = _generator(manifest.rng_seeds["occlusion"])
    height = round(config.image_size * config.occluder_height_fraction)
    result = []
    for index in range(config.occluder_count):
        fraction = (index + 1) / (config.occluder_count + 1)
        x_jitter = _uniform(rng, -0.035, 0.035) * config.image_size
        y_jitter = round(_uniform(rng, -0.035, 0.035) * config.image_size)
        center_x = round(fraction * (config.image_size - 1) + x_jitter)
        center_y = config.image_size // 2 + y_jitter
        top = max(0, min(config.image_size - height, center_y - height // 2))
        x0 = max(0, center_x - config.occluder_width // 2)
        x1 = min(config.image_size, x0 + config.occluder_width)
        result.append((center_x, center_y, top, top + height, x0, x1))
    return tuple(result)


def _post_behavior_velocity(
    mode: str,
    vx: float,
    config: ObjectPermanenceConfig,
) -> Tuple[float, float]:
    if mode == "speed_up":
        return vx * 1.45, 0.0
    if mode == "slow_down":
        return vx * 0.60, 0.0
    if mode == "turn_up":
        return vx * 0.85, -config.hidden_turn_speed
    if mode == "turn_down":
        return vx * 0.85, config.hidden_turn_speed
    if mode == "reverse":
        return -vx * 0.85, 0.0
    return vx, 0.0


def _dense_relay_waypoints(
    config: ObjectPermanenceConfig,
    generator: torch.Generator,
    tracked_slot: int,
    operator_bits: Sequence[int],
    start: Tuple[float, float],
) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[Tuple[float, float], ...]]:
    """Sample irregular left/right turns with nonperiodic lengths and angles."""

    margin = max(12.0, config.object_radius + 5.0)
    high = config.image_size - margin

    def fold(value: float) -> float:
        span = high - margin
        phase = (value - margin) % (2.0 * span)
        return margin + (phase if phase <= span else 2.0 * span - phase)

    def project(origin: Tuple[float, float], angle: float, distance: float) -> Tuple[float, float]:
        for _ in range(12):
            candidate = (
                origin[0] + distance * math.cos(angle),
                origin[1] + distance * math.sin(angle),
            )
            if margin <= candidate[0] <= high and margin <= candidate[1] <= high:
                return candidate
            distance *= 0.72
        return (
            min(high, max(margin, candidate[0])),
            min(high, max(margin, candidate[1])),
        )

    route = []
    alternates = []
    previous = start
    initial_angle = _uniform(generator, -math.pi, math.pi) + tracked_slot * math.pi
    initial_distance = _uniform(generator, config.image_size * 0.22, config.image_size * 0.42)
    current = (
        fold(start[0] + initial_distance * math.cos(initial_angle)),
        fold(start[1] + initial_distance * math.sin(initial_angle)),
    )
    for step in range(config.relay_chain_length):
        route.append(current)
        incoming = math.atan2(current[1] - previous[1], current[0] - previous[0])
        turn = _uniform(generator, math.radians(38), math.radians(148))
        left_distance = _uniform(generator, config.image_size * 0.18, config.image_size * 0.46)
        right_distance = _uniform(generator, config.image_size * 0.18, config.image_size * 0.46)
        left = project(current, incoming + turn, left_distance)
        right = project(current, incoming - turn, right_distance)
        selected, alternate = (left, right) if operator_bits[step] else (right, left)
        alternates.append(alternate)
        previous, current = current, selected
    return tuple(route), tuple(alternates)


def _generate_dense_relay_physics(
    config: ObjectPermanenceConfig,
    family_index: int,
) -> ObjectPermanencePhysicalEpisode:
    """Generate ordered local relay chains embedded in dense isotropic clutter."""

    manifest = _make_manifest(config, family_index)
    motion_rng = _generator(manifest.rng_seeds["motion"])
    geometry_rng = _generator(manifest.rng_seeds["geometry"])
    frames = config.episode_length
    count = config.object_count
    centers = torch.zeros((frames, count, 2), dtype=torch.float32)
    velocities = torch.zeros_like(centers)
    alive = torch.ones((frames, count), dtype=torch.bool)
    low = config.object_radius + 1.0
    high = config.image_size - config.object_radius - 2.0
    segment = max(3, (frames - 1) // (config.relay_chain_length + 1))
    relay_events = []
    occupied_relays = set()

    for slot in range(config.tracked_count):
        operator_bits = tuple(
            int(torch.randint(0, 2, (), generator=motion_rng))
            for _ in range(config.relay_chain_length)
        )
        start = torch.tensor((
            config.image_size / 2.0 + _uniform(geometry_rng, -8.0, 8.0),
            config.image_size / 2.0 + _uniform(geometry_rng, -8.0, 8.0),
        ), dtype=torch.float32)
        start_point = tuple(float(v) for v in start)
        waypoints, alternate_waypoints = _dense_relay_waypoints(
            config, geometry_rng, slot, operator_bits, start_point,
        )
        route = (start_point,) + waypoints
        for step, waypoint in enumerate(waypoints):
            pair_offset = 2 * (slot * config.relay_chain_length + step)
            relay_index = config.tracked_count + pair_offset
            alternate_index = relay_index + 1
            occupied_relays.update((relay_index, alternate_index))
            point = torch.tensor(waypoint, dtype=torch.float32)
            centers[:, relay_index] = point
            alternate = alternate_waypoints[step]
            centers[:, alternate_index] = torch.tensor(alternate, dtype=torch.float32)
            operator_bit = operator_bits[step]
            alternate_operator_bit = int(torch.randint(0, 2, (), generator=motion_rng))
            event_frame = min(frames - 1, (step + 1) * segment)
            relay_events.append(RelayEvent(
                tracked_index=slot,
                relay_object_index=relay_index,
                frame=event_frame,
                operator_bit=operator_bit,
                position=waypoint,
                alternate_relay_object_index=alternate_index,
                alternate_operator_bit=alternate_operator_bit,
                alternate_position=alternate,
            ))
        for step in range(config.relay_chain_length):
            begin = step * segment
            end = min(frames - 1, (step + 1) * segment)
            source = torch.tensor(route[step], dtype=torch.float32)
            destination = torch.tensor(route[step + 1], dtype=torch.float32)
            velocity = (destination - source) / max(1, end - begin)
            for frame in range(begin, end + 1):
                alpha = (frame - begin) / max(1, end - begin)
                centers[frame, slot] = source.lerp(destination, alpha)
                velocities[frame, slot] = velocity
        final_frame = min(frames - 1, config.relay_chain_length * segment)
        centers[final_frame:, slot] = centers[final_frame, slot]

    # Remaining objects move isotropically through the whole field. Their
    # statistics overlap the tracked paths, preventing a stationary/motion cue.
    for index in range(config.tracked_count, config.moving_count):
        if index in occupied_relays:
            continue
        start_x = _uniform(geometry_rng, low, high)
        start_y = _uniform(geometry_rng, low, high)
        angle = _uniform(motion_rng, -math.pi, math.pi)
        speed = _uniform(motion_rng, *config.speed_range)
        vx, vy = speed * math.cos(angle), speed * math.sin(angle)
        trajectory, trajectory_velocity = _irregular_bounce(
            (start_x, start_y), (vx, vy), frames, low, high, motion_rng,
        )
        centers[:, index] = trajectory
        velocities[:, index] = trajectory_velocity

    masks = torch.zeros((frames, count, config.image_size, config.image_size), dtype=torch.bool)
    masks[:, :config.moving_count] = _moving_masks(
        centers[:, :config.moving_count], config.object_radius,
        config.image_size, config.moving_shape_count,
    )
    occluders = _occluder_geometry(config, manifest)
    for index, (center_x, center_y, top, bottom, x0, x1) in enumerate(occluders):
        object_index = config.moving_count + index
        centers[:, object_index] = torch.tensor((center_x, center_y), dtype=torch.float32)
        masks[:, object_index, top:bottom, x0:x1] = True

    query_frames = torch.arange(
        config.query_start,
        config.query_end or config.episode_length - max(config.target_horizons),
        config.query_stride,
        dtype=torch.int64,
    )
    tracked = tuple(range(config.tracked_count))
    occupancy = torch.stack([
        torch.stack([masks[query + horizon, list(tracked)] for horizon in config.target_horizons])
        for query in query_frames.tolist()
    ])
    density = torch.stack([
        torch.stack([
            torch.stack([
                _density(centers[query + horizon, index], config.image_size, config.center_density_sigma)
                for index in tracked
            ])
            for horizon in config.target_horizons
        ])
        for query in query_frames.tolist()
    ])
    behavior_events = tuple(BehaviorChangeEvent(
        object_index=slot,
        object_id=manifest.object_specs[slot].object_id,
        trigger_frame=frames - 1,
        reveal_frame=frames - 1,
        mode="relay_chain",
        pre_velocity=(0.0, 0.0),
        post_velocity=(0.0, 0.0),
    ) for slot in tracked)
    return ObjectPermanencePhysicalEpisode(
        centers=centers,
        velocities=velocities,
        alive=alive,
        amodal_masks=masks,
        object_names=tuple(spec.object_id for spec in manifest.object_specs),
        manifest=manifest,
        tracked_indices=tracked,
        query_frames=query_frames,
        horizons=config.target_horizons,
        target_occupancy=occupancy,
        target_center_density=density,
        target_marginal_occupancy=occupancy.float(),
        target_marginal_center_density=density,
        behavior_variant=0,
        behavior_events=behavior_events,
        behavior_branch_centers=centers[None, :, list(tracked)],
        behavior_branch_trigger_frames=torch.full(
            (1, config.tracked_count), frames - 1, dtype=torch.int64,
        ),
        behavior_branch_reveal_frames=torch.full(
            (1, config.tracked_count), frames - 1, dtype=torch.int64,
        ),
        object_radius=config.object_radius,
        center_density_sigma=config.center_density_sigma,
        relay_events=tuple(relay_events),
    )


def _generate_single_physics(
    config: ObjectPermanenceConfig,
    family_index: int,
    behavior_variant: int,
) -> ObjectPermanencePhysicalEpisode:
    if config.motion_layout == "dense_relay":
        return _generate_dense_relay_physics(config, family_index)
    if config.motion_layout == "causal_pinch":
        return _generate_causal_pinch_physics(config, family_index)
    manifest = _make_manifest(config, family_index)
    motion_rng = _generator(manifest.rng_seeds["motion"])
    geometry_rng = _generator(manifest.rng_seeds["geometry"])
    hidden_rng = _generator(_derived_seed(
        manifest.rng_seeds["hidden_behavior"], behavior_variant, "variant"
    ))
    occluders = _occluder_geometry(config, manifest)
    frames = config.episode_length
    count = config.object_count
    centers = torch.zeros((frames, count, 2), dtype=torch.float32)
    velocities = torch.zeros_like(centers)
    alive = torch.ones((frames, count), dtype=torch.bool)
    low = config.object_radius + 1.0
    high = config.image_size - config.object_radius - 2.0
    pending_events = []
    modes = ("coast", "speed_up", "slow_down", "turn_up", "turn_down", "reverse")

    for index in range(config.moving_count):
        if index < config.tracked_count:
            direction = 1.0 if index % 2 == 0 else -1.0
            start_x = low + 2.0 if direction > 0 else high - 2.0
            pair_count = math.ceil(config.tracked_count / 2)
            pair_index = index // 2
            lane_fraction = 0.30 + 0.40 * (pair_index + 0.5) / pair_count
            start_y = lane_fraction * config.image_size
            speed = _uniform(motion_rng, *config.speed_range)
            vx = direction * speed
            target_x = min(item[0] for item in occluders) if direction > 0 else max(item[0] for item in occluders)
            trigger = max(1, min(frames - 2, round((target_x - start_x) / vx)))
            mode_offset = int(torch.randint(0, len(modes), (), generator=hidden_rng))
            mode = (
                modes[(mode_offset + index + behavior_variant) % len(modes)]
                if config.hidden_behavior_enabled else "coast"
            )
            trigger_x, trigger_vx = _reflect(start_x, vx, trigger, low, high)
            post_vx, post_vy = _post_behavior_velocity(mode, trigger_vx, config)
            for frame in range(frames):
                if frame <= trigger:
                    x, frame_vx = _reflect(start_x, vx, frame, low, high)
                    y, frame_vy = start_y, 0.0
                else:
                    step = frame - trigger
                    x, frame_vx = _reflect(trigger_x, post_vx, step, low, high)
                    y, frame_vy = _reflect(start_y, post_vy, step, low, high)
                centers[frame, index] = torch.tensor((x, y))
                velocities[frame, index] = torch.tensor((frame_vx, frame_vy))
            pending_events.append((index, trigger, mode, (trigger_vx, 0.0), (post_vx, post_vy)))
        else:
            start_x = _uniform(geometry_rng, low, high)
            start_y = _uniform(geometry_rng, low, high)
            angle = _uniform(motion_rng, -math.pi, math.pi)
            speed = _uniform(motion_rng, *config.speed_range)
            vx, vy = speed * math.cos(angle), speed * math.sin(angle)
            for frame in range(frames):
                x, frame_vx = _reflect(start_x, vx, frame, low, high)
                y, frame_vy = _reflect(start_y, vy, frame, low, high)
                centers[frame, index] = torch.tensor((x, y))
                velocities[frame, index] = torch.tensor((frame_vx, frame_vy))

    masks = torch.zeros((frames, count, config.image_size, config.image_size), dtype=torch.bool)
    masks[:, :config.moving_count] = _moving_masks(
        centers[:, :config.moving_count], config.object_radius,
        config.image_size, config.moving_shape_count,
    )
    for index, (center_x, center_y, top, bottom, x0, x1) in enumerate(occluders):
        object_index = config.moving_count + index
        centers[:, object_index] = torch.tensor((center_x, center_y), dtype=torch.float32)
        masks[:, object_index, top:bottom, x0:x1] = True

    occluder_union = masks[:, config.moving_count:].any(dim=1)
    events = []
    for index, trigger, mode, pre_velocity, post_velocity in pending_events:
        visible_pixels = masks[:, index] & ~occluder_union
        reveal = frames - 1
        for frame in range(trigger + 1, frames):
            if bool(visible_pixels[frame].any()):
                reveal = frame
                break
        events.append(BehaviorChangeEvent(
            object_index=index,
            object_id=manifest.object_specs[index].object_id,
            trigger_frame=trigger,
            reveal_frame=reveal,
            mode=mode,
            pre_velocity=tuple(float(v) for v in pre_velocity),
            post_velocity=tuple(float(v) for v in post_velocity),
        ))

    query_frames = torch.arange(
        config.query_start,
        config.query_end or config.episode_length - max(config.target_horizons),
        config.query_stride,
        dtype=torch.int64,
    )
    tracked = tuple(range(config.tracked_count))
    occupancy_rows = []
    density_rows = []
    for query in query_frames.tolist():
        occupancy_rows.append(torch.stack([
            masks[query + horizon, list(tracked)]
            for horizon in config.target_horizons
        ]))
        density_rows.append(torch.stack([
            torch.stack([
                _density(centers[query + horizon, index], config.image_size, config.center_density_sigma)
                for index in tracked
            ])
            for horizon in config.target_horizons
        ]))
    occupancy = torch.stack(occupancy_rows)
    density = torch.stack(density_rows)
    return ObjectPermanencePhysicalEpisode(
        centers=centers,
        velocities=velocities,
        alive=alive,
        amodal_masks=masks,
        object_names=tuple(spec.object_id for spec in manifest.object_specs),
        manifest=manifest,
        tracked_indices=tracked,
        query_frames=query_frames,
        horizons=config.target_horizons,
        target_occupancy=occupancy,
        target_center_density=density,
        target_marginal_occupancy=occupancy.float(),
        target_marginal_center_density=density,
        behavior_variant=behavior_variant,
        behavior_events=tuple(events),
        behavior_branch_centers=centers[None, :, list(tracked)],
        behavior_branch_trigger_frames=torch.tensor(
            [[event.trigger_frame for event in events]], dtype=torch.int64
        ),
        behavior_branch_reveal_frames=torch.tensor(
            [[event.reveal_frame for event in events]], dtype=torch.int64
        ),
        object_radius=config.object_radius,
        center_density_sigma=config.center_density_sigma,
    )


def generate_object_permanence_physics(
    config: ObjectPermanenceConfig,
    family_index: int,
) -> ObjectPermanencePhysicalEpisode:
    """Generate one realized branch plus hidden-behavior marginal targets."""

    if family_index < 0:
        raise ValueError("family_index must be non-negative")
    if config.motion_layout == "dense_relay":
        return _generate_dense_relay_physics(config, family_index)
    if config.motion_layout == "causal_pinch":
        return _generate_causal_pinch_physics(config, family_index)
    realized = _generate_single_physics(config, family_index, 0)
    occupancy_sum = realized.target_occupancy.float().clone()
    density_sum = realized.target_center_density.clone()
    branch_centers = [realized.centers[:, list(realized.tracked_indices)]]
    branch_triggers = [[event.trigger_frame for event in realized.behavior_events]]
    branch_reveals = [[event.reveal_frame for event in realized.behavior_events]]
    for variant in range(1, config.behavior_branch_count):
        branch = _generate_single_physics(config, family_index, variant)
        occupancy_sum += branch.target_occupancy.float()
        density_sum += branch.target_center_density
        branch_centers.append(branch.centers[:, list(branch.tracked_indices)])
        branch_triggers.append([event.trigger_frame for event in branch.behavior_events])
        branch_reveals.append([event.reveal_frame for event in branch.behavior_events])
    return replace(
        realized,
        target_marginal_occupancy=occupancy_sum / config.behavior_branch_count,
        target_marginal_center_density=density_sum / config.behavior_branch_count,
        behavior_branch_centers=torch.stack(branch_centers),
        behavior_branch_trigger_frames=torch.tensor(branch_triggers, dtype=torch.int64),
        behavior_branch_reveal_frames=torch.tensor(branch_reveals, dtype=torch.int64),
    )


def _render_episode(
    physical: ObjectPermanencePhysicalEpisode,
    visual_manifest: VisualManifest,
) -> ObjectPermanenceEpisode:
    frames, visible = render_rgb_layers(
        physical.amodal_masks,
        physical.centers,
        physical.alive,
        physical.object_names,
        physical.manifest.family_id,
        visual_manifest,
        depth_order=physical.manifest.depth_order,
        allow_occlusion=True,
    )
    if physical.grasp_program is not None and physical.contact_events:
        # A compact, persistent surface code makes the material coefficient
        # observable without changing object outline. One bar means slippery;
        # three bars mean high friction. The brief contact flash appears only
        # at actual contact and therefore cannot leak the later outcome.
        event = physical.contact_events[0]
        marked_objects = torch.nonzero(
            torch.isfinite(physical.surface_friction), as_tuple=False,
        ).flatten().tolist() if physical.surface_friction is not None else [event.target_object_index]
        for object_index in marked_objects:
            object_friction = float(physical.surface_friction[object_index]) if physical.surface_friction is not None else event.friction
            bar_count = 3 if object_friction >= 0.5 else 1
            for frame in range(frames.shape[0]):
                center = physical.centers[frame, object_index]
                angle = float(physical.angles[frame, object_index]) if physical.angles is not None else 0.0
                tangent = torch.tensor((math.cos(angle), math.sin(angle)))
                normal = torch.tensor((-math.sin(angle), math.cos(angle)))
                for bar in range(bar_count):
                    point = center + tangent * ((bar - (bar_count - 1) / 2.0) * 3.0)
                    for offset in (-1.0, 0.0, 1.0):
                        pixel = point + normal * offset
                        x, y = round(float(pixel[0])), round(float(pixel[1]))
                        if 0 <= x < frames.shape[-1] and 0 <= y < frames.shape[-2] and visible[frame, object_index, y, x]:
                            frames[frame, :, y, x] = 1.0
        if 0 <= event.frame < frames.shape[0]:
            for point in event.contact_points:
                x, y = round(point[0]), round(point[1])
                y0, y1 = max(0, y - 2), min(frames.shape[-2], y + 3)
                x0, x1 = max(0, x - 2), min(frames.shape[-1], x + 3)
                frames[event.frame, :, y0:y1, x0:x1] = torch.tensor(
                    (1.0, 0.85, 0.1), dtype=frames.dtype,
                )[:, None, None]
    if physical.relay_events:
        # Neutral one-pixel connectors make the ordered local computation
        # identifiable without introducing a second object class. They sit
        # behind every object and occluder, and routes share the same style.
        occupied = physical.amodal_masks.any(dim=1)
        for tracked_index in physical.tracked_indices:
            events = sorted(
                (event for event in physical.relay_events
                 if event.tracked_index == tracked_index),
                key=lambda event: event.frame,
            )
            edges = []
            if events:
                edges.append((
                    tuple(float(v) for v in physical.centers[0, tracked_index]),
                    events[0].position,
                ))
            for event, following in zip(events, events[1:]):
                edges.extend((
                    (event.position, following.position),
                    (event.position, event.alternate_position),
                ))
            for source, destination in edges:
                count = max(2, round(math.dist(source, destination)))
                xs = torch.linspace(source[0], destination[0], count).round().long()
                ys = torch.linspace(source[1], destination[1], count).round().long()
                xs.clamp_(0, frames.shape[-1] - 1)
                ys.clamp_(0, frames.shape[-2] - 1)
                for x, y in zip(xs.tolist(), ys.tolist()):
                    background = ~occupied[:, y, x]
                    frames[background, :, y, x] = 0.32
    # Dense-relay nodes remain the same disk class. A two-pixel notch on the
    # left/right rim is their only model-visible binary state; that state
    # selects which neighboring quadrant the tracked object visits next.
    for event in physical.relay_events:
        markers = (
            (event.relay_object_index, event.operator_bit, event.position),
            (
                event.alternate_relay_object_index,
                event.alternate_operator_bit,
                event.alternate_position,
            ),
        )
        for object_index, operator_bit, position in markers:
            x, y = position
            offset = physical.object_radius - 0.5
            notch_x = round(x + (offset if operator_bit else -offset))
            notch_y = round(y)
            x0, x1 = max(0, notch_x - 1), min(frames.shape[-1], notch_x + 2)
            y0, y1 = max(0, notch_y - 1), min(frames.shape[-2], notch_y + 2)
            for frame in range(frames.shape[0]):
                local_visible = visible[frame, object_index, y0:y1, x0:x1]
                frames[frame, :, y0:y1, x0:x1] = torch.where(
                    local_visible.unsqueeze(0),
                    torch.ones_like(frames[frame, :, y0:y1, x0:x1]),
                    frames[frame, :, y0:y1, x0:x1],
                )
    # A brief model-visible query cue binds output slots to target identities.
    # The cue disappears long before the first prediction query.
    cue_colors = (
        (1.0, 0.15, 0.15), (0.1, 0.85, 1.0),
        (0.2, 1.0, 0.3), (1.0, 0.8, 0.1),
        (0.85, 0.2, 1.0), (1.0, 0.45, 0.1),
    )
    cue_frames = min(
        physical.centers.shape[0], physical.manifest.target_cue_frames
    )
    side = frames.shape[-1]
    values = torch.arange(side, dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    for frame in range(cue_frames):
        for slot, object_index in enumerate(physical.tracked_indices):
            center = physical.centers[frame, object_index]
            distance2 = (xx - center[0]).square() + (yy - center[1]).square()
            outer = distance2 <= 10.0**2
            inner = distance2 < 7.0**2
            ring = outer & ~inner
            color = torch.tensor(cue_colors[slot % len(cue_colors)])[:, None, None]
            frames[frame] = torch.where(ring.unsqueeze(0), color, frames[frame])
    visibility_rows = []
    for query in physical.query_frames.tolist():
        horizon_rows = []
        for horizon in physical.horizons:
            future = query + horizon
            amodal_area = physical.amodal_masks[future, list(physical.tracked_indices)].sum((-1, -2))
            visible_area = visible[future, list(physical.tracked_indices)].sum((-1, -2))
            horizon_rows.append(visible_area.float() / amodal_area.clamp_min(1).float())
        visibility_rows.append(torch.stack(horizon_rows))
    return ObjectPermanenceEpisode(
        frames=frames,
        visible_masks=visible,
        physical_episode=physical,
        visual_manifest=visual_manifest,
        target_visibility=torch.stack(visibility_rows),
    )


def build_object_permanence_training_examples(
    family: ObjectPermanenceFamily,
) -> Tuple[ObjectPermanenceTrainingExample, ...]:
    """Materialize causal query examples containing no frames after the query."""

    physical = family.physical_episode
    examples = []
    for episode in family.episodes.values():
        for query_index, query_frame in enumerate(physical.query_frames.tolist()):
            occupancy = physical.target_occupancy[query_index].float().clone()
            center_density = physical.target_center_density[query_index].clone()
            uncertainty = torch.zeros(
                (len(physical.horizons), len(physical.tracked_indices)),
                dtype=torch.bool,
            )
            events = {event.object_index: event for event in physical.behavior_events}
            for slot, object_index in enumerate(physical.tracked_indices):
                event = events[object_index]
                for horizon_index, horizon in enumerate(physical.horizons):
                    future = query_frame + horizon
                    unresolved = (
                        query_frame < event.reveal_frame
                        and future >= event.trigger_frame
                    )
                    if unresolved:
                        if query_frame < event.trigger_frame:
                            candidates = torch.arange(
                                physical.behavior_branch_centers.shape[0]
                            )
                        else:
                            candidates = torch.nonzero(
                                physical.behavior_branch_reveal_frames[:, slot]
                                > query_frame,
                                as_tuple=False,
                            ).flatten()
                        if candidates.numel() == 0:
                            candidates = torch.tensor([physical.behavior_variant])
                        candidate_centers = physical.behavior_branch_centers[
                            candidates, future, slot
                        ]
                        occupancy[horizon_index, slot] = _disk_masks(
                            candidate_centers[:, None, :],
                            physical.object_radius,
                            physical.amodal_masks.shape[-1],
                        )[:, 0].float().mean(dim=0)
                        center_density[horizon_index, slot] = torch.stack([
                            _density(
                                center,
                                physical.amodal_masks.shape[-1],
                                physical.center_density_sigma,
                            )
                            for center in candidate_centers
                        ]).mean(dim=0)
                        disagreement = (
                            torch.pdist(candidate_centers).max() > 1e-4
                            if candidate_centers.shape[0] > 1
                            else torch.tensor(False)
                        )
                        uncertainty[horizon_index, slot] = bool(disagreement)
            examples.append(ObjectPermanenceTrainingExample(
                observations=episode.frames[:query_frame + 1],
                future_rgb=torch.stack([
                    episode.frames[query_frame + horizon]
                    for horizon in physical.horizons
                ]),
                query_frame=query_frame,
                horizons=physical.horizons,
                target_occupancy=occupancy,
                target_center_density=center_density,
                family_id=physical.manifest.family_id,
                visual_family_id=episode.visual_manifest.visual_family_id,
                evaluator_target_visibility=episode.target_visibility[query_index],
                evaluator_uncertainty_mask=uncertainty,
            ))
    return tuple(examples)


def generate_object_permanence_family(
    config: ObjectPermanenceConfig,
    visual_config: VisualDomainConfig,
    family_index: int,
    appearance_indices: Sequence[int] = (0, 1, 2),
) -> ObjectPermanenceFamily:
    """Render one physical trajectory under several independent appearances."""

    indices = tuple(int(index) for index in appearance_indices)
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError("appearance_indices must be unique non-negative integers")
    physical = generate_object_permanence_physics(config, family_index)
    episodes = {}
    for appearance_index in indices:
        visual_manifest = make_visual_manifest(
            visual_config,
            physical.manifest.family_id,
            family_index,
            appearance_index,
            physical.object_names,
        )
        if config.ambiguous_tracked_appearance:
            styles = list(visual_manifest.object_styles)
            for target in range(1, config.tracked_count, 2):
                source = styles[target - 1]
                styles[target] = replace(source, object_name=styles[target].object_name)
            style_payload = [style.to_dict() for style in styles]
            ambiguous_id = "vdf-{}".format(hashlib.sha256(
                json.dumps(
                    {
                        "base": visual_manifest.visual_family_id,
                        "tracked_pair_styles": style_payload[:config.tracked_count],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()[:16])
            visual_manifest = replace(
                visual_manifest,
                visual_family_id=ambiguous_id,
                object_styles=tuple(styles),
            )
        if physical.relay_events:
            styles = list(visual_manifest.object_styles)
            source = styles[0]
            for moving_index in range(config.moving_count):
                styles[moving_index] = replace(
                    source, object_name=styles[moving_index].object_name,
                )
            relay_visual_id = "vdf-{}".format(hashlib.sha256(
                json.dumps({
                    "base": visual_manifest.visual_family_id,
                    "same_moving_class": True,
                }, sort_keys=True).encode()
            ).hexdigest()[:16])
            visual_manifest = replace(
                visual_manifest,
                visual_family_id=relay_visual_id,
                object_styles=tuple(styles),
            )
        episodes[appearance_index] = _render_episode(physical, visual_manifest)
    return ObjectPermanenceFamily(physical, episodes)


__all__ = [
    "ContactEvent",
    "GraspProgram",
    "ObjectPermanenceConfig",
    "ObjectPermanenceEpisode",
    "ObjectPermanenceFamily",
    "ObjectPermanenceManifest",
    "ObjectPermanencePhysicalEpisode",
    "ObjectPermanenceTrainingExample",
    "RelayEvent",
    "SceneObjectSpec",
    "build_object_permanence_training_examples",
    "generate_object_permanence_family",
    "generate_object_permanence_physics",
]
