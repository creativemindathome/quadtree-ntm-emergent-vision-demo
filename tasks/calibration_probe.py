"""Deterministic calibration-delay-probe episodes.

The rendered movie is the complete model-visible observation.  Exact object
state, the material law, instance masks, and contact records are evaluator-only
metadata.  A counterfactual family shares every random choice except the
hidden angular bias applied after a wall contact.

The implementation intentionally uses only analytic, fixed-step motion and a
small Torch renderer.  It has no windowing, physics-engine, or model
dependencies, so it is suitable for deterministic dataset generation in
headless jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch


CALIBRATION_OBJECT = "calibration"
PROBE_OBJECT = "probe"
WALLS = ("left", "right", "top", "bottom")

CALIBRATION_IMPACT = "calibration_impact"
CALIBRATION_OUTGOING_1 = "calibration_outgoing_1"
CALIBRATION_OUTGOING_2 = "calibration_outgoing_2"
DELAY = "delay"
PROBE_PREIMPACT = "probe_preimpact"
PROBE_IMPACT = "probe_impact"
PROBE_OUTGOING_1 = "probe_outgoing_1"
PROBE_OUTGOING_2 = "probe_outgoing_2"

_STREAM_NAMES = ("timing", "geometry", "distractors", "law")


@dataclass(frozen=True)
class CalibrationProbeConfig:
    """Configuration for a family of 128x128 calibration-probe episodes.

    Impact windows are inclusive.  Timing is sampled once per family from the
    named ``timing`` stream, while the supplied branch bias is the only value
    allowed to differ between counterfactual manifests.
    """

    image_size: int = 128
    episode_length: int = 64
    radius: float = 5.0
    speed: float = 4.0
    seed: int = 17
    calibration_impact_window: Tuple[int, int] = (6, 10)
    probe_impact_window: Tuple[int, int] = (44, 48)
    outgoing_evidence_frames: int = 2
    probe_preimpact_frames: int = 3
    bias_min_degrees: float = -45.0
    bias_max_degrees: float = 45.0
    incidence_angle_degrees: float = 0.0
    probe_count: int = 1
    probe_tangent_spacing: float = 10.0
    distractor_count: int = 1
    distractor_speed: float = 0.75
    target_horizons: Tuple[int, ...] = (1, 4, 8)
    center_density_sigma: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calibration_impact_window",
            tuple(self.calibration_impact_window),
        )
        object.__setattr__(
            self,
            "probe_impact_window",
            tuple(self.probe_impact_window),
        )
        object.__setattr__(self, "target_horizons", tuple(self.target_horizons))

        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if self.episode_length < 16:
            raise ValueError("episode_length must be at least 16")
        if not 0.0 < self.radius < self.image_size / 4.0:
            raise ValueError(
                "radius must be positive and smaller than image_size / 4"
            )
        if not math.isfinite(self.speed) or self.speed <= 0.0:
            raise ValueError("speed must be finite and positive")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")

        self._validate_window("calibration_impact_window", self.calibration_impact_window)
        self._validate_window("probe_impact_window", self.probe_impact_window)
        calibration_low, calibration_high = self.calibration_impact_window
        probe_low, probe_high = self.probe_impact_window

        if calibration_low < 1:
            raise ValueError("calibration impact must occur after frame zero")
        if self.outgoing_evidence_frames < 2:
            raise ValueError("outgoing_evidence_frames must be at least two")
        if self.probe_preimpact_frames < 1:
            raise ValueError("probe_preimpact_frames must be positive")
        if (
            calibration_high + self.outgoing_evidence_frames
            >= probe_low - self.probe_preimpact_frames
        ):
            raise ValueError(
                "impact windows must leave a non-empty calibration-free delay"
            )

        if not self.target_horizons:
            raise ValueError("target_horizons must not be empty")
        if any(
            not isinstance(horizon, int) or horizon <= 0
            for horizon in self.target_horizons
        ):
            raise ValueError("target_horizons must contain positive integers")
        if tuple(sorted(set(self.target_horizons))) != self.target_horizons:
            raise ValueError("target_horizons must be unique and increasing")
        if probe_high + max(self.target_horizons) >= self.episode_length:
            raise ValueError(
                "probe impact window plus largest target horizon must fit in episode"
            )

        if not (
            math.isfinite(self.bias_min_degrees)
            and math.isfinite(self.bias_max_degrees)
            and self.bias_min_degrees <= self.bias_max_degrees
        ):
            raise ValueError("bias bounds must be finite and ordered")
        if not math.isfinite(self.incidence_angle_degrees):
            raise ValueError("incidence_angle_degrees must be finite")
        maximum_outgoing_angle = max(
            abs(self.bias_min_degrees), abs(self.bias_max_degrees)
        ) + abs(self.incidence_angle_degrees)
        if maximum_outgoing_angle >= 89.0:
            raise ValueError(
                "bias plus incidence must remain below 89 degrees so rebounds point inward"
            )

        available_span = self.image_size - 1.0 - 2.0 * self.radius
        if self.speed * calibration_high > available_span:
            raise ValueError(
                "speed and calibration timing place the initial disk outside the image"
            )
        remaining_after_earliest_probe = self.episode_length - 1 - probe_low
        maximum_tangent_travel = (
            remaining_after_earliest_probe
            * self.speed
            * math.sin(math.radians(maximum_outgoing_angle))
        )
        if 2.0 * maximum_tangent_travel >= available_span:
            raise ValueError(
                "probe trajectory can reach a second wall; reduce speed, bias, or duration"
            )
        if not isinstance(self.probe_count, int) or self.probe_count < 1:
            raise ValueError("probe_count must be a positive integer")
        if not math.isfinite(self.probe_tangent_spacing) or self.probe_tangent_spacing <= 0:
            raise ValueError("probe_tangent_spacing must be finite and positive")
        probe_span = (self.probe_count - 1) * self.probe_tangent_spacing
        usable_tangent_span = available_span - 2.0 * maximum_tangent_travel
        if probe_span > usable_tangent_span:
            raise ValueError(
                "probe array does not fit safely along the contacted wall"
            )

        if not isinstance(self.distractor_count, int) or self.distractor_count < 1:
            raise ValueError("distractor_count must be a positive integer")
        if not math.isfinite(self.distractor_speed) or self.distractor_speed < 0.0:
            raise ValueError("distractor_speed must be finite and non-negative")
        if (
            not math.isfinite(self.center_density_sigma)
            or self.center_density_sigma <= 0.0
        ):
            raise ValueError("center_density_sigma must be finite and positive")

    @staticmethod
    def _validate_window(name: str, window: Tuple[int, int]) -> None:
        if len(window) != 2:
            raise ValueError("{} must contain (first, last)".format(name))
        first, last = window
        if not isinstance(first, int) or not isinstance(last, int):
            raise TypeError("{} bounds must be integers".format(name))
        if first > last:
            raise ValueError("{} must be ordered".format(name))


@dataclass(frozen=True)
class ReflectionLaw:
    """Episode-constant latent material law."""

    bias_degrees: float
    name: str = "biased_specular"

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "bias_degrees": float(self.bias_degrees)}


@dataclass(frozen=True)
class ContactEvent:
    """Evaluator record for a rendered wall-contact frame."""

    frame: int
    phase: str
    object: str
    wall: str
    position: Tuple[float, float]
    pre_velocity: Tuple[float, float]
    post_velocity: Tuple[float, float]
    bias: float

    @property
    def bias_degrees(self) -> float:
        return self.bias


@dataclass(frozen=True)
class EpisodeManifest:
    """Reproducibility record for one branch in a counterfactual family."""

    family_id: str
    root_seed: int
    family_index: int
    rng_seeds: Mapping[str, int]
    latent_law: ReflectionLaw

    def __post_init__(self) -> None:
        object.__setattr__(self, "rng_seeds", MappingProxyType(dict(self.rng_seeds)))

    @property
    def named_seeds(self) -> Mapping[str, int]:
        return self.rng_seeds

    @property
    def law(self) -> ReflectionLaw:
        return self.latent_law

    def to_json_dict(self) -> Dict[str, object]:
        """Return a JSON-native representation without dataclass magic."""

        return {
            "family_id": self.family_id,
            "root_seed": int(self.root_seed),
            "family_index": int(self.family_index),
            "rng_seeds": {
                name: int(seed) for name, seed in self.rng_seeds.items()
            },
            "latent_law": self.latent_law.to_dict(),
        }

    def to_dict(self) -> Dict[str, object]:
        return self.to_json_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True)


@dataclass(frozen=True)
class CalibrationProbeEpisode:
    """Rendered observations and evaluator-only state for one episode."""

    frames: torch.Tensor
    instance_masks: torch.Tensor
    centers: torch.Tensor
    velocities: torch.Tensor
    alive: torch.Tensor
    contact_events: Tuple[ContactEvent, ...]
    phase_labels: Tuple[str, ...]
    law: ReflectionLaw
    manifest: EpisodeManifest
    object_names: Tuple[str, ...]

    @property
    def observations(self) -> torch.Tensor:
        """The complete model-visible contract; metadata is never included."""

        return self.frames

    @property
    def observation(self) -> torch.Tensor:
        return self.frames

    @property
    def evaluator_metadata(self) -> Dict[str, object]:
        """Return evaluator fields separately from the observation tensor."""

        return {
            "instance_masks": self.instance_masks,
            "centers": self.centers,
            "velocities": self.velocities,
            "alive": self.alive,
            "contact_events": self.contact_events,
            "phase_labels": self.phase_labels,
            "law": self.law,
            "manifest": self.manifest,
            "object_names": self.object_names,
        }


@dataclass(frozen=True)
class CounterfactualFamily:
    """Paired branches and evaluator targets conditioned on the family."""

    family_id: str
    episodes: Mapping[float, CalibrationProbeEpisode]
    query_frames: torch.Tensor
    horizons: Tuple[int, ...]
    conditional_occupancy: Mapping[float, torch.Tensor]
    center_density: Mapping[float, torch.Tensor]
    law_marginal_occupancy: torch.Tensor
    law_marginal_center_density: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", MappingProxyType(dict(self.episodes)))
        object.__setattr__(
            self,
            "conditional_occupancy",
            MappingProxyType(dict(self.conditional_occupancy)),
        )
        object.__setattr__(
            self,
            "center_density",
            MappingProxyType(dict(self.center_density)),
        )

    @property
    def branches(self) -> Mapping[float, CalibrationProbeEpisode]:
        return self.episodes

    @property
    def valid_query_frames(self) -> torch.Tensor:
        return self.query_frames

    @property
    def conditional_occupancy_targets(self) -> Mapping[float, torch.Tensor]:
        return self.conditional_occupancy

    @property
    def center_density_targets(self) -> Mapping[float, torch.Tensor]:
        return self.center_density

    @property
    def branch_biases(self) -> Tuple[float, ...]:
        return tuple(self.episodes)

    @property
    def stacked_conditional_occupancy(self) -> torch.Tensor:
        return torch.stack(
            [self.conditional_occupancy[bias] for bias in self.branch_biases]
        )

    @property
    def stacked_center_density(self) -> torch.Tensor:
        return torch.stack([self.center_density[bias] for bias in self.branch_biases])

    def __getitem__(self, key: str) -> object:
        """Small dict-style compatibility surface for data-pipeline callers."""

        aliases = {
            "branches": self.episodes,
            "episodes": self.episodes,
            "query_frames": self.query_frames,
            "valid_query_frames": self.query_frames,
            "horizons": self.horizons,
            "conditional_occupancy": self.conditional_occupancy,
            "center_density": self.center_density,
            "law_marginal_occupancy": self.law_marginal_occupancy,
            "law_marginal_center_density": self.law_marginal_center_density,
        }
        try:
            return aliases[key]
        except KeyError as error:
            raise KeyError(key) from error


def _derived_seed(root_seed: int, family_index: int, stream_name: str) -> int:
    payload = "calibration-probe-v2|{}|{}|{}".format(
        root_seed, family_index, stream_name
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Torch generators accept signed 64-bit seeds.  Zero is valid, but adding
    # one gives a compact, explicitly positive range.
    return int.from_bytes(digest[:8], "big") % (2**63 - 2) + 1


def _family_id(root_seed: int, family_index: int) -> str:
    payload = "calibration-probe-family|{}|{}".format(
        root_seed, family_index
    ).encode("utf-8")
    return "cpf-{}".format(hashlib.sha256(payload).hexdigest()[:16])


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    if low == high:
        return float(low)
    unit = float(torch.rand((), generator=generator).item())
    return float(low + (high - low) * unit)


def make_manifest(
    config: CalibrationProbeConfig,
    family_index: int,
    branch_bias_degrees: Optional[float] = None,
) -> EpisodeManifest:
    """Build a deterministic manifest with independent named RNG streams."""

    if not isinstance(family_index, int) or family_index < 0:
        raise ValueError("family_index must be a non-negative integer")

    seeds = {
        name: _derived_seed(config.seed, family_index, name)
        for name in _STREAM_NAMES
    }
    if branch_bias_degrees is None:
        branch_bias_degrees = _uniform(
            _generator(seeds["law"]),
            config.bias_min_degrees,
            config.bias_max_degrees,
        )
    if not math.isfinite(branch_bias_degrees):
        raise ValueError("branch bias must be finite")
    if not config.bias_min_degrees <= branch_bias_degrees <= config.bias_max_degrees:
        raise ValueError("branch bias must lie within the configured bias bounds")

    return EpisodeManifest(
        family_id=_family_id(config.seed, family_index),
        root_seed=config.seed,
        family_index=family_index,
        rng_seeds=seeds,
        latent_law=ReflectionLaw(float(branch_bias_degrees)),
    )


def _sample_inclusive(
    generator: torch.Generator, window: Tuple[int, int]
) -> int:
    first, last = window
    return int(torch.randint(first, last + 1, (), generator=generator).item())


def _normal_and_tangent(wall: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # ``normal`` points from the wall into the image.  Image y increases down.
    vectors = {
        "left": ((1.0, 0.0), (0.0, 1.0)),
        "right": ((-1.0, 0.0), (0.0, -1.0)),
        "top": ((0.0, 1.0), (-1.0, 0.0)),
        "bottom": ((0.0, -1.0), (1.0, 0.0)),
    }
    normal_values, tangent_values = vectors[wall]
    return (
        torch.tensor(normal_values, dtype=torch.float32),
        torch.tensor(tangent_values, dtype=torch.float32),
    )


def _contact_position(
    wall: str, tangent_coordinate: float, config: CalibrationProbeConfig
) -> torch.Tensor:
    low = config.radius
    high = config.image_size - 1.0 - config.radius
    if wall == "left":
        values = (low, tangent_coordinate)
    elif wall == "right":
        values = (high, tangent_coordinate)
    elif wall == "top":
        values = (tangent_coordinate, low)
    else:
        values = (tangent_coordinate, high)
    return torch.tensor(values, dtype=torch.float32)


def _incoming_velocity(
    wall: str, config: CalibrationProbeConfig
) -> torch.Tensor:
    normal, tangent = _normal_and_tangent(wall)
    incidence = math.radians(config.incidence_angle_degrees)
    # The normal component points toward the wall before impact.
    return config.speed * (-math.cos(incidence) * normal + math.sin(incidence) * tangent)


def _outgoing_velocity(
    incoming: torch.Tensor, wall: str, bias_degrees: float
) -> torch.Tensor:
    normal, _ = _normal_and_tangent(wall)
    specular = incoming - 2.0 * torch.dot(incoming, normal) * normal
    angle = math.radians(bias_degrees)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float32,
    )
    outgoing = rotation @ specular
    if float(torch.dot(outgoing, normal).item()) <= 0.0:
        raise ValueError("configured law does not produce an inward rebound")
    return outgoing


_GRID_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _coordinate_grid(image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    grid = _GRID_CACHE.get(image_size)
    if grid is None:
        coordinates = torch.arange(image_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        grid = (xx, yy)
        _GRID_CACHE[image_size] = grid
    return grid


def _disk_mask(
    center: torch.Tensor, config: CalibrationProbeConfig
) -> torch.Tensor:
    xx, yy = _coordinate_grid(config.image_size)
    signed_distance_squared = (
        (xx - center[0]).square()
        + (yy - center[1]).square()
        - config.radius**2
    )
    return signed_distance_squared <= 0.0


def _density(
    center: torch.Tensor, config: CalibrationProbeConfig
) -> torch.Tensor:
    xx, yy = _coordinate_grid(config.image_size)
    squared_distance = (xx - center[0]).square() + (yy - center[1]).square()
    result = torch.exp(
        -squared_distance / (2.0 * config.center_density_sigma**2)
    )
    return result / result.sum()


def _reflecting_coordinate(
    start: float, velocity: float, step: int, low: float, high: float
) -> Tuple[float, float]:
    """Analytic 1-D reflection, used only by branch-invariant distractors."""

    span = high - low
    unfolded = (start - low + velocity * step) % (2.0 * span)
    if unfolded <= span:
        return low + unfolded, velocity
    return low + 2.0 * span - unfolded, -velocity


def _phase_labels(
    config: CalibrationProbeConfig,
    calibration_impact: int,
    probe_impact: int,
) -> List[str]:
    labels = [DELAY] * config.episode_length
    for frame in range(calibration_impact):
        labels[frame] = "calibration_approach"
    labels[calibration_impact] = CALIBRATION_IMPACT
    labels[calibration_impact + 1] = CALIBRATION_OUTGOING_1
    labels[calibration_impact + 2] = CALIBRATION_OUTGOING_2

    probe_start = probe_impact - config.probe_preimpact_frames
    for frame in range(probe_start, probe_impact):
        labels[frame] = PROBE_PREIMPACT
    labels[probe_impact] = PROBE_IMPACT
    labels[probe_impact + 1] = PROBE_OUTGOING_1
    labels[probe_impact + 2] = PROBE_OUTGOING_2
    for frame in range(probe_impact + 3, config.episode_length):
        labels[frame] = "probe_outgoing"
    return labels


def _as_pair(values: torch.Tensor) -> Tuple[float, float]:
    return float(values[0].item()), float(values[1].item())


def generate_calibration_probe_episode(
    config: CalibrationProbeConfig,
    manifest: EpisodeManifest,
) -> CalibrationProbeEpisode:
    """Generate one branch from an explicit, reproducible manifest."""

    expected_seeds = {
        name: _derived_seed(manifest.root_seed, manifest.family_index, name)
        for name in _STREAM_NAMES
    }
    if manifest.root_seed != config.seed:
        raise ValueError("manifest root_seed does not match config.seed")
    if dict(manifest.rng_seeds) != expected_seeds:
        raise ValueError("manifest RNG seeds do not match its family identity")
    if not (
        config.bias_min_degrees
        <= manifest.latent_law.bias_degrees
        <= config.bias_max_degrees
    ):
        raise ValueError("manifest law lies outside the configured bias bounds")

    timing_generator = _generator(manifest.rng_seeds["timing"])
    geometry_generator = _generator(manifest.rng_seeds["geometry"])
    distractor_generator = _generator(manifest.rng_seeds["distractors"])

    calibration_impact = _sample_inclusive(
        timing_generator, config.calibration_impact_window
    )
    probe_impact = _sample_inclusive(timing_generator, config.probe_impact_window)
    wall = WALLS[
        int(torch.randint(0, len(WALLS), (), generator=geometry_generator).item())
    ]

    maximum_outgoing_angle = max(
        abs(config.bias_min_degrees), abs(config.bias_max_degrees)
    ) + abs(config.incidence_angle_degrees)
    maximum_tangent_travel = (
        (config.episode_length - 1 - config.probe_impact_window[0])
        * config.speed
        * math.sin(math.radians(maximum_outgoing_angle))
    )
    low = config.radius
    high = config.image_size - 1.0 - config.radius
    tangent_low = low + maximum_tangent_travel
    tangent_high = high - maximum_tangent_travel
    calibration_tangent = _uniform(
        geometry_generator, tangent_low, tangent_high
    )
    probe_half_span = 0.5 * (config.probe_count - 1) * config.probe_tangent_spacing
    probe_center_tangent = _uniform(
        geometry_generator,
        tangent_low + probe_half_span,
        tangent_high - probe_half_span,
    )
    probe_tangents = [
        probe_center_tangent
        + (index - 0.5 * (config.probe_count - 1)) * config.probe_tangent_spacing
        for index in range(config.probe_count)
    ]

    incoming = _incoming_velocity(wall, config)
    outgoing = _outgoing_velocity(
        incoming, wall, manifest.latent_law.bias_degrees
    )
    calibration_contact = _contact_position(wall, calibration_tangent, config)
    probe_contacts = [
        _contact_position(wall, tangent, config) for tangent in probe_tangents
    ]

    num_instances = 1 + config.probe_count + config.distractor_count
    centers = torch.zeros(
        (config.episode_length, num_instances, 2), dtype=torch.float32
    )
    velocities = torch.zeros_like(centers)
    alive = torch.zeros(
        (config.episode_length, num_instances), dtype=torch.bool
    )

    # Contact is rendered with the incoming state.  The biased post-contact
    # velocity first affects the following frame.
    for frame in range(calibration_impact + 1):
        centers[frame, 0] = calibration_contact - incoming * (
            calibration_impact - frame
        )
        velocities[frame, 0] = incoming
        alive[frame, 0] = True
    for offset in range(1, config.outgoing_evidence_frames + 1):
        frame = calibration_impact + offset
        centers[frame, 0] = calibration_contact + outgoing * offset
        velocities[frame, 0] = outgoing
        alive[frame, 0] = True

    probe_start = probe_impact - config.probe_preimpact_frames
    for probe_index, probe_contact in enumerate(probe_contacts):
        instance_index = 1 + probe_index
        for frame in range(probe_start, probe_impact + 1):
            centers[frame, instance_index] = probe_contact - incoming * (
                probe_impact - frame
            )
            velocities[frame, instance_index] = incoming
            alive[frame, instance_index] = True
        for frame in range(probe_impact + 1, config.episode_length):
            offset = frame - probe_impact
            centers[frame, instance_index] = probe_contact + outgoing * offset
            velocities[frame, instance_index] = outgoing
            alive[frame, instance_index] = True

    delay_start = calibration_impact + config.outgoing_evidence_frames + 1
    delay_end = probe_start
    for distractor_index in range(config.distractor_count):
        instance_index = 1 + config.probe_count + distractor_index
        start = torch.tensor(
            [
                _uniform(distractor_generator, low, high),
                _uniform(distractor_generator, low, high),
            ],
            dtype=torch.float32,
        )
        angle = _uniform(distractor_generator, -math.pi, math.pi)
        base_velocity = torch.tensor(
            [
                config.distractor_speed * math.cos(angle),
                config.distractor_speed * math.sin(angle),
            ],
            dtype=torch.float32,
        )
        for frame in range(delay_start, delay_end):
            step = frame - delay_start
            x, vx = _reflecting_coordinate(
                float(start[0].item()),
                float(base_velocity[0].item()),
                step,
                low,
                high,
            )
            y, vy = _reflecting_coordinate(
                float(start[1].item()),
                float(base_velocity[1].item()),
                step,
                low,
                high,
            )
            centers[frame, instance_index] = torch.tensor((x, y))
            velocities[frame, instance_index] = torch.tensor((vx, vy))
            alive[frame, instance_index] = True

    instance_masks = torch.zeros(
        (
            config.episode_length,
            num_instances,
            config.image_size,
            config.image_size,
        ),
        dtype=torch.bool,
    )
    for frame in range(config.episode_length):
        for instance_index in range(num_instances):
            if bool(alive[frame, instance_index].item()):
                instance_masks[frame, instance_index] = _disk_mask(
                    centers[frame, instance_index], config
                )
    frames = instance_masks.any(dim=1).to(torch.float32).unsqueeze(1)

    labels = _phase_labels(config, calibration_impact, probe_impact)
    events = tuple([
        ContactEvent(
            frame=calibration_impact,
            phase=CALIBRATION_IMPACT,
            object=CALIBRATION_OBJECT,
            wall=wall,
            position=_as_pair(calibration_contact),
            pre_velocity=_as_pair(incoming),
            post_velocity=_as_pair(outgoing),
            bias=manifest.latent_law.bias_degrees,
        ),
    ] + [
        ContactEvent(
            frame=probe_impact,
            phase=PROBE_IMPACT,
            object=(PROBE_OBJECT if index == 0 else "probe_{}".format(index)),
            wall=wall,
            position=_as_pair(probe_contact),
            pre_velocity=_as_pair(incoming),
            post_velocity=_as_pair(outgoing),
            bias=manifest.latent_law.bias_degrees,
        )
        for index, probe_contact in enumerate(probe_contacts)
    ])
    object_names = tuple([
        CALIBRATION_OBJECT,
    ] + [
        PROBE_OBJECT if index == 0 else "probe_{}".format(index)
        for index in range(config.probe_count)
    ] + [
        "distractor_{}".format(index) for index in range(config.distractor_count)
    ])
    return CalibrationProbeEpisode(
        frames=frames,
        instance_masks=instance_masks,
        centers=centers,
        velocities=velocities,
        alive=alive,
        contact_events=events,
        phase_labels=tuple(labels),
        law=manifest.latent_law,
        manifest=manifest,
        object_names=object_names,
    )


def generate_counterfactual_family(
    config: CalibrationProbeConfig,
    family_index: int,
    biases: Sequence[float] = (-45.0, 45.0),
) -> CounterfactualFamily:
    """Generate paired branches and both conditional and marginal targets.

    Each bias key in ``conditional_occupancy`` and ``center_density`` maps to a
    tensor shaped ``[query, horizon, height, width]``.  These law-conditioned
    targets are appropriate after the calibration evidence has been retained.
    The explicitly named ``law_marginal_*`` tensors average those branches for
    reset/unknown-law evaluation.  Query frames are probe pre-impact/contact
    observations for which every configured future horizon is present and the
    probe is alive in every branch.
    """

    branch_biases = tuple(float(bias) for bias in biases)
    if not branch_biases:
        raise ValueError("biases must contain at least one branch")
    if len(set(branch_biases)) != len(branch_biases):
        raise ValueError("biases must be unique")

    episodes: Dict[float, CalibrationProbeEpisode] = {}
    for bias in branch_biases:
        manifest = make_manifest(config, family_index, bias)
        episodes[bias] = generate_calibration_probe_episode(config, manifest)

    episode_values = tuple(episodes.values())
    family_ids = {episode.manifest.family_id for episode in episode_values}
    if len(family_ids) != 1:
        raise RuntimeError("counterfactual branches do not share a family id")

    valid_queries: List[int] = []
    for frame, phase in enumerate(episode_values[0].phase_labels):
        if phase not in (PROBE_PREIMPACT, PROBE_IMPACT):
            continue
        if all(
            frame + horizon < config.episode_length
            and all(
                bool(episode.alive[
                    frame + horizon, 1:1 + config.probe_count
                ].all().item())
                for episode in episode_values
            )
            for horizon in config.target_horizons
        ):
            valid_queries.append(frame)
    if not valid_queries:
        raise RuntimeError("configuration produced no valid probe query frames")

    conditional_occupancy: Dict[float, torch.Tensor] = {}
    center_density: Dict[float, torch.Tensor] = {}
    for bias, episode in episodes.items():
        occupancy_rows: List[torch.Tensor] = []
        density_rows: List[torch.Tensor] = []
        for query_frame in valid_queries:
            occupancy_rows.append(
                torch.stack(
                    [
                        episode.instance_masks[
                            query_frame + horizon, 1:1 + config.probe_count
                        ].any(dim=0).to(torch.float32)
                        for horizon in config.target_horizons
                    ]
                )
            )
            density_rows.append(
                torch.stack(
                    [
                        torch.stack([
                            _density(
                                episode.centers[
                                    query_frame + horizon, 1 + probe_index
                                ],
                                config,
                            )
                            for probe_index in range(config.probe_count)
                        ]).mean(dim=0)
                        for horizon in config.target_horizons
                    ]
                )
            )
        conditional_occupancy[bias] = torch.stack(occupancy_rows)
        center_density[bias] = torch.stack(density_rows)

    law_marginal_occupancy = torch.stack(
        [conditional_occupancy[bias] for bias in branch_biases]
    ).mean(dim=0)
    law_marginal_center_density = torch.stack(
        [center_density[bias] for bias in branch_biases]
    ).mean(dim=0)

    return CounterfactualFamily(
        family_id=episode_values[0].manifest.family_id,
        episodes=episodes,
        query_frames=torch.tensor(valid_queries, dtype=torch.int64),
        horizons=config.target_horizons,
        conditional_occupancy=conditional_occupancy,
        center_density=center_density,
        law_marginal_occupancy=law_marginal_occupancy,
        law_marginal_center_density=law_marginal_center_density,
    )


__all__ = [
    "CALIBRATION_IMPACT",
    "CALIBRATION_OUTGOING_1",
    "CALIBRATION_OUTGOING_2",
    "DELAY",
    "PROBE_PREIMPACT",
    "PROBE_IMPACT",
    "PROBE_OUTGOING_1",
    "PROBE_OUTGOING_2",
    "CalibrationProbeConfig",
    "CalibrationProbeEpisode",
    "ContactEvent",
    "CounterfactualFamily",
    "EpisodeManifest",
    "ReflectionLaw",
    "generate_calibration_probe_episode",
    "generate_counterfactual_family",
    "make_manifest",
]
