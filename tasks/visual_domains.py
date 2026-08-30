"""Deterministic RGB visual domains for causal video environments.

Physics and evaluator masks are inputs to this module.  Appearance is sampled
from independent named streams and replayed from an explicit manifest.  Kornia
is used only for deterministic tensor transforms; no hidden global RNG state is
part of the replay contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple

import kornia
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_saturation
from kornia.filters import gaussian_blur2d
import torch

from tasks.calibration_probe import CalibrationProbeEpisode


_STREAM_NAMES = (
    "appearance_object",
    "appearance_background",
    "illumination",
    "sensor_noise",
)

_PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (0.92, 0.25, 0.22),
    (0.18, 0.68, 0.92),
    (0.23, 0.82, 0.43),
    (0.96, 0.73, 0.18),
    (0.67, 0.35, 0.91),
    (0.95, 0.42, 0.68),
    (0.16, 0.82, 0.78),
    (0.95, 0.51, 0.16),
    (0.70, 0.82, 0.22),
    (0.44, 0.55, 0.96),
    (0.92, 0.92, 0.88),
    (0.56, 0.74, 0.86),
)

_BACKGROUND_PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (0.035, 0.045, 0.070),
    (0.070, 0.045, 0.090),
    (0.035, 0.075, 0.075),
    (0.085, 0.060, 0.035),
    (0.055, 0.065, 0.095),
    (0.080, 0.040, 0.050),
)


@dataclass(frozen=True)
class VisualDomainConfig:
    """Bounded RGB-v1 appearance distribution."""

    seed: int = 211
    object_color_count: int = 12
    object_texture_count: int = 4
    background_family_count: int = 12
    brightness_range: Tuple[float, float] = (-0.06, 0.06)
    contrast_range: Tuple[float, float] = (0.82, 1.18)
    saturation_range: Tuple[float, float] = (0.82, 1.18)
    illumination_strength_range: Tuple[float, float] = (0.05, 0.22)
    noise_std_range: Tuple[float, float] = (0.0, 0.025)
    blur_sigma_range: Tuple[float, float] = (0.15, 0.85)
    renderer_version: str = "rgb-v1-kornia-0.8.3"

    def __post_init__(self) -> None:
        for name in (
            "brightness_range",
            "contrast_range",
            "saturation_range",
            "illumination_strength_range",
            "noise_std_range",
            "blur_sigma_range",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain two finite values")
            if values[0] > values[1]:
                raise ValueError(f"{name} must be ordered")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if not 1 <= self.object_color_count <= len(_PALETTE):
            raise ValueError("object_color_count exceeds the built-in palette")
        if not 1 <= self.object_texture_count <= 4:
            raise ValueError("object_texture_count must lie in [1, 4]")
        if not 1 <= self.background_family_count <= 12:
            raise ValueError("background_family_count must lie in [1, 12]")
        if self.noise_std_range[0] < 0.0 or self.blur_sigma_range[0] <= 0.0:
            raise ValueError("noise must be non-negative and blur sigma positive")
        if self.contrast_range[0] < 0.0 or self.saturation_range[0] < 0.0:
            raise ValueError("contrast and saturation factors must be non-negative")


@dataclass(frozen=True)
class ObjectStyle:
    """Episode-persistent appearance for one evaluator object."""

    object_name: str
    color_id: int
    accent_color_id: int
    texture_id: int
    frequency: float
    phase: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_name": self.object_name,
            "color_id": self.color_id,
            "accent_color_id": self.accent_color_id,
            "texture_id": self.texture_id,
            "frequency": self.frequency,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class VisualManifest:
    """Complete replay record for one visual realization."""

    visual_family_id: str
    physical_family_id: str
    root_seed: int
    family_index: int
    appearance_index: int
    rng_seeds: Mapping[str, int]
    object_styles: Tuple[ObjectStyle, ...]
    background_family_id: int
    background_color_id: int
    background_accent_id: int
    background_frequency: float
    background_phase: float
    illumination_angle: float
    illumination_strength: float
    brightness: float
    contrast: float
    saturation: float
    noise_std: float
    blur_sigma: float
    renderer_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rng_seeds", MappingProxyType(dict(self.rng_seeds)))
        object.__setattr__(self, "object_styles", tuple(self.object_styles))

    @property
    def object_style_ids(self) -> Mapping[str, str]:
        return MappingProxyType({
            style.object_name: f"c{style.color_id}-t{style.texture_id}"
            for style in self.object_styles
        })

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "visual_family_id": self.visual_family_id,
            "physical_family_id": self.physical_family_id,
            "root_seed": self.root_seed,
            "family_index": self.family_index,
            "appearance_index": self.appearance_index,
            "rng_seeds": dict(self.rng_seeds),
            "object_styles": [style.to_dict() for style in self.object_styles],
            "object_style_ids": dict(self.object_style_ids),
            "background": {
                "family_id": self.background_family_id,
                "color_id": self.background_color_id,
                "accent_id": self.background_accent_id,
                "frequency": self.background_frequency,
                "phase": self.background_phase,
            },
            "illumination": {
                "angle": self.illumination_angle,
                "strength": self.illumination_strength,
            },
            "sensor": {
                "brightness": self.brightness,
                "contrast": self.contrast,
                "saturation": self.saturation,
                "noise_std": self.noise_std,
                "blur_sigma": self.blur_sigma,
            },
            "renderer_version": self.renderer_version,
            "kornia_version": kornia.__version__,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True)

    def __reduce__(self):
        return (VisualManifest.from_json_dict, (self.to_json_dict(),))

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "VisualManifest":
        """Restore a persisted manifest without resampling any RNG stream."""

        background = dict(payload["background"])
        illumination = dict(payload["illumination"])
        sensor = dict(payload["sensor"])
        styles = tuple(
            ObjectStyle(
                object_name=str(style["object_name"]),
                color_id=int(style["color_id"]),
                accent_color_id=int(style["accent_color_id"]),
                texture_id=int(style["texture_id"]),
                frequency=float(style["frequency"]),
                phase=float(style["phase"]),
            )
            for style in payload["object_styles"]
        )
        return cls(
            visual_family_id=str(payload["visual_family_id"]),
            physical_family_id=str(payload["physical_family_id"]),
            root_seed=int(payload["root_seed"]),
            family_index=int(payload["family_index"]),
            appearance_index=int(payload["appearance_index"]),
            rng_seeds={str(key): int(value) for key, value in dict(payload["rng_seeds"]).items()},
            object_styles=styles,
            background_family_id=int(background["family_id"]),
            background_color_id=int(background["color_id"]),
            background_accent_id=int(background["accent_id"]),
            background_frequency=float(background["frequency"]),
            background_phase=float(background["phase"]),
            illumination_angle=float(illumination["angle"]),
            illumination_strength=float(illumination["strength"]),
            brightness=float(sensor["brightness"]),
            contrast=float(sensor["contrast"]),
            saturation=float(sensor["saturation"]),
            noise_std=float(sensor["noise_std"]),
            blur_sigma=float(sensor["blur_sigma"]),
            renderer_version=str(payload["renderer_version"]),
        )


def _derived_seed(
    root_seed: int,
    physical_family_id: str,
    family_index: int,
    appearance_index: int,
    stream_name: str,
) -> int:
    payload = (
        f"visual-domain-v1|{root_seed}|{physical_family_id}|"
        f"{family_index}|{appearance_index}|{stream_name}"
    ).encode("utf8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 2) + 1


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def _integer(generator: torch.Generator, high: int) -> int:
    return int(torch.randint(0, high, (), generator=generator).item())


def make_visual_manifest(
    config: VisualDomainConfig,
    physical_family_id: str,
    family_index: int,
    appearance_index: int,
    object_names: Sequence[str],
) -> VisualManifest:
    """Sample a branch-independent visual manifest from named streams."""

    if not physical_family_id:
        raise ValueError("physical_family_id must not be empty")
    if family_index < 0 or appearance_index < 0:
        raise ValueError("family and appearance indices must be non-negative")
    names = tuple(str(name) for name in object_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("object_names must be non-empty and unique")

    seeds = {
        name: _derived_seed(
            config.seed,
            physical_family_id,
            family_index,
            appearance_index,
            name,
        )
        for name in _STREAM_NAMES
    }
    object_rng = _generator(seeds["appearance_object"])
    background_rng = _generator(seeds["appearance_background"])
    illumination_rng = _generator(seeds["illumination"])
    sensor_rng = _generator(seeds["sensor_noise"])

    styles = []
    for name in names:
        color_id = _integer(object_rng, config.object_color_count)
        accent_id = _integer(object_rng, config.object_color_count)
        if config.object_color_count > 1 and accent_id == color_id:
            accent_id = (accent_id + 1) % config.object_color_count
        styles.append(ObjectStyle(
            object_name=name,
            color_id=color_id,
            accent_color_id=accent_id,
            texture_id=_integer(object_rng, config.object_texture_count),
            frequency=_uniform(object_rng, 1.2, 3.8),
            phase=_uniform(object_rng, -math.pi, math.pi),
        ))

    family_id = _integer(background_rng, config.background_family_count)
    visual_payload = json.dumps(
        {
            "physical_family_id": physical_family_id,
            "family_index": family_index,
            "appearance_index": appearance_index,
            "object_names": names,
            "config": config.__dict__,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf8")
    visual_id = "vdf-{}".format(hashlib.sha256(visual_payload).hexdigest()[:16])
    return VisualManifest(
        visual_family_id=visual_id,
        physical_family_id=physical_family_id,
        root_seed=config.seed,
        family_index=family_index,
        appearance_index=appearance_index,
        rng_seeds=seeds,
        object_styles=tuple(styles),
        background_family_id=family_id,
        background_color_id=_integer(background_rng, len(_BACKGROUND_PALETTE)),
        background_accent_id=_integer(background_rng, len(_BACKGROUND_PALETTE)),
        background_frequency=_uniform(background_rng, 0.7, 2.6),
        background_phase=_uniform(background_rng, -math.pi, math.pi),
        illumination_angle=_uniform(illumination_rng, -math.pi, math.pi),
        illumination_strength=_uniform(
            illumination_rng, *config.illumination_strength_range
        ),
        brightness=_uniform(sensor_rng, *config.brightness_range),
        contrast=_uniform(sensor_rng, *config.contrast_range),
        saturation=_uniform(sensor_rng, *config.saturation_range),
        noise_std=_uniform(sensor_rng, *config.noise_std_range),
        blur_sigma=_uniform(sensor_rng, *config.blur_sigma_range),
        renderer_version=config.renderer_version,
    )


def _normalized_grid(side: int) -> Tuple[torch.Tensor, torch.Tensor]:
    values = torch.linspace(-1.0, 1.0, side, dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return xx, yy


def _background(manifest: VisualManifest, side: int) -> torch.Tensor:
    xx, yy = _normalized_grid(side)
    family = manifest.background_family_id % 4
    frequency = manifest.background_frequency * math.pi
    phase = manifest.background_phase
    if family == 0:
        pattern = 0.5 + 0.25 * (xx + yy)
    elif family == 1:
        pattern = 0.5 + 0.5 * torch.sin(frequency * xx + phase)
    elif family == 2:
        pattern = 0.5 + 0.5 * torch.sin(frequency * xx + phase) * torch.sin(
            frequency * yy - phase
        )
    else:
        radius = torch.sqrt(xx.square() + yy.square())
        pattern = 0.5 + 0.5 * torch.cos(frequency * radius + phase)
    pattern = pattern.clamp(0.0, 1.0)
    base = torch.tensor(
        _BACKGROUND_PALETTE[manifest.background_color_id], dtype=torch.float32
    )[:, None, None]
    accent = torch.tensor(
        _BACKGROUND_PALETTE[manifest.background_accent_id], dtype=torch.float32
    )[:, None, None]
    return (base * (0.78 + 0.22 * pattern) + accent * (0.08 * pattern)).clamp(0, 1)


def _object_rgb(
    style: ObjectStyle,
    center: torch.Tensor,
    side: int,
) -> torch.Tensor:
    xx, yy = _normalized_grid(side)
    scale = 0.5 * max(side - 1, 1)
    local_x = xx * scale - (float(center[0]) - scale)
    local_y = yy * scale - (float(center[1]) - scale)
    frequency = style.frequency
    if style.texture_id == 0:
        pattern = torch.full_like(local_x, 0.88)
    elif style.texture_id == 1:
        pattern = 0.62 + 0.28 * torch.sin(frequency * local_x + style.phase)
    elif style.texture_id == 2:
        pattern = 0.66 + 0.24 * torch.sign(
            torch.sin(frequency * local_x + style.phase)
            * torch.sin(frequency * local_y - style.phase)
        )
    else:
        radius = torch.sqrt(local_x.square() + local_y.square())
        pattern = 0.65 + 0.27 * torch.cos(frequency * radius + style.phase)
    pattern = pattern.clamp(0.2, 1.0)
    base = torch.tensor(_PALETTE[style.color_id], dtype=torch.float32)[:, None, None]
    accent = torch.tensor(
        _PALETTE[style.accent_color_id], dtype=torch.float32
    )[:, None, None]
    return (base * pattern + accent * (1.0 - pattern) * 0.45).clamp(0, 1)


def _apply_illumination(rgb: torch.Tensor, manifest: VisualManifest) -> torch.Tensor:
    side = rgb.shape[-1]
    xx, yy = _normalized_grid(side)
    direction = math.cos(manifest.illumination_angle) * xx + math.sin(
        manifest.illumination_angle
    ) * yy
    field = 1.0 + manifest.illumination_strength * direction
    return rgb * field[None, None]


def _apply_sensor_model(rgb: torch.Tensor, manifest: VisualManifest) -> torch.Tensor:
    transformed = adjust_brightness(rgb, manifest.brightness)
    transformed = adjust_contrast(transformed, manifest.contrast)
    transformed = adjust_saturation(transformed, manifest.saturation)
    transformed = gaussian_blur2d(
        transformed,
        kernel_size=(5, 5),
        sigma=(manifest.blur_sigma, manifest.blur_sigma),
    )
    if manifest.noise_std > 0.0:
        generator = _generator(manifest.rng_seeds["sensor_noise"])
        noise = torch.randn(
            transformed.shape,
            dtype=transformed.dtype,
            generator=generator,
        ) * manifest.noise_std
        transformed = transformed + noise
    return transformed.clamp(0.0, 1.0)


def render_rgb_layers(
    masks: torch.Tensor,
    centers: torch.Tensor,
    alive: torch.Tensor,
    object_names: Sequence[str],
    physical_family_id: str,
    manifest: VisualManifest,
    *,
    depth_order: Sequence[int] | None = None,
    allow_occlusion: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Composite generic object layers and return RGB plus visible masks.

    ``depth_order`` is back-to-front.  Canonical input masks remain amodal;
    returned visible masks remove pixels covered by later layers.
    """

    names = tuple(object_names)
    if manifest.physical_family_id != physical_family_id:
        raise ValueError("visual and physical family IDs do not match")
    if tuple(style.object_name for style in manifest.object_styles) != names:
        raise ValueError("visual object styles do not match physical object order")
    if any(tensor.device.type != "cpu" for tensor in (masks, centers, alive)):
        raise ValueError(
            "RGB-v1 rendering is CPU-only; render first, then transfer frames to "
            "the training device"
        )
    if masks.ndim != 4 or masks.shape[1] != len(manifest.object_styles):
        raise ValueError("expected instance masks shaped [T,K,H,W]")
    frames, object_count, side, width = masks.shape
    if centers.shape != (frames, object_count, 2):
        raise ValueError("centers must be shaped [T,K,2]")
    if alive.shape != (frames, object_count):
        raise ValueError("alive must be shaped [T,K]")
    order = tuple(range(object_count)) if depth_order is None else tuple(depth_order)
    if sorted(order) != list(range(object_count)):
        raise ValueError("depth_order must be a permutation of object indices")
    effective_masks = masks & alive[:, :, None, None]
    overlap = effective_masks.sum(dim=1)
    if not allow_occlusion and bool((overlap > 1).any()):
        raise ValueError("RGB-v1 does not define an occlusion order")

    if side != width:
        raise ValueError("RGB-v1 requires square frames")
    background = _background(manifest, side)
    rgb = background.unsqueeze(0).repeat(frames, 1, 1, 1)
    styles = {style.object_name: style for style in manifest.object_styles}
    for object_index in order:
        style = styles[names[object_index]]
        for frame in range(frames):
            if not bool(alive[frame, object_index]):
                continue
            mask = effective_masks[frame, object_index]
            color = _object_rgb(style, centers[frame, object_index], side)
            rgb[frame] = torch.where(mask.unsqueeze(0), color, rgb[frame])
    rgb = _apply_illumination(rgb, manifest)
    rgb = _apply_sensor_model(rgb, manifest)

    visible = torch.zeros_like(masks)
    covered = torch.zeros((frames, side, side), dtype=torch.bool)
    for object_index in reversed(order):
        visible[:, object_index] = effective_masks[:, object_index] & ~covered
        covered |= effective_masks[:, object_index]
    return rgb, visible


def render_rgb_episode(
    episode: CalibrationProbeEpisode,
    manifest: VisualManifest,
) -> torch.Tensor:
    """Render `[T,3,H,W]` observations while preserving physical masks."""

    rgb, _ = render_rgb_layers(
        episode.instance_masks,
        episode.centers,
        episode.alive,
        episode.object_names,
        episode.manifest.family_id,
        manifest,
    )
    return rgb


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable hash for a contiguous CPU tensor."""

    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


__all__ = [
    "ObjectStyle",
    "VisualDomainConfig",
    "VisualManifest",
    "make_visual_manifest",
    "render_rgb_episode",
    "render_rgb_layers",
    "tensor_sha256",
]
