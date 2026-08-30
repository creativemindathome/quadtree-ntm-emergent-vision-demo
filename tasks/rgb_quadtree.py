"""Fast, budgeted RGB quadtree observations for the unchanged v2 model.

The adapter is deliberately outside :class:`QuadtreeWorldModelV2`.  It turns
an RGB frame into the same sparse ``heap_indices``/``memory`` contract used by
the original grayscale environment, while exposing richer sensor rows:

``RGB mean | RGB maximum | RGB variance | valid fraction | y | x | scale``.

Statistics are computed once per complete level with tensor reductions. The
default exposes a complete candidate tree so the model's differentiable
STOP/SPLIT head, not this adapter, is responsible for learned sparsity.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ntm.quadtree_memory import address_to_bounds


RGB_MEMORY_FIELDS = (
    "red_mean", "green_mean", "blue_mean",
    "red_maximum", "green_maximum", "blue_maximum",
    "red_variance", "green_variance", "blue_variance",
    "valid_fraction", "center_row", "center_col", "scale",
)


@dataclass(frozen=True)
class RGBQuadtreeConfig:
    image_size: int = 192
    canvas_size: int = 256
    max_depth: int = 8
    active_depth: int = 4
    split_variance: float = 0.0025
    max_nodes: Optional[int] = 341
    allocation_mode: str = "complete"

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.image_size > self.canvas_size:
            raise ValueError("image_size must lie in (0, canvas_size]")
        if self.canvas_size & (self.canvas_size - 1):
            raise ValueError("canvas_size must be a power of two")
        if 2 ** self.max_depth != self.canvas_size:
            raise ValueError("max_depth must end in one-pixel canvas cells")
        if not 0 <= self.active_depth <= self.max_depth:
            raise ValueError("active_depth must lie in [0, max_depth]")
        if self.split_variance < 0:
            raise ValueError("split_variance must be non-negative")
        if self.max_nodes is not None and (
            self.max_nodes < 1 or (self.max_nodes - 1) % 4
        ):
            raise ValueError("max_nodes must be 1 + 4*k so child groups remain complete")
        if self.allocation_mode not in ("complete", "variance_budgeted", "learned_frontier"):
            raise ValueError(
                "allocation_mode must be complete, variance_budgeted, or learned_frontier"
            )
        complete_nodes = (4 ** (self.active_depth + 1) - 1) // 3
        if self.allocation_mode == "complete" and (
            self.max_nodes is None or self.max_nodes < complete_nodes
        ):
            raise ValueError("complete mode requires capacity for every candidate node")


def _level_statistics(
    padded: torch.Tensor,
    depth: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return channel mean/max/variance as ``[blocks, blocks, 3]``."""
    channels, side, _ = padded.shape
    blocks = 2 ** depth
    cell = side // blocks
    patches = padded.reshape(channels, blocks, cell, blocks, cell)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(blocks, blocks, channels, -1)
    return (
        patches.mean(dim=-1),
        patches.amax(dim=-1),
        patches.var(dim=-1, unbiased=False),
    )


def _level_valid_fraction(valid: torch.Tensor, depth: int) -> torch.Tensor:
    side = valid.shape[-1]
    blocks = 2 ** depth
    cell = side // blocks
    return valid.reshape(blocks, cell, blocks, cell).permute(0, 2, 1, 3).reshape(
        blocks, blocks, -1,
    ).mean(dim=-1)


def _address_grid_position(address: int) -> Tuple[int, int, int]:
    """Decode an address into ``(depth, grid_y, grid_x)`` without pixels."""
    quadrants: List[int] = []
    cursor = address
    while cursor:
        quadrants.append((cursor - 1) % 4)
        cursor = (cursor - 1) // 4
    x = y = 0
    for quadrant in reversed(quadrants):
        x = 2 * x + (quadrant & 1)
        y = 2 * y + ((quadrant >> 1) & 1)
    return len(quadrants), y, x


def build_rgb_quadtree_sample(
    image: torch.Tensor,
    config: RGBQuadtreeConfig,
    addresses: Optional[Sequence[int] | torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Sample RGB region statistics at explicit or adapter-selected addresses.

    ``addresses`` is the model-controlled path: the adapter performs no
    variance ranking and merely exposes statistics from the full RGB frame at
    the requested quadtree locations.  This allows a learned frontier to reach
    ``max_depth`` without materializing the complete depth-8 tree.
    """
    expected = (3, config.image_size, config.image_size)
    if tuple(image.shape) != expected:
        raise ValueError(f"expected image shape {expected}, got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError("RGB image must be floating point")

    padded = torch.zeros(
        3, config.canvas_size, config.canvas_size, dtype=image.dtype, device=image.device,
    )
    padded[:, :config.image_size, :config.image_size] = image
    valid = torch.zeros(
        config.canvas_size, config.canvas_size, dtype=image.dtype, device=image.device,
    )
    valid[:config.image_size, :config.image_size] = 1.0

    level_stats = []
    valid_fractions = []
    requested_addresses = None
    if addresses is not None:
        requested_addresses = sorted(
            {int(value) for value in (
                addresses.detach().cpu().tolist()
                if isinstance(addresses, torch.Tensor) else addresses
            )},
            key=lambda item: (_address_grid_position(item)[0], item),
        )
        if not requested_addresses or requested_addresses[0] != 0:
            raise ValueError("explicit addresses must contain the root")
        requested_depth = max(_address_grid_position(item)[0] for item in requested_addresses)
        if requested_depth > config.max_depth:
            raise ValueError("explicit address exceeds max_depth")
    else:
        requested_depth = config.active_depth

    for depth in range(requested_depth + 1):
        level_stats.append(_level_statistics(padded, depth))
        valid_fractions.append(_level_valid_fraction(valid, depth))

    if requested_addresses is not None:
        selected = set(requested_addresses)
    elif config.allocation_mode == "learned_frontier":
        selected = {0}
    elif config.allocation_mode == "complete":
        selected = set(range((4 ** (config.active_depth + 1) - 1) // 3))
    else:
        selected = {0}
        root_score = float(level_stats[0][2][0, 0].amax().detach().cpu())
        frontier = [(-root_score, 0)]
        if config.max_nodes is None:
            raise ValueError("variance_budgeted mode requires max_nodes")
        while frontier and len(selected) + 4 <= config.max_nodes:
            negative_score, address = heapq.heappop(frontier)
            score = -negative_score
            depth, _, _ = _address_grid_position(address)
            if depth >= config.active_depth or score <= config.split_variance:
                continue
            for quadrant in range(4):
                child = 4 * address + 1 + quadrant
                selected.add(child)
                child_depth, child_y, child_x = _address_grid_position(child)
                child_score = float(
                    level_stats[child_depth][2][child_y, child_x].amax().detach().cpu()
                )
                heapq.heappush(frontier, (-child_score, child))

    # Diagnostic/bootstrap labels only; these do not select the candidate tree.
    split_addresses = set()
    for address in selected:
        depth, grid_y, grid_x = _address_grid_position(address)
        if (
            depth < config.active_depth
            and float(level_stats[depth][2][grid_y, grid_x].amax().detach().cpu())
            > config.split_variance
        ):
            split_addresses.add(address)

    addresses = sorted(selected, key=lambda item: (_address_grid_position(item)[0], item))
    row_for_address = {address: row for row, address in enumerate(addresses)}
    rows = []
    parent_rows = []
    depths = []
    bounds = []
    for address in addresses:
        depth, grid_y, grid_x = _address_grid_position(address)
        mean, maximum, variance = (
            values[grid_y, grid_x] for values in level_stats[depth]
        )
        valid_fraction = valid_fractions[depth][grid_y, grid_x].reshape(1)
        decoded_depth, x0, y0, size = address_to_bounds(address, config.canvas_size)
        geometry = image.new_tensor((
            (y0 + size / 2) / config.canvas_size,
            (x0 + size / 2) / config.canvas_size,
            size / config.canvas_size,
        ))
        rows.append(torch.cat((mean, maximum, variance, valid_fraction, geometry)))
        parent = -1 if address == 0 else row_for_address[(address - 1) // 4]
        parent_rows.append(parent)
        depths.append(decoded_depth)
        bounds.append((x0, y0, size))

    heap_indices = torch.tensor(addresses, dtype=torch.long, device=image.device)
    split_targets = torch.tensor(
        [address in split_addresses for address in addresses],
        dtype=image.dtype,
        device=image.device,
    )
    # Auxiliary current-frame supervision only.  The model-visible sensor row
    # above retains all three channels independently; this luminance raster is
    # never fed back as an observation.
    auxiliary_luminance_target = (
        0.2126 * image[0:1] + 0.7152 * image[1:2] + 0.0722 * image[2:3]
    )
    return {
        "image": auxiliary_luminance_target,
        "auxiliary_luminance_target": auxiliary_luminance_target,
        "rgb_image": image,
        "padded_image": padded,
        "valid_mask": valid.unsqueeze(0),
        "memory": torch.stack(rows),
        "heap_indices": heap_indices,
        "parent_rows": torch.tensor(parent_rows, dtype=torch.long, device=image.device),
        "split_targets": split_targets,
        "depths": torch.tensor(depths, dtype=torch.long, device=image.device),
        "bounds": torch.tensor(bounds, dtype=torch.long, device=image.device),
        "leaf_mask": ~split_targets.bool(),
    }


__all__ = ["RGB_MEMORY_FIELDS", "RGBQuadtreeConfig", "build_rgb_quadtree_sample"]
