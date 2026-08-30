"""Exact-pointer quadtree memory primitives for the second model iteration.

This module deliberately does *not* implement Neural Turing Machine content
addressing.  A quadtree already gives every spatial cell a stable logical
address, so reads use exact address arithmetic and never normalize over
unrelated rows.  Spatial rows persist as the observed topology changes, while
a small set of episode rows carries information that is not tied to one cell
(for example, an episode-constant collision law).

The module is intentionally self-contained and consumes the sparse sample
dictionaries produced by :mod:`tasks.quadtree`.  Its soft tree renderer is a
mechanism primitive rather than a complete future-dynamics model: it makes the
STOP/SPLIT decision differentiable so later predictive losses can train the
allocation policy directly.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ntm.quadtree_memory import address_to_bounds


@dataclass(frozen=True)
class V2ModelConfig:
    """Dimensions of the compact exact-pointer memory updater."""

    canvas_size: int = 128
    image_size: int = 128
    max_depth: int = 7
    sensor_features: int = 7
    payload_dim: int = 16
    episode_slots: int = 4
    path_dim: int = 8
    hidden_dim: int = 32
    prediction_slots: int = 0
    episode_attention: bool = False
    controller_norm: bool = False
    prediction_split_bias: float = 0.0
    split_prior_probability: float = 0.2
    rgb_flip_prior_probability: float = 0.01
    rgb_bit_output: bool = False

    def __post_init__(self) -> None:
        values = (
            self.canvas_size,
            self.image_size,
            self.max_depth,
            self.sensor_features,
            self.payload_dim,
            self.path_dim,
            self.hidden_dim,
        )
        if any(value <= 0 for value in values) or self.episode_slots < 0:
            raise ValueError("all V2 model dimensions must be positive")
        if self.canvas_size & (self.canvas_size - 1):
            raise ValueError("canvas_size must be a power of two")
        if 2 ** self.max_depth != self.canvas_size:
            raise ValueError("max_depth must end in one-pixel canvas cells")
        if self.image_size > self.canvas_size:
            raise ValueError("image_size cannot exceed canvas_size")
        if self.prediction_slots < 0 or self.prediction_slots > self.episode_slots:
            raise ValueError("prediction_slots must lie in [0, episode_slots]")
        if not math.isfinite(self.prediction_split_bias):
            raise ValueError("prediction_split_bias must be finite")
        if not 0.0 < self.split_prior_probability < 1.0:
            raise ValueError("split_prior_probability must lie strictly between zero and one")
        if not 0.0 < self.rgb_flip_prior_probability < 0.5:
            raise ValueError("rgb_flip_prior_probability must lie between zero and one half")


@dataclass
class V2MemoryState:
    """Persistent sparse spatial rows plus fixed episode-level rows."""

    spatial_addresses: torch.Tensor
    spatial_payloads: torch.Tensor
    episode_payloads: torch.Tensor

    def __post_init__(self) -> None:
        if self.spatial_addresses.ndim != 1:
            raise ValueError("spatial_addresses must have shape [rows]")
        if self.spatial_payloads.ndim != 2:
            raise ValueError("spatial_payloads must have shape [rows, payload]")
        if self.episode_payloads.ndim != 2:
            raise ValueError("episode_payloads must have shape [slots, payload]")
        if self.spatial_addresses.shape[0] != self.spatial_payloads.shape[0]:
            raise ValueError("spatial address and payload counts must match")
        if self.spatial_payloads.shape[1] != self.episode_payloads.shape[1]:
            raise ValueError("spatial and episode payload widths must match")
        if self.spatial_addresses.dtype != torch.long:
            raise ValueError("spatial addresses must use torch.long")
        if self.spatial_addresses.numel() != torch.unique(
                self.spatial_addresses).numel():
            raise ValueError("spatial addresses must be unique")

    @classmethod
    def empty(
            cls,
            config: V2ModelConfig,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
    ) -> "V2MemoryState":
        """Construct an empty spatial memory and zeroed episode memory."""
        return cls(
            spatial_addresses=torch.empty(0, dtype=torch.long, device=device),
            spatial_payloads=torch.empty(
                0, config.payload_dim, dtype=dtype, device=device,
            ),
            episode_payloads=torch.zeros(
                config.episode_slots, config.payload_dim, dtype=dtype, device=device,
            ),
        )

    def detach(self) -> "V2MemoryState":
        """Detach recurrent values at an explicit truncated-BPTT boundary."""
        return V2MemoryState(
            spatial_addresses=self.spatial_addresses,
            spatial_payloads=self.spatial_payloads.detach(),
            episode_payloads=self.episode_payloads.detach(),
        )

    def detached(self) -> "V2MemoryState":
        """Backward-compatible spelling matching the first model iteration."""
        return self.detach()

    def reset_episode(self) -> "V2MemoryState":
        """Clear only global episode rows, preserving exact spatial memory."""
        return V2MemoryState(
            spatial_addresses=self.spatial_addresses,
            spatial_payloads=self.spatial_payloads,
            episode_payloads=torch.zeros_like(self.episode_payloads),
        )
@dataclass
class ExactTopologyReads:
    """Exact structural reads for a batch of logical query addresses."""

    same: torch.Tensor
    parent: torch.Tensor
    children_mean: torch.Tensor
    siblings_mean: torch.Tensor
    episode_mean: torch.Tensor
    same_found: torch.Tensor
    parent_found: torch.Tensor
    child_count: torch.Tensor
    sibling_count: torch.Tensor


@dataclass
class HierarchicalWeights:
    """Soft reach, split, and stop mass on one candidate quadtree."""

    reach: torch.Tensor
    split_probability: torch.Tensor
    stop: torch.Tensor
    can_split: torch.Tensor


@dataclass
class V2ObservationOutput:
    """Persistent state and differentiable reconstruction for one observation."""

    state: V2MemoryState
    current_payloads: torch.Tensor
    value_logits: torch.Tensor
    split_logits: torch.Tensor
    hierarchy: HierarchicalWeights
    frame_probabilities: torch.Tensor
    frame_logits: torch.Tensor
    expected_nodes: torch.Tensor
    episode_attention_weights: Optional[torch.Tensor]


@dataclass
class V2PredictionOutput:
    """Future tree decoded from current memory without target topology."""

    state: V2MemoryState
    candidate_addresses: torch.Tensor
    predicted_payloads: torch.Tensor
    value_logits: torch.Tensor
    split_logits: torch.Tensor
    hierarchy: HierarchicalWeights
    frame_probabilities: torch.Tensor
    frame_logits: torch.Tensor
    expected_nodes: torch.Tensor
    horizon: int
    slot_frame_probabilities: Optional[torch.Tensor]
    slot_frame_logits: Optional[torch.Tensor]
    episode_attention_weights: Optional[torch.Tensor]
    rgb_bit_logits: Optional[torch.Tensor] = None


@dataclass
class HardTreeSelection:
    """One hard, structurally valid subtree of a supplied candidate tree."""

    addresses: torch.Tensor
    leaf_mask: torch.Tensor
    split_mask: torch.Tensor
    depths: torch.Tensor


@lru_cache(maxsize=64)
def _cached_path_bits(address_tuple: Tuple[int, ...], max_depth: int) -> torch.Tensor:
    """Build immutable CPU path features once per candidate topology."""
    rows = []
    for raw_address in address_tuple:
        cursor = int(raw_address)
        quadrants = []
        while cursor:
            quadrants.append((cursor - 1) % 4)
            cursor = (cursor - 1) // 4
        quadrants.reverse()
        if len(quadrants) > max_depth:
            raise ValueError("address exceeds configured maximum depth")
        row = [0.0] * (3 * max_depth)
        for level, quadrant in enumerate(quadrants):
            row[3 * level] = float(quadrant & 1)
            row[3 * level + 1] = float((quadrant >> 1) & 1)
            row[3 * level + 2] = 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def _address_tuple(addresses: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(value) for value in addresses.detach().cpu().tolist())


def _path_bits(addresses: torch.Tensor, max_depth: int) -> torch.Tensor:
    """Encode addresses as ``[x_bit, y_bit, valid]`` at every level."""
    return _cached_path_bits(_address_tuple(addresses), max_depth).to(addresses.device)


@lru_cache(maxsize=64)
def _cached_tree_structure(
        address_tuple: Tuple[int, ...],
        max_depth: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    row_for_address = {address: row for row, address in enumerate(address_tuple)}
    depths = []
    parent_rows = []
    can_split = []
    for address in address_tuple:
        cursor = address
        depth = 0
        while cursor:
            cursor = (cursor - 1) // 4
            depth += 1
        depths.append(depth)
        parent_rows.append(-1 if address == 0 else row_for_address.get((address - 1) // 4, -1))
        complete = all(4 * address + offset in row_for_address for offset in (1, 2, 3, 4))
        can_split.append(complete and depth < max_depth)
    return (
        torch.tensor(depths, dtype=torch.long),
        torch.tensor(parent_rows, dtype=torch.long),
        torch.tensor(can_split, dtype=torch.bool),
        row_for_address.get(0, -1),
    )


def _depths(addresses: torch.Tensor) -> torch.Tensor:
    """Return exact heap depth for every logical address."""
    depths, _, _, _ = _cached_tree_structure(_address_tuple(addresses), 10 ** 6)
    return depths.to(addresses.device)


def _exact_gather(
        query_addresses: torch.Tensor,
        state: V2MemoryState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather exact rows from an unordered sparse state without softmax.

    ``query_addresses`` may have any leading shape.  Missing or negative
    addresses produce zero payloads and a false mask.  Sorting the source rows
    makes the result invariant to their physical order while retaining
    gradients to the gathered payloads.
    """
    shape = query_addresses.shape
    flat_queries = query_addresses.reshape(-1)
    payload_dim = state.spatial_payloads.shape[-1]
    if state.spatial_addresses.numel() == 0:
        values = state.spatial_payloads.new_zeros(flat_queries.shape[0], payload_dim)
        found = torch.zeros(flat_queries.shape[0], dtype=torch.bool, device=flat_queries.device)
        return values.reshape(*shape, payload_dim), found.reshape(shape)

    order = torch.argsort(state.spatial_addresses)
    stored_addresses = state.spatial_addresses[order]
    stored_payloads = state.spatial_payloads[order]
    safe_queries = flat_queries.clamp_min(0)
    positions = torch.searchsorted(stored_addresses, safe_queries)
    in_range = positions < stored_addresses.numel()
    safe_positions = positions.clamp_max(stored_addresses.numel() - 1)
    found = (
        in_range
        & (flat_queries >= 0)
        & (stored_addresses[safe_positions] == safe_queries)
    )
    gathered = stored_payloads[safe_positions]
    gathered = gathered * found.to(gathered.dtype).unsqueeze(-1)
    return gathered.reshape(*shape, payload_dim), found.reshape(shape)


def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=-1)
    total = (values * mask.to(values.dtype).unsqueeze(-1)).sum(dim=-2)
    mean = total / count.clamp_min(1).to(values.dtype).unsqueeze(-1)
    return mean, count


class ExactTopologyReader(nn.Module):
    """Read only exact quadtree relations; unrelated rows cannot dilute reads."""

    def forward(
            self,
            query_addresses: torch.Tensor,
            state: V2MemoryState,
    ) -> ExactTopologyReads:
        if query_addresses.ndim != 1:
            raise ValueError("query addresses must have shape [queries]")
        same, same_found = _exact_gather(query_addresses, state)

        parent_addresses = torch.where(
            query_addresses > 0,
            (query_addresses - 1) // 4,
            torch.full_like(query_addresses, -1),
        )
        parent, parent_found = _exact_gather(parent_addresses, state)

        offsets = torch.arange(1, 5, device=query_addresses.device)
        child_addresses = 4 * query_addresses[:, None] + offsets[None, :]
        children, child_mask = _exact_gather(child_addresses, state)
        children_mean, child_count = _masked_mean(children, child_mask)

        sibling_addresses = 4 * parent_addresses[:, None] + offsets[None, :]
        siblings, sibling_mask = _exact_gather(sibling_addresses, state)
        sibling_mask = (
            sibling_mask
            & (parent_addresses[:, None] >= 0)
            & (sibling_addresses != query_addresses[:, None])
        )
        siblings_mean, sibling_count = _masked_mean(siblings, sibling_mask)

        episode = state.episode_payloads.mean(dim=0, keepdim=True)
        episode = episode.expand(query_addresses.shape[0], -1)
        return ExactTopologyReads(
            same=same,
            parent=parent,
            children_mean=children_mean,
            siblings_mean=siblings_mean,
            episode_mean=episode,
            same_found=same_found,
            parent_found=parent_found,
            child_count=child_count,
            sibling_count=sibling_count,
        )


def hierarchical_reach_stop_weights(
        addresses: torch.Tensor,
        split_logits: torch.Tensor,
        max_depth: int,
) -> HierarchicalWeights:
    """Compute differentiable reach and stop mass on a candidate tree.

    A node can split only when all four children are in the candidate address
    set and it is shallower than ``max_depth``.  Incomplete groups and terminal
    depth are therefore forced to STOP without a special target label.
    """
    if addresses.ndim != 1 or split_logits.ndim != 1:
        raise ValueError("addresses and split_logits must be one-dimensional")
    if addresses.shape[0] != split_logits.shape[0]:
        raise ValueError("addresses and split logits must have equal lengths")
    if addresses.numel() == 0 or not bool(torch.any(addresses == 0)):
        raise ValueError("candidate tree must contain root address 0")
    if addresses.numel() != torch.unique(addresses).numel():
        raise ValueError("candidate addresses must be unique")

    address_tuple = _address_tuple(addresses)
    depth_tensor, parent_rows, can_split, root_row = _cached_tree_structure(
        address_tuple, max_depth,
    )
    depth_tensor = depth_tensor.to(addresses.device)
    parent_rows = parent_rows.to(addresses.device)
    can_split = can_split.to(addresses.device)
    split_probability = torch.sigmoid(split_logits) * can_split.to(split_logits.dtype)

    reach = split_logits.new_zeros(split_logits.shape)
    reach = reach.index_fill(0, torch.tensor([root_row], device=addresses.device), 1.0)
    for depth in range(1, max_depth + 1):
        rows = torch.nonzero(depth_tensor == depth, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        parents = parent_rows[rows]
        valid = parents >= 0
        rows = rows[valid]
        parents = parents[valid]
        reach = reach.index_copy(0, rows, reach[parents] * split_probability[parents])
    stop = reach * (1.0 - split_probability)
    return HierarchicalWeights(
        reach=reach,
        split_probability=split_probability,
        stop=stop,
        can_split=can_split,
    )


def expected_node_count(hierarchy: HierarchicalWeights) -> torch.Tensor:
    """Expected visited rows under the hierarchy's Bernoulli split gates."""
    return hierarchy.reach.sum()


def soft_hierarchical_rasterize(
        addresses: torch.Tensor,
        value_logits: torch.Tensor,
        hierarchy: HierarchicalWeights,
        image_size: int,
        canvas_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a probability mixture whose split gates remain differentiable."""
    if addresses.shape != value_logits.shape or addresses.shape != hierarchy.stop.shape:
        raise ValueError("address, value, and hierarchy row counts must match")
    values = torch.sigmoid(value_logits)
    corners, signs, valid_rows = _cached_raster_corners(
        _address_tuple(addresses), image_size, canvas_size,
    )
    corners = corners.to(addresses.device)
    signs = signs.to(device=addresses.device, dtype=value_logits.dtype)
    valid_rows = valid_rows.to(addresses.device)
    contributions = (hierarchy.stop[valid_rows] * values[valid_rows])[:, None] * signs
    difference = value_logits.new_zeros((image_size + 1) * (image_size + 1))
    difference = difference.index_add(0, corners.reshape(-1), contributions.reshape(-1))
    difference = difference.reshape(image_size + 1, image_size + 1)
    canvas = difference.cumsum(dim=0).cumsum(dim=1)[:image_size, :image_size].unsqueeze(0)
    probabilities = canvas.clamp(1e-6, 1.0 - 1e-6)
    logits = torch.logit(probabilities)
    return probabilities, logits


@lru_cache(maxsize=64)
def _cached_raster_corners(
        address_tuple: Tuple[int, ...],
        image_size: int,
        canvas_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rectangle-difference indices for a static candidate topology."""
    corners = []
    valid_rows = []
    stride = image_size + 1
    for row, address in enumerate(address_tuple):
        _, x0, y0, size = address_to_bounds(address, canvas_size)
        x1 = min(x0 + size, image_size)
        y1 = min(y0 + size, image_size)
        if x0 < image_size and y0 < image_size and x1 > x0 and y1 > y0:
            corners.append((y0 * stride + x0, y0 * stride + x1,
                            y1 * stride + x0, y1 * stride + x1))
            valid_rows.append(row)
    return (
        torch.tensor(corners, dtype=torch.long),
        torch.tensor((1.0, -1.0, -1.0, 1.0)).expand(len(corners), -1).clone(),
        torch.tensor(valid_rows, dtype=torch.long),
    )


def depth_class_balanced_split_bce(
        split_logits: torch.Tensor,
        targets: torch.Tensor,
        depths: torch.Tensor,
        max_depth: int,
) -> torch.Tensor:
    """Average depths and classes equally, excluding forced terminal STOPs."""
    if not (split_logits.shape == targets.shape == depths.shape):
        raise ValueError("split logits, targets, and depths must have equal shapes")
    losses = []
    for depth in range(max_depth):
        at_depth = depths == depth
        if not bool(at_depth.any()):
            continue
        positive = at_depth & (targets >= 0.5)
        negative = at_depth & (targets < 0.5)
        class_losses = []
        if bool(positive.any()):
            class_losses.append(F.softplus(-split_logits[positive]).mean())
        if bool(negative.any()):
            class_losses.append(F.softplus(split_logits[negative]).mean())
        losses.append(torch.stack(class_losses).mean())
    if not losses:
        return split_logits.sum() * 0.0
    return torch.stack(losses).mean()


def balanced_pixel_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary pixel BCE with equal foreground/background class contribution."""
    if logits.shape != targets.shape:
        raise ValueError("pixel logits and targets must have equal shapes")
    positive = targets >= 0.5
    negative = ~positive
    losses = []
    if bool(positive.any()):
        losses.append(F.softplus(-logits[positive]).mean())
    if bool(negative.any()):
        losses.append(F.softplus(logits[negative]).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def select_hard_tree(
        candidate_addresses: torch.Tensor,
        split_logits: torch.Tensor,
        max_depth: int,
        threshold: float = 0.5,
        minimum_depth: int = 0,
        max_nodes: int = 4096,
) -> HardTreeSelection:
    """Select a hard valid tree with complete child groups under a node cap."""
    if candidate_addresses.ndim != 1 or split_logits.ndim != 1:
        raise ValueError("candidate addresses and split logits must be one-dimensional")
    if candidate_addresses.shape[0] != split_logits.shape[0]:
        raise ValueError("candidate addresses and split logits must have equal lengths")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0 <= minimum_depth <= max_depth:
        raise ValueError("minimum_depth must lie in [0, max_depth]")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")

    candidate = set(int(value) for value in candidate_addresses.detach().cpu().tolist())
    if 0 not in candidate:
        raise ValueError("candidate tree must contain root address 0")
    if len(candidate) != candidate_addresses.numel():
        raise ValueError("candidate addresses must be unique")
    probability_for_address = {
        int(address): float(probability)
        for address, probability in zip(
            candidate_addresses.detach().cpu().tolist(),
            torch.sigmoid(split_logits).detach().cpu().tolist(),
        )
    }

    selected = {0}
    split = set()
    frontier = [0]
    for depth in range(max_depth + 1):
        if not frontier or depth == max_depth:
            break
        proposals = []
        for address in frontier:
            children = tuple(4 * address + offset for offset in (1, 2, 3, 4))
            if not all(child in candidate for child in children):
                continue
            forced = depth < minimum_depth
            probability = probability_for_address[address]
            if forced or probability >= threshold:
                proposals.append((forced, probability, address, children))

        available_groups = max(0, (max_nodes - len(selected)) // 4)
        proposals.sort(key=lambda item: (-int(item[0]), -item[1], item[2]))
        proposals = proposals[:available_groups]
        next_frontier = []
        for _, _, address, children in proposals:
            split.add(address)
            selected.update(children)
            next_frontier.extend(children)
        frontier = sorted(next_frontier)

    ordered = sorted(selected, key=lambda address: (address_to_bounds(address, 2 ** max_depth)[0], address))
    device = candidate_addresses.device
    addresses = torch.tensor(ordered, dtype=torch.long, device=device)
    split_mask = torch.tensor([address in split for address in ordered], dtype=torch.bool, device=device)
    return HardTreeSelection(
        addresses=addresses,
        leaf_mask=~split_mask,
        split_mask=split_mask,
        depths=_depths(addresses),
    )


def expand_candidate_addresses(
        current_addresses: torch.Tensor,
        max_depth: int,
        expansion_levels: int = 1,
        max_nodes: int = 4096,
        global_depth: int = 0,
) -> torch.Tensor:
    """Expand current leaves by complete groups without consulting a target.

    The current observed topology is copied exactly.  At each expansion level,
    every eligible leaf receives all four logical children, subject to the
    global node cap.  This supplies plausible support for moving boundaries
    while preserving the invariant that a node has zero or four children.
    """
    if current_addresses.ndim != 1:
        raise ValueError("current_addresses must have shape [rows]")
    if expansion_levels < 0:
        raise ValueError("expansion_levels must be non-negative")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    if not 0 <= global_depth <= max_depth:
        raise ValueError("global_depth must lie in [0, max_depth]")
    current = set(int(value) for value in current_addresses.detach().cpu().tolist())
    if 0 not in current:
        raise ValueError("current topology must contain root address 0")
    if len(current) != current_addresses.numel():
        raise ValueError("current addresses must be unique")
    global_capacity = (4 ** (global_depth + 1) - 1) // 3
    if global_capacity > max_nodes:
        raise ValueError("global-depth support exceeds max_nodes")
    if len(current) > max_nodes:
        raise ValueError("current topology already exceeds max_nodes")

    # A uniform global floor guarantees that a moving object's future location
    # has representational support even when it was background in the current
    # observation.  This is target-independent and therefore cannot leak the
    # realized future topology.
    selected = set(range(global_capacity)) | current
    if len(selected) > max_nodes:
        raise ValueError("current plus global support exceeds max_nodes")
    for _ in range(expansion_levels):
        proposals = []
        for address in sorted(selected):
            depth = address_to_bounds(address, 2 ** max_depth)[0]
            children = tuple(4 * address + offset for offset in (1, 2, 3, 4))
            is_leaf = not any(child in selected for child in children)
            if is_leaf and depth < max_depth:
                proposals.append(children)
        available_groups = max(0, (max_nodes - len(selected)) // 4)
        if available_groups == 0:
            break
        for children in proposals[:available_groups]:
            selected.update(children)

    ordered = sorted(
        selected,
        key=lambda address: (address_to_bounds(address, 2 ** max_depth)[0], address),
    )
    return torch.tensor(ordered, dtype=torch.long, device=current_addresses.device)


class QuadtreeWorldModelV2(nn.Module):
    """Compact observation updater with exact structural and episode memory."""

    def __init__(self, config: Optional[V2ModelConfig] = None):
        super().__init__()
        self.config = config or V2ModelConfig()
        config = self.config
        self.path_encoder = nn.Sequential(
            nn.Linear(3 * config.max_depth, 2 * config.path_dim),
            nn.Tanh(),
            nn.Linear(2 * config.path_dim, config.path_dim),
            nn.Tanh(),
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(config.sensor_features, config.path_dim),
            nn.Tanh(),
        )
        self.reader = ExactTopologyReader()
        self.quadrant_embedding = nn.Parameter(torch.zeros(4, config.payload_dim))

        node_input_dim = 2 * config.path_dim + 5 * config.payload_dim
        self.node_input_norm = (
            nn.LayerNorm(node_input_dim) if config.controller_norm else nn.Identity()
        )
        self.node_core = nn.Linear(node_input_dim, config.hidden_dim)
        self.write_gate = nn.Linear(config.hidden_dim, config.payload_dim)
        self.erase = nn.Linear(config.hidden_dim, config.payload_dim)
        self.add = nn.Linear(config.hidden_dim, config.payload_dim)
        self.split_head = nn.Linear(config.hidden_dim, 1)
        nn.init.constant_(
            self.split_head.bias,
            math.log(config.split_prior_probability / (1.0 - config.split_prior_probability)),
        )
        self.value_head = nn.Linear(config.payload_dim, 1)

        # Prediction uses only exact current memory and a requested horizon.
        # It never accepts the future sample or its oracle address set.
        self.supported_horizons = (1, 4, 8)
        self.horizon_embedding = nn.Embedding(
            len(self.supported_horizons), config.path_dim,
        )
        prediction_input_dim = 2 * config.path_dim + 5 * config.payload_dim
        self.prediction_input_norm = (
            nn.LayerNorm(prediction_input_dim) if config.controller_norm else nn.Identity()
        )
        self.prediction_core = nn.Linear(prediction_input_dim, config.hidden_dim)
        self.prediction_gate = nn.Linear(config.hidden_dim, config.payload_dim)
        self.prediction_erase = nn.Linear(config.hidden_dim, config.payload_dim)
        self.prediction_add = nn.Linear(config.hidden_dim, config.payload_dim)
        self.prediction_split = nn.Linear(config.hidden_dim, 1)
        nn.init.constant_(self.prediction_split.bias, config.prediction_split_bias)
        self.prediction_value = nn.Linear(config.payload_dim, 1)
        self.rgb_bit_head = (
            nn.Linear(config.payload_dim, 24) if config.rgb_bit_output else None
        )
        if self.rgb_bit_head is not None:
            nn.init.normal_(self.rgb_bit_head.weight, std=1e-3)
            nn.init.constant_(
                self.rgb_bit_head.bias,
                math.log(
                    config.rgb_flip_prior_probability
                    / (1.0 - config.rgb_flip_prior_probability)
                ),
            )

        if config.prediction_slots:
            self.slot_prediction_core = nn.Linear(
                3 * config.payload_dim, config.payload_dim,
            )
            self.slot_prediction_value = nn.Linear(config.payload_dim, 1)
            self.slot_prediction_key = nn.Linear(
                2 * config.payload_dim, config.payload_dim, bias=False,
            )
            self.slot_prediction_query = nn.Linear(
                config.payload_dim, config.payload_dim, bias=False,
            )
            self.slot_prediction_logit_scale = nn.Parameter(torch.tensor(1.0))

        self.episode_slot_embedding = nn.Parameter(
            torch.empty(config.episode_slots, config.payload_dim),
        )
        nn.init.normal_(self.episode_slot_embedding, std=0.02)
        self.episode_core = nn.Linear(4 * config.payload_dim, config.hidden_dim)
        self.episode_input_norm = (
            nn.LayerNorm(4 * config.payload_dim) if config.controller_norm else nn.Identity()
        )
        self.episode_gate = nn.Linear(config.hidden_dim, config.payload_dim)
        self.episode_erase = nn.Linear(config.hidden_dim, config.payload_dim)
        self.episode_add = nn.Linear(config.hidden_dim, config.payload_dim)

        if config.episode_attention:
            self.episode_write_query = nn.Linear(config.payload_dim, config.payload_dim, bias=False)
            self.episode_write_key = nn.Linear(config.payload_dim, config.payload_dim, bias=False)
            self.episode_read_key = nn.Linear(config.payload_dim, config.payload_dim, bias=False)
            self.episode_read_value = nn.Linear(config.payload_dim, config.payload_dim, bias=False)
            self.observation_episode_query = nn.Linear(
                config.payload_dim + config.path_dim, config.payload_dim, bias=False,
            )
            self.prediction_episode_query = nn.Linear(
                config.payload_dim + 2 * config.path_dim, config.payload_dim, bias=False,
            )
            self.observation_episode_gate = nn.Linear(
                config.payload_dim + config.path_dim, 1,
            )
            self.prediction_episode_gate = nn.Linear(
                config.payload_dim + 2 * config.path_dim, 1,
            )
            nn.init.zeros_(self.observation_episode_gate.weight)
            nn.init.zeros_(self.prediction_episode_gate.weight)
            nn.init.constant_(self.observation_episode_gate.bias, -2.0)
            nn.init.constant_(self.prediction_episode_gate.bias, -2.0)

    def path_embeddings(self, addresses: torch.Tensor) -> torch.Tensor:
        bits = _path_bits(addresses, self.config.max_depth)
        return self.path_encoder(bits.to(dtype=self.quadrant_embedding.dtype))

    def initial_payloads(
            self,
            addresses: torch.Tensor,
            reads: ExactTopologyReads,
    ) -> torch.Tensor:
        """Choose same row, complete child merge, or parent/quadrant init.

        Priority is exact temporal identity first.  If a coarse row reappears
        after refinement, its four children are averaged.  A newly refined
        child starts from its parent plus an explicit quadrant embedding.
        """
        quadrant = torch.where(
            addresses > 0,
            (addresses - 1) % 4,
            torch.zeros_like(addresses),
        )
        from_parent = torch.tanh(reads.parent + self.quadrant_embedding[quadrant])
        zeros = torch.zeros_like(reads.same)
        base = torch.where(reads.parent_found[:, None], from_parent, zeros)
        base = torch.where((reads.child_count == 4)[:, None], reads.children_mean, base)
        base = torch.where(reads.same_found[:, None], reads.same, base)
        return base

    def _update_episode(
            self,
            previous: torch.Tensor,
            innovations: torch.Tensor,
            payloads: torch.Tensor,
    ) -> torch.Tensor:
        slots = previous.shape[0]
        if slots == 0:
            return previous
        if self.config.episode_attention:
            queries = F.normalize(
                self.episode_write_query(previous + self.episode_slot_embedding), dim=-1,
            )
            keys = F.normalize(self.episode_write_key(payloads), dim=-1)
            scores = queries @ keys.transpose(0, 1) * math.sqrt(self.config.payload_dim)
            object_slots = self.config.prediction_slots or slots
            # Object slots compete for each node before each slot normalizes
            # its evidence. Independent row softmax lets every slot absorb the
            # same object and empirically collapses identity memory.
            object_scores = scores[:object_slots]
            object_weights = torch.softmax(object_scores, dim=0) + 1e-8
            object_weights = object_weights / object_weights.sum(
                dim=-1, keepdim=True,
            ).clamp_min(1e-8)
            if object_slots < slots:
                context_weights = torch.softmax(scores[object_slots:], dim=-1)
                weights = torch.cat((object_weights, context_weights), dim=0)
            else:
                weights = object_weights
            pooled_innovation = weights @ innovations
            pooled_payload = weights @ payloads
        else:
            pooled_innovation = innovations.mean(dim=0, keepdim=True).expand(slots, -1)
            pooled_payload = payloads.mean(dim=0, keepdim=True).expand(slots, -1)
        controller_input = torch.cat((
            previous,
            pooled_innovation,
            pooled_payload,
            self.episode_slot_embedding,
        ), dim=-1)
        hidden = torch.tanh(self.episode_core(self.episode_input_norm(controller_input)))
        gate = torch.sigmoid(self.episode_gate(hidden))
        erase = torch.sigmoid(self.episode_erase(hidden))
        add = torch.tanh(self.episode_add(hidden))
        return previous * (1.0 - gate * erase) + gate * add

    def _read_episode(
            self,
            query_input: torch.Tensor,
            episode_payloads: torch.Tensor,
            prediction: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Read persistent slots selectively while bounding global broadcast."""
        if episode_payloads.shape[0] == 0:
            return query_input.new_zeros((query_input.shape[0], self.config.payload_dim)), None
        if not self.config.episode_attention:
            mean = episode_payloads.mean(dim=0, keepdim=True)
            return mean.expand(query_input.shape[0], -1), None
        query_layer = (
            self.prediction_episode_query if prediction
            else self.observation_episode_query
        )
        gate_layer = (
            self.prediction_episode_gate if prediction
            else self.observation_episode_gate
        )
        queries = F.normalize(query_layer(query_input), dim=-1)
        keys = F.normalize(self.episode_read_key(episode_payloads), dim=-1)
        scores = queries @ keys.transpose(0, 1) * math.sqrt(self.config.payload_dim)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ self.episode_read_value(episode_payloads)
        gate = torch.sigmoid(gate_layer(query_input))
        return gate * context, weights

    @staticmethod
    def _merge_spatial_rows(
            previous: V2MemoryState,
            update_addresses: torch.Tensor,
            update_payloads: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        union = torch.unique(
            torch.cat((previous.spatial_addresses, update_addresses)),
            sorted=True,
        )
        prior_values, prior_found = _exact_gather(union, previous)
        update_state = V2MemoryState(
            spatial_addresses=update_addresses,
            spatial_payloads=update_payloads,
            episode_payloads=previous.episode_payloads,
        )
        update_values, update_found = _exact_gather(union, update_state)
        payloads = torch.where(update_found[:, None], update_values, prior_values)
        if not bool(torch.all(update_found | prior_found)):
            raise RuntimeError("spatial union contains an unresolved address")
        return union, payloads

    def observe(
            self,
            sample: Dict[str, torch.Tensor],
            state: Optional[V2MemoryState] = None,
    ) -> V2ObservationOutput:
        """Update exact spatial rows and persistent episode rows from one frame."""
        device = self.quadrant_embedding.device
        addresses = sample["heap_indices"].to(device=device, dtype=torch.long)
        sensors = sample["memory"].to(device=device, dtype=self.quadrant_embedding.dtype)
        if sensors.ndim != 2 or sensors.shape[1] < self.config.sensor_features:
            raise ValueError("sample memory has the wrong sensor feature width")
        # The entropy-coded mode deliberately exposes only regional mean RGB;
        # the adapter may retain auxiliary diagnostics outside the controller.
        sensors = sensors[:, :self.config.sensor_features]
        if state is None:
            state = V2MemoryState.empty(
                self.config, device=device, dtype=self.quadrant_embedding.dtype,
            )

        reads = self.reader(addresses, state)
        base = self.initial_payloads(addresses, reads)
        path_embedding = self.path_embeddings(addresses)
        episode_context, episode_attention = self._read_episode(
            torch.cat((base, path_embedding), dim=-1),
            state.episode_payloads,
            prediction=False,
        )
        node_input = torch.cat((
            self.sensor_encoder(sensors),
            path_embedding,
            base,
            reads.parent,
            reads.children_mean,
            reads.siblings_mean,
            episode_context,
        ), dim=-1)
        hidden = torch.tanh(self.node_core(self.node_input_norm(node_input)))
        gate = torch.sigmoid(self.write_gate(hidden))
        erase = torch.sigmoid(self.erase(hidden))
        add = torch.tanh(self.add(hidden))
        proposed_payloads = base * (1.0 - gate * erase) + gate * add
        split_logits = self.split_head(hidden).squeeze(-1)
        hierarchy = hierarchical_reach_stop_weights(
            addresses, split_logits, self.config.max_depth,
        )
        # Reach is now the differentiable write-allocation decision, rather
        # than only a visualization weight applied after every row was stored.
        # Future predictive gradients can therefore teach the observation
        # controller whether refining and writing a branch was worth its rate.
        payloads = base + hierarchy.reach[:, None] * (proposed_payloads - base)

        episode_payloads = self._update_episode(
            state.episode_payloads,
            payloads - base,
            payloads,
        )
        spatial_addresses, spatial_payloads = self._merge_spatial_rows(
            state, addresses, payloads,
        )
        new_state = V2MemoryState(
            spatial_addresses=spatial_addresses,
            spatial_payloads=spatial_payloads,
            episode_payloads=episode_payloads,
        )

        value_logits = self.value_head(payloads).squeeze(-1)
        probabilities, frame_logits = soft_hierarchical_rasterize(
            addresses,
            value_logits,
            hierarchy,
            image_size=self.config.image_size,
            canvas_size=self.config.canvas_size,
        )
        return V2ObservationOutput(
            state=new_state,
            current_payloads=payloads,
            value_logits=value_logits,
            split_logits=split_logits,
            hierarchy=hierarchy,
            frame_probabilities=probabilities,
            frame_logits=frame_logits,
            expected_nodes=expected_node_count(hierarchy),
            episode_attention_weights=episode_attention,
        )

    def prediction_candidates(
            self,
            current_addresses: torch.Tensor,
            expansion_levels: int = 1,
            max_nodes: int = 4096,
            global_depth: int = 0,
    ) -> torch.Tensor:
        """Build future candidate support from current topology alone."""
        return expand_candidate_addresses(
            current_addresses,
            max_depth=self.config.max_depth,
            expansion_levels=expansion_levels,
            max_nodes=max_nodes,
            global_depth=global_depth,
        )

    def predict(
            self,
            state: V2MemoryState,
            candidate_addresses: torch.Tensor,
            horizon: int = 1,
    ) -> V2PredictionOutput:
        """Decode a future quadtree from exact memory, never target structure.

        ``candidate_addresses`` must be derived from currently available
        structure, normally via :meth:`prediction_candidates`.  The predicted
        rows form a possible recurrent state for multi-step rollout, while the
        episode rows persist unchanged until another observation updates them.
        """
        if horizon not in self.supported_horizons:
            raise ValueError(
                "horizon must be one of {}".format(self.supported_horizons)
            )
        device = self.quadrant_embedding.device
        addresses = candidate_addresses.to(device=device, dtype=torch.long)
        reads = self.reader(addresses, state)
        base = self.initial_payloads(addresses, reads)
        horizon_index = self.supported_horizons.index(horizon)
        horizon_tensor = torch.tensor(horizon_index, dtype=torch.long, device=device)
        horizon_embedding = self.horizon_embedding(horizon_tensor)
        horizon_embedding = horizon_embedding.expand(addresses.shape[0], -1)
        path_embedding = self.path_embeddings(addresses)
        episode_context, episode_attention = self._read_episode(
            torch.cat((base, path_embedding, horizon_embedding), dim=-1),
            state.episode_payloads,
            prediction=True,
        )
        controller_input = torch.cat((
            horizon_embedding,
            path_embedding,
            base,
            reads.parent,
            reads.children_mean,
            reads.siblings_mean,
            episode_context,
        ), dim=-1)
        hidden = torch.tanh(self.prediction_core(
            self.prediction_input_norm(controller_input)
        ))
        gate = torch.sigmoid(self.prediction_gate(hidden))
        erase = torch.sigmoid(self.prediction_erase(hidden))
        add = torch.tanh(self.prediction_add(hidden))
        proposed_payloads = base * (1.0 - gate * erase) + gate * add
        split_logits = self.prediction_split(hidden).squeeze(-1)
        hierarchy = hierarchical_reach_stop_weights(
            addresses, split_logits, self.config.max_depth,
        )
        payloads = base + hierarchy.reach[:, None] * (proposed_payloads - base)
        value_logits = self.prediction_value(payloads).squeeze(-1)
        rgb_bit_logits = (
            self.rgb_bit_head(payloads).reshape(-1, 3, 8)
            if self.rgb_bit_head is not None else None
        )
        probabilities, frame_logits = soft_hierarchical_rasterize(
            addresses,
            value_logits,
            hierarchy,
            image_size=self.config.image_size,
            canvas_size=self.config.canvas_size,
        )
        slot_probabilities = None
        slot_frame_logits = None
        if self.config.prediction_slots:
            count = self.config.prediction_slots
            slot_memory = (
                state.episode_payloads[:count]
                + self.episode_slot_embedding[:count]
            )
            expanded_payloads = payloads[:, None, :].expand(-1, count, -1)
            expanded_base = base[:, None, :].expand(-1, count, -1)
            expanded_slots = slot_memory[None, :, :].expand(addresses.shape[0], -1, -1)
            slot_hidden = torch.tanh(self.slot_prediction_core(torch.cat((
                expanded_payloads, expanded_base, expanded_slots,
            ), dim=-1)))
            slot_value_logits = self.slot_prediction_value(slot_hidden).squeeze(-1).transpose(0, 1)
            spatial_keys = F.normalize(
                self.slot_prediction_key(torch.cat((payloads, base), dim=-1)), dim=-1,
            )
            slot_queries = F.normalize(
                self.slot_prediction_query(slot_memory), dim=-1,
            )
            slot_compatibility = slot_queries @ spatial_keys.transpose(0, 1)
            slot_value_logits = slot_value_logits + (
                self.slot_prediction_logit_scale.exp().clamp(max=20.0)
                * slot_compatibility
            )
            slot_probability_rows = []
            slot_logit_rows = []
            for slot in range(count):
                slot_probability, slot_logit = soft_hierarchical_rasterize(
                    addresses,
                    slot_value_logits[slot],
                    hierarchy,
                    image_size=self.config.image_size,
                    canvas_size=self.config.canvas_size,
                )
                slot_probability_rows.append(slot_probability)
                slot_logit_rows.append(slot_logit)
            slot_probabilities = torch.stack(slot_probability_rows, dim=0)
            slot_frame_logits = torch.stack(slot_logit_rows, dim=0)
            probabilities = 1.0 - torch.prod(1.0 - slot_probabilities, dim=0)
            frame_logits = torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))
        predicted_state = V2MemoryState(
            spatial_addresses=addresses,
            spatial_payloads=payloads,
            episode_payloads=state.episode_payloads,
        )
        return V2PredictionOutput(
            state=predicted_state,
            candidate_addresses=addresses,
            predicted_payloads=payloads,
            value_logits=value_logits,
            split_logits=split_logits,
            hierarchy=hierarchy,
            frame_probabilities=probabilities,
            frame_logits=frame_logits,
            expected_nodes=expected_node_count(hierarchy),
            horizon=horizon,
            slot_frame_probabilities=slot_probabilities,
            slot_frame_logits=slot_frame_logits,
            episode_attention_weights=episode_attention,
            rgb_bit_logits=rgb_bit_logits,
        )
