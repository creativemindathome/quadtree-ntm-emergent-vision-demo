"""Animate a trained Candidate A episode beside its learned observation tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from ntm.quadtree_memory import address_to_bounds
from ntm.quadtree_world_model_v2 import (
    QuadtreeWorldModelV2, V2ModelConfig, _depths, select_hard_tree,
    soft_hierarchical_rasterize,
)
from tasks.object_permanence_arena import generate_object_permanence_family
from tasks.rgb_quadtree import RGBQuadtreeConfig, build_rgb_quadtree_sample
from tasks.visual_domains import VisualDomainConfig
from train_object_permanence_quadtree_v2 import curriculum_config


def _rgb(frame: torch.Tensor, scale: int) -> Image.Image:
    pixels = frame.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
    image = Image.fromarray(pixels, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _heat(probability: torch.Tensor, scale: int) -> Image.Image:
    value = probability.squeeze().detach().cpu().clamp(0, 1)
    rgb = torch.stack((value, value.square(), 1.0 - value), dim=-1).mul(255).byte().numpy()
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _highlight_target(
        image: Image.Image, mask: torch.Tensor, center: torch.Tensor, scale: int,
) -> None:
    """Mark the evaluator-only tracked object without altering model input."""
    draw = ImageDraw.Draw(image)
    ys, xs = torch.where(mask.detach().cpu())
    if xs.numel():
        pad = 4 * scale
        box = (
            max(0, int(xs.min()) * scale - pad),
            max(0, int(ys.min()) * scale - pad),
            min(image.width - 1, (int(xs.max()) + 1) * scale + pad),
            min(image.height - 1, (int(ys.max()) + 1) * scale + pad),
        )
        draw.rectangle(box, outline=(255, 35, 65), width=max(2, 2 * scale))
        draw.text((box[0], max(0, box[1] - 12)), "TARGET", fill=(255, 35, 65))
    cx, cy = (float(center[0]) * scale, float(center[1]) * scale)
    radius = max(4, 4 * scale)
    draw.line((cx - radius, cy, cx + radius, cy), fill=(255, 255, 255), width=max(1, scale))
    draw.line((cx, cy - radius, cx, cy + radius), fill=(255, 255, 255), width=max(1, scale))


def render(
        checkpoint: Path, output: Path, seed: int, scale: int,
        highlight_target: bool = False, show_depth: bool = False,
) -> dict:
    run = checkpoint.parent
    config = json.loads((run / "config.json").read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = QuadtreeWorldModelV2(V2ModelConfig(**config["model_config"]))
    model.load_state_dict(saved["model"])
    model.eval()
    tree_config = RGBQuadtreeConfig(**config["tree_config"])

    _, arena_config = curriculum_config(3000, 3000, seed, "causal_pinch")
    family = generate_object_permanence_family(
        arena_config, VisualDomainConfig(seed=seed + 17), family_index=3000,
        appearance_indices=(0,),
    )
    episode = family.episodes[0]
    physical = episode.physical_episode
    target_index = physical.tracked_indices[0]
    event = episode.physical_episode.contact_events[0]
    state = None
    frames = []
    leaf_counts = []
    max_depths = []
    for frame_index, frame in enumerate(episode.frames):
        with torch.no_grad():
            sample = build_rgb_quadtree_sample(frame, tree_config)
            observed = model.observe(sample, state)
            state = observed.state
            hard = select_hard_tree(
                sample["heap_indices"], observed.split_logits,
                model.config.max_depth, threshold=0.5,
                max_nodes=tree_config.max_nodes,
            )
        raw = _rgb(frame, scale)
        tree = raw.copy()
        if highlight_target:
            target_mask = physical.amodal_masks[frame_index, target_index]
            target_center = physical.centers[frame_index, target_index]
            _highlight_target(raw, target_mask, target_center, scale)
            _highlight_target(tree, target_mask, target_center, scale)
        draw = ImageDraw.Draw(tree)
        leaves = hard.addresses[hard.leaf_mask].detach().cpu().tolist()
        depths = hard.depths[hard.leaf_mask].detach().cpu().tolist()
        for address, depth in zip(leaves, depths):
            _, x, y, size = address_to_bounds(address, model.config.canvas_size)
            if x >= model.config.image_size or y >= model.config.image_size:
                continue
            x1 = min(x + size, model.config.image_size) * scale - 1
            y1 = min(y + size, model.config.image_size) * scale - 1
            color = (255, 230, 50) if depth >= 4 else (70, 220, 255)
            draw.rectangle((x * scale, y * scale, x1, y1), outline=color, width=max(1, scale))
        reconstruction = _heat(observed.frame_probabilities, scale)
        panels = [raw, tree]
        if show_depth:
            normalized_depth = (
                _depths(sample["heap_indices"]).to(observed.value_logits.device).float()
                / model.config.max_depth
            ).clamp(1e-4, 1.0 - 1e-4)
            depth_probability, _ = soft_hierarchical_rasterize(
                sample["heap_indices"].to(observed.value_logits.device),
                torch.logit(normalized_depth), observed.hierarchy,
                image_size=model.config.image_size,
                canvas_size=model.config.canvas_size,
            )
            depth_panel = _heat(depth_probability, scale)
            if highlight_target:
                _highlight_target(depth_panel, target_mask, target_center, scale)
            panels.append(depth_panel)
        panels.append(reconstruction)
        header = 48
        canvas = Image.new("RGB", (raw.width * len(panels), raw.height + header), "white")
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (panel_index * raw.width, header))
        label = ImageDraw.Draw(canvas)
        label.text((8, 7), f"RAW ENV  t={frame_index:02d}", fill="black")
        label.text((raw.width + 8, 7), f"HARD LEAVES  n={len(leaves)}  max-d={max(depths)}", fill="black")
        if show_depth:
            label.text((2 * raw.width + 8, 7), "SOFT EFFECTIVE DEPTH", fill="black")
            label.text((3 * raw.width + 8, 7), "SOFT RECONSTRUCTION", fill="black")
        else:
            label.text((2 * raw.width + 8, 7), "SOFT RECONSTRUCTION", fill="black")
        label.text((8, 26), f"outcome={event.outcome}  friction={event.friction:.2f}", fill="black")
        frames.append(canvas)
        leaf_counts.append(len(leaves))
        max_depths.append(max(depths))

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=120, loop=0, disposal=2)
    return {
        "gif": str(output.resolve()), "frames": len(frames),
        "leaf_count_range": [min(leaf_counts), max(leaf_counts)],
        "hard_max_depth_range": [min(max_depths), max(max_depths)],
        "candidate_active_depth": tree_config.active_depth,
        "candidate_nodes": tree_config.max_nodes,
        "target_highlighted": highlight_target,
        "effective_depth_panel": show_depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/causal-pinch-v3-3k-final/checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("inspection/causal_pinch/long_quadtree.gif"))
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--highlight-target", action="store_true")
    parser.add_argument("--show-depth", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(
        args.checkpoint, args.output, args.seed, args.scale,
        highlight_target=args.highlight_target, show_depth=args.show_depth,
    ), indent=2))


if __name__ == "__main__":
    main()
