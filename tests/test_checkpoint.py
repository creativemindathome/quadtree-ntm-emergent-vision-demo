import json
from pathlib import Path

import torch

from ntm.quadtree_world_model_v2 import QuadtreeWorldModelV2, V2ModelConfig


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_loads_exactly():
    config = json.loads((ROOT / "checkpoints/config.json").read_text())
    checkpoint = torch.load(
        ROOT / "checkpoints/checkpoint-010000.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = QuadtreeWorldModelV2(V2ModelConfig(**config["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    assert checkpoint["update"] == 10_000
    assert sum(parameter.numel() for parameter in model.parameters()) == 600_632

