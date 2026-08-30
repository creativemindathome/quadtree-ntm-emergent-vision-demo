import json
from pathlib import Path

import torch

from ntm.quadtree_world_model_v2 import QuadtreeWorldModelV2, V2ModelConfig


ROOT = Path(__file__).resolve().parents[1]


def test_pretrained_model_loads_exactly():
    config = json.loads((ROOT / "pretrained/config.json").read_text())
    trained = torch.load(
        ROOT / "pretrained/model-10000.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = QuadtreeWorldModelV2(V2ModelConfig(**config["model_config"]))
    model.load_state_dict(trained["model"], strict=True)
    assert trained["update"] == 10_000
    assert sum(parameter.numel() for parameter in model.parameters()) == 600_632
