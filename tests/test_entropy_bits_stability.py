import torch

from train_object_permanence_quadtree_v2 import recursive_rgb_bit_loss


def _one_level_tree():
    return torch.arange(5, dtype=torch.long)


def test_soft_clipped_entropy_code_has_finite_extreme_logit_gradients():
    addresses = _one_level_tree()
    split_logits = torch.tensor([100.0, -100.0, 100.0, -100.0, 100.0], requires_grad=True)
    rgb_logits = torch.empty(5, 3, 8).fill_(100.0)
    rgb_logits[::2].fill_(-100.0)
    rgb_logits.requires_grad_()
    target = torch.zeros(3, 2, 2)

    loss, metrics, _ = recursive_rgb_bit_loss(
        addresses,
        split_logits,
        rgb_logits,
        target,
        canvas_size=2,
        max_depth=1,
        structure_temperature_bpp=0.02,
        predictive_logit_soft_clip=12.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(split_logits.grad).all()
    assert torch.isfinite(rgb_logits.grad).all()
    assert metrics["predictive_logit_abs_max"] == 100.0


def test_symmetric_codes_keep_stop_and_split_branches_trainable():
    addresses = _one_level_tree()
    split_logits = torch.zeros(5, requires_grad=True)
    rgb_logits = torch.zeros(5, 3, 8, requires_grad=True)
    target = torch.zeros(3, 2, 2)

    loss, metrics, _ = recursive_rgb_bit_loss(
        addresses,
        split_logits,
        rgb_logits,
        target,
        canvas_size=2,
        max_depth=1,
        structure_temperature_bpp=0.02,
        predictive_logit_soft_clip=12.0,
    )
    loss.backward()

    assert rgb_logits.grad[0].abs().sum() > 0
    assert rgb_logits.grad[1:].abs().sum() > 0
    assert 0.0 < metrics["posterior_entropy_bits"] <= 1.0
    assert 0.0 < metrics["reachable_split_probability"] < 1.0
    assert metrics["posterior_saturation_fraction"] == 0.0


def test_posterior_diagnostics_detect_saturated_structure():
    addresses = _one_level_tree()
    split_logits = torch.full((5,), -100.0)
    rgb_logits = torch.zeros(5, 3, 8)
    target = torch.zeros(3, 2, 2)

    _, metrics, _ = recursive_rgb_bit_loss(
        addresses,
        split_logits,
        rgb_logits,
        target,
        canvas_size=2,
        max_depth=1,
        structure_temperature_bpp=0.02,
        predictive_logit_soft_clip=12.0,
    )

    assert metrics["posterior_saturation_fraction"] == 1.0
    assert metrics["posterior_entropy_bits"] < 1e-4
    assert metrics["evidence_gap_bpp_q10"] <= metrics["evidence_gap_bpp_q50"]
    assert metrics["evidence_gap_bpp_q50"] <= metrics["evidence_gap_bpp_q90"]
