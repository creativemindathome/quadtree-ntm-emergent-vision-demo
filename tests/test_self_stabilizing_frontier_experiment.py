from argparse import Namespace
from pathlib import Path

from experiment_self_stabilizing_frontier import command_for


def test_pure_experiment_has_no_split_bound_or_manual_stabilizer():
    args = Namespace(updates=10, device="cpu", checkpoint_every=250)
    command = command_for(args, "pure", 108, Path("run"))
    joined = " ".join(command)

    assert "--candidate-max-nodes" not in command
    assert "--minimum-prediction-depth 0" in joined
    assert "--structure-temperature-bpp 0.0" in joined
    assert "--predictive-logit-soft-clip 0.0" in joined
    assert "--proposal-distillation-weight 1.0" in joined
    assert "--candidate-exploration-paths 1" in joined
    assert "--skip-final-eval" in command


def test_no_distillation_is_an_exact_control_of_the_proposal_bridge():
    args = Namespace(updates=10, device="cpu", checkpoint_every=250)
    command = command_for(args, "no_distillation", 108, Path("run"))
    joined = " ".join(command)

    assert "--proposal-distillation-weight 0.0" in joined
    assert "--structure-temperature-bpp 0.0" in joined
    assert "--predictive-logit-soft-clip 0.0" in joined
