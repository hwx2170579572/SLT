from __future__ import annotations

from algos.sb3_torch.hybrid_policy_v4_9_qguard import DECODER
from tools import replay_v4_8_promotion_pair_qguard_posthoc as replay


def test_qguard_posthoc_parser_is_validation_safe() -> None:
    parser = replay.parser()
    decoder_choices = next(
        action.choices for action in parser._actions if action.dest == "decoder"
    )
    partition_choices = next(
        action.choices
        for action in parser._actions
        if action.dest == "traffic_partition"
    )
    assert tuple(decoder_choices) == (DECODER,)
    assert tuple(partition_choices) == ("train", "validation")
    assert "test" not in partition_choices


def test_qguard_posthoc_metadata_is_non_gating_and_fixed() -> None:
    source = __import__("inspect").getsource(replay.main)
    assert '"maximum_actor_target_q_regret": 0.05' in source
    assert '"counted_for_promotion_gate": False' in source
    assert '"counted_for_formal_gate": False' in source
    assert '"formal_test_accessed": False' in source
    assert '"promotion_result_reinterpretation_forbidden": True' in source
