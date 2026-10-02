from __future__ import annotations

from tools import replay_v4_8_promotion_pair_guarded_posthoc as replay
from tools.action_diagnostics_v4_9 import DECODER


def test_guarded_posthoc_parser_forbids_all_but_validation_safe_decoder() -> None:
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


def test_guarded_posthoc_metadata_is_explicitly_non_gating() -> None:
    source = __import__("inspect").getsource(replay.main)
    assert '"post_hoc_diagnostic_only": True' in source
    assert '"counted_for_promotion_gate": False' in source
    assert '"counted_for_formal_gate": False' in source
    assert '"formal_test_accessed": False' in source
    assert '"promotion_result_reinterpretation_forbidden": True' in source
