"""Shared adapter helpers in ollajev.adapters."""

from ollajev.adapters import wire_questions


def test_instructions_fall_back_to_the_humanized_question_id():
    assert wire_questions({"is_refund": {"type": "noul"}}) == {
        "is_refund": {"type": "noul", "instructions": "is refund"}
    }


def test_none_criteria_are_dropped_but_empty_criteria_are_kept():
    qs = wire_questions(
        {
            "c": {"type": "choice", "instructions": "Team?", "criteria": {"a": None, "b": "B"}},
            "s": {"type": "score", "criteria": []},
            "n": {"type": "noul", "criteria": None},
        }
    )
    assert qs["c"] == {"type": "choice", "instructions": "Team?", "criteria": {"a": None, "b": "B"}}
    assert qs["s"]["criteria"] == []
    assert "criteria" not in qs["n"]
