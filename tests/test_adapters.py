"""Shared adapter helpers in ollajev.adapters."""

import pytest

from ollajev import names
from ollajev.adapters import softmax, wire_questions


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


def test_softmax_defaults_to_the_plain_one_and_stays_finite():
    assert softmax([2.0, 0.0]) == pytest.approx([0.8808, 0.1192], abs=1e-4)
    assert softmax([2.0, 0.0], 2.0) == pytest.approx([0.7311, 0.2689], abs=1e-4)
    assert sum(softmax([1000.0, 1001.0, 999.0])) == pytest.approx(1.0)


def test_runtime_of_reads_bare_tags_without_fake_filenames():
    assert names.runtime_of("Q4_K_M") == "llama.cpp"
    assert names.runtime_of("model.gguf") == "llama.cpp"
    assert names.runtime_of("int8") == "ONNX"
    assert names.runtime_of("model.onnx") == "ONNX"
    assert names.runtime_of(None) == "PyTorch"
    assert names.runtime_of("weird") == "PyTorch"


def test_kev_head_metadata_is_read_without_torch(tmp_path):
    import collections

    import torch

    from ollajev.adapters import kev

    meta = {
        "base": "Qwen/Qwen3.5-4B-Base",
        "base_revision": "a" * 40,
        "head": collections.OrderedDict(weight=torch.zeros(4, 4, dtype=torch.bfloat16)),
        "temperature": 1.5,
        "holdout": ["x"],
    }
    head = tmp_path / "head.pt"
    torch.save(meta, head)
    read = kev._head_metadata(str(head))
    assert read["base"] == meta["base"] and read["base_revision"] == meta["base_revision"]
    assert read["temperature"] == 1.5 and read["head"]["weight"] is None  # tensors are skipped, not loaded


def test_kev_head_reader_builds_no_foreign_classes():
    import io
    import pickle

    from ollajev.adapters import kev

    payload = pickle.dumps({"base": "x", "evil": io.StringIO()})
    with pytest.raises(pickle.UnpicklingError):
        kev._MetadataOnly(io.BytesIO(payload)).load()


def test_kev_head_metadata_falls_back_to_torch_for_an_old_style_file(tmp_path):
    import torch

    from ollajev.adapters import kev

    head = tmp_path / "head.pt"
    torch.save({"base": "Qwen/Qwen3-0.6B-Base", "head": torch.zeros(2)}, head, _use_new_zipfile_serialization=False)
    assert kev._head_metadata(str(head))["base"] == "Qwen/Qwen3-0.6B-Base"


def test_fast_cpu_convs_match_the_reference_causal_conv(monkeypatch):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    from ollajev import adapters

    reference = qwen.causal_conv1d_fn
    monkeypatch.setattr(qwen, "causal_conv1d_fn", reference)  # restored after the test
    adapters.fast_cpu_convs()
    fast = qwen.causal_conv1d_fn
    assert fast is not reference
    adapters.fast_cpu_convs()
    assert qwen.causal_conv1d_fn is fast  # applied once
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        x = torch.randn(2, 64, 37, dtype=dtype)
        weight, bias = torch.randn(64, 4, dtype=dtype), torch.randn(64, dtype=dtype)
        for b, act in ((None, None), (bias, "silu")):
            want, got = reference(x, weight, bias=b, activation=act), fast(x, weight, bias=b, activation=act)
            assert got.dtype == want.dtype and got.shape == want.shape
            assert torch.allclose(got.float(), want.float(), atol=tol, rtol=tol)
