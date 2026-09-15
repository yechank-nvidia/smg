"""Engine-free regression tests for TokenSpeed's protobuf-to-engine seed mapping."""

import ast
import json
from pathlib import Path
from typing import Any

import pytest

tokenspeed_scheduler_pb2 = pytest.importorskip("smg_grpc_proto.generated.tokenspeed_scheduler_pb2")

_SERVICER_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed" / "servicer.py"


@pytest.fixture(scope="module")
def sampling_params_from_proto():
    # Execute the original static method, not a reimplementation. Importing
    # the whole servicer eagerly loads the GPU-only TokenSpeed engine stack.
    tree = ast.parse(_SERVICER_PATH.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TokenSpeedSchedulerServicer"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sampling_params_from_proto"
    )
    namespace = {"tokenspeed_scheduler_pb2": tokenspeed_scheduler_pb2, "Any": Any, "json": json}
    exec(compile(ast.Module(body=[method], type_ignores=[]), _SERVICER_PATH, "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("seed", [0, 1, 20260910, 2**32 - 1, 2**32, 2**63 - 1, 2**64 - 1])
def test_explicit_seed_roundtrips_into_engine_field(sampling_params_from_proto, seed):
    # This tests lossless uint64 wire conversion, not the engine's supported seed range.
    params = tokenspeed_scheduler_pb2.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=2048,
        stop=["stop-here"],
        ignore_eos=True,
        n=1,
    )
    unseeded = sampling_params_from_proto(params)
    params.sampling_seed = seed
    parsed = tokenspeed_scheduler_pb2.SamplingParams.FromString(params.SerializeToString())

    assert parsed.HasField("sampling_seed")
    converted = sampling_params_from_proto(parsed)
    assert converted == {**unseeded, "seed": seed}
    assert "sampling_seed" not in converted
    assert converted["temperature"] == 0.0


def test_absent_seed_keeps_engine_default(sampling_params_from_proto):
    params = tokenspeed_scheduler_pb2.SamplingParams()
    parsed = tokenspeed_scheduler_pb2.SamplingParams.FromString(params.SerializeToString())

    assert not parsed.HasField("sampling_seed")
    converted = sampling_params_from_proto(parsed)
    assert "seed" not in converted
    assert "sampling_seed" not in converted
