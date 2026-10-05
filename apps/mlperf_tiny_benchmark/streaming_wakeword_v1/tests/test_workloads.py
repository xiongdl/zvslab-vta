"""Local workload snapshot identity, activation and tamper contracts."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _workload_module():
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    from python import vta_workload

    return vta_workload


def test_int8_activation_and_relay_function_roundtrip(tmp_path):
    import tvm
    from tvm import relay

    workloads = _workload_module()
    data = relay.var("audio", shape=(1, 30, 1, 40), dtype="int8")
    function = relay.Function([data], relay.add(data, relay.const(1, "int8")))
    activation = np.full((1, 30, 1, 40), -32, dtype=np.int8)
    function_sha = hashlib.sha256(tvm.ir.save_json(function).encode("utf-8")).hexdigest()
    layer = workloads.make_layer_record(
        index=0, symbol="wakeword_vta_0", function=function,
        compute_sha256=function_sha, inputs=((1, 30, 1, 40),), input_dtypes=("int8",),
        output_shape=(1, 30, 1, 40), output_dtype="int8", activation=activation,
        config_space_identity="a" * 64,
    )
    document = workloads.make_document(
        model_sha256="b" * 64, input_sha256="c" * 64,
        config_bytes=b'{"geometry":"test"}', config_basename="vta.json",
        geometry={"batch": 1}, tvm_version=tvm.__version__, vta_version="test",
        workloads=(layer,),
    )
    path = workloads.write_workloads(document, tmp_path / "workloads.json")
    loaded = workloads.load_workloads(path, validate_config_space=False)
    assert loaded.layers[0].symbol == "wakeword_vta_0"
    assert loaded.layers[0].activation.dtype == np.int8
    np.testing.assert_array_equal(loaded.layers[0].activation, activation)
    result = relay.create_executor(
        "debug", mod=tvm.IRModule.from_expr(loaded.layers[0].function),
        device=tvm.cpu(), target="llvm",
    ).evaluate()(activation).numpy()
    np.testing.assert_array_equal(result, activation + np.int8(1))


def test_snapshot_rejects_tampered_activation_before_returning_a_layer(tmp_path):
    import tvm
    from tvm import relay

    workloads = _workload_module()
    data = relay.var("audio", shape=(1,), dtype="int8")
    function = relay.Function([data], data)
    function_sha = hashlib.sha256(tvm.ir.save_json(function).encode("utf-8")).hexdigest()
    layer = workloads.make_layer_record(
        index=0, symbol="wakeword_vta_0", function=function,
        compute_sha256=function_sha, inputs=((1,),), input_dtypes=("int8",),
        output_shape=(1,), output_dtype="int8", activation=np.array([-1], dtype=np.int8),
        config_space_identity="d" * 64,
    )
    document = workloads.make_document(
        model_sha256="e" * 64, input_sha256="f" * 64,
        config_bytes=b"{}", config_basename="vta.json", geometry={},
        tvm_version=tvm.__version__, vta_version="test", workloads=(layer,),
    )
    path = workloads.write_workloads(document, tmp_path / "tampered.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["workloads"][0]["activation"]["data"] = "AA=="
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity hash mismatch"):
        workloads.load_workloads(path, validate_config_space=False)
