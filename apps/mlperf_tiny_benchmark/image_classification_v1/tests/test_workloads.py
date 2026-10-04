"""Integrity and Relay/tensor roundtrip tests for exported workloads."""

import importlib.util
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_workloads():
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("ic_v1_workloads", APP_ROOT / "workloads.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_relay_function_and_activation_roundtrip_without_model_or_image(tmp_path):
    import tvm
    from tvm import relay

    workloads = _load_workloads()
    data = relay.var("data", shape=(1, 2), dtype="float32")
    function = relay.Function([data], relay.add(data, relay.const(1.0)))
    activation = np.array([[2.0, 3.0]], dtype="float32")
    record = workloads.make_layer_record(
        index=0, symbol="layer0", function=function,
        compute_sha256=hashlib.sha256(tvm.ir.save_json(function).encode()).hexdigest(),
        inputs=((1, 2),), input_dtypes=("float32",), output_shape=(1, 2),
        output_dtype="float32", activation=activation, config_space_identity="d" * 64,
    )
    document = workloads.make_document(
        model_sha256="a" * 64, input_sha256="b" * 64,
        config_bytes=b'{"geometry":1}', config_basename="vta.json",
        geometry={"batch": 1}, tvm_version=tvm.__version__, vta_version="test",
        workloads=(record,),
    )
    path = tmp_path / "workloads.json"
    workloads.write_workloads(document, path)
    loaded = workloads.load_workloads(path, validate_config_space=False)
    assert loaded.layers[0].symbol == "layer0"
    np.testing.assert_array_equal(loaded.layers[0].activation, activation)
    restored = loaded.layers[0].function
    module = tvm.IRModule.from_expr(restored)
    result = relay.create_executor("debug", mod=module, device=tvm.cpu(), target="llvm").evaluate()(activation)
    np.testing.assert_array_equal(result.numpy(), [[3.0, 4.0]])


@pytest.mark.parametrize("mutation", ["activation", "function", "config", "document"])
def test_loader_rejects_integrity_corruption(tmp_path, mutation):
    import json
    import tvm
    from tvm import relay

    workloads = _load_workloads()
    data = relay.var("data", shape=(1,), dtype="float32")
    function = relay.Function([data], data)
    record = workloads.make_layer_record(
        index=0, symbol="layer0", function=function,
        compute_sha256=hashlib.sha256(tvm.ir.save_json(function).encode()).hexdigest(),
        inputs=((1,),), input_dtypes=("float32",), output_shape=(1,),
        output_dtype="float32", activation=np.array([1], dtype="float32"),
        config_space_identity="d" * 64,
    )
    document = workloads.make_document(
        model_sha256="a" * 64, input_sha256="b" * 64,
        config_bytes=b"{}", config_basename="vta.json", geometry={},
        tvm_version=tvm.__version__, vta_version="test", workloads=(record,),
    )
    path = tmp_path / "bad.json"
    workloads.write_workloads(document, path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "activation":
        value["workloads"][0]["activation"]["data"] = "AA=="
    elif mutation == "function":
        value["workloads"][0]["function"]["data"] += " "
    elif mutation == "config":
        value["config"]["raw_base64"] = "Y2hhbmdlZA=="
    else:
        value["model"]["sha256"] = "c" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|integrity|size|snapshot"):
        workloads.load_workloads(path, validate_config_space=False)


def test_loader_rejects_oversized_files_before_json_parse(tmp_path):
    workloads = _load_workloads()
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (workloads.MAX_WORKLOAD_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds maximum"):
        workloads.load_workloads(path)


def test_real_fsim_export_recovers_all_layers_without_model_or_image(tmp_path):
    import tvm
    from tvm import relay

    sys.path.insert(0, str(APP_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("ic_v1_workload_runtime", APP_ROOT / "runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runtime
        spec.loader.exec_module(runtime)
        from deployment_compute import capture_deployment_compute

        prepared = runtime.prepare_model(runtime.MODEL_PATH)
        original = capture_deployment_compute(
            prepared.mixed_module, runtime.MODEL_ID, prepared.imported.model_sha256
        )
        path = tmp_path / "workloads.json"
        runtime.run_selected(
            target="vta,llvm", simulator="fsim", output_dir=tmp_path / "bundle",
            export_workloads=path,
        )
        snapshot = _load_workloads().load_workloads(path)
        assert [item.index for item in snapshot.layers] == list(range(8))
        assert [item.symbol for item in snapshot.layers] == [item.symbol for item in original.layers]
        for actual, restored in zip(original.layers, snapshot.layers):
            original_fn = relay.Function(actual.function.params, actual.function.body)
            restored_fn = relay.Function(restored.function.params, restored.function.body)
            original_module = relay.transform.InferType()(tvm.IRModule.from_expr(original_fn))
            restored_module = relay.transform.InferType()(tvm.IRModule.from_expr(restored_fn))
            original_output = relay.create_executor(
                "debug", mod=original_module, device=tvm.cpu(), target="llvm"
            ).evaluate()(restored.activation).numpy()
            restored_output = relay.create_executor(
                "debug", mod=restored_module, device=tvm.cpu(), target="llvm"
            ).evaluate()(restored.activation).numpy()
            np.testing.assert_array_equal(restored_output, original_output)
    finally:
        sys.path.pop(0)

    env = os.environ.copy()
    env["VTA_BACKEND"] = "tsim"
    env["VTA_CONFIG_FILE"] = str(APP_ROOT.parents[2] / "config" / "vta_64mac.json")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(APP_ROOT.parents[3] / "tvm" / "python"), str(APP_ROOT.parents[2] / "python"), str(APP_ROOT)]
    )
    code = (
        "from workloads import load_workloads; "
        f"snapshot = load_workloads({str(path)!r}); "
        "assert len(snapshot.layers) == 8"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
