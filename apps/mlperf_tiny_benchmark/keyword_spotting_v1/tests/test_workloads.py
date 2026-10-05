"""Integrity and Relay/tensor roundtrip tests for exported workloads."""

import importlib.util
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_workloads():
    package_name = "keyword_spotting_v1_test_app"
    if package_name not in sys.modules:
        import importlib.util
        package_path = APP_ROOT / "python" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package_name, package_path,
            submodule_search_locations=[str(package_path.parent)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    import importlib

    return importlib.import_module(f"{package_name}.vta_workload")


def test_relay_function_and_activation_roundtrip_without_source_model_or_input(tmp_path):
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


def test_loader_rejects_resealed_snapshot_from_another_model(tmp_path):
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
    document["model"]["id"] = "foreign_model"
    document["snapshot_sha256"] = workloads._seal(document)
    path = tmp_path / "foreign-workloads.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="different model"):
        workloads.load_workloads(path, validate_config_space=False)


def test_loader_rejects_oversized_files_before_json_parse(tmp_path):
    workloads = _load_workloads()
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (workloads.MAX_WORKLOAD_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds maximum"):
        workloads.load_workloads(path)
