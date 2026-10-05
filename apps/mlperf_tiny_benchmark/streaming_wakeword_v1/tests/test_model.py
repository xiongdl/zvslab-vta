"""Streaming feature input and exact fixed-point graph contracts."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[3]
SAMPLES = (
    "marvin-00176480_nohash_0.wav",
    "silence-doing_the_dishes-00000000.wav",
    "unknown-0165e0e8_nohash_0.wav",
)


def test_cpu_preparation_does_not_import_or_initialize_vta():
    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env.pop("VTA_CONFIG_FILE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "tvm/python"), str(REPO_ROOT / "vta/python"), str(APP_ROOT))
    )
    code = (
        "import sys; from python.model import prepare_model; "
        f"p=prepare_model({str(APP_ROOT / 'model/str_ww_ref_model.tflite')!r}); "
        "assert p.routing.symbols == (); assert 'vta' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_imported_qnn_and_prepared_cpu_graphs_match_exactly_on_all_samples():
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor

    sys.path.insert(0, str(APP_ROOT))
    try:
        from python.model import INPUT_NAME, import_model, load_sample, prepare_model

        model_path = APP_ROOT / "model/str_ww_ref_model.tflite"
        imported = import_model(model_path)
        prepared = prepare_model(model_path)
        original_lib = relay.build(imported.module, target="llvm", params=imported.params)
        prepared_lib = relay.build(prepared.reference_module, target="llvm", params=imported.params)
        original = graph_executor.GraphModule(original_lib["default"](tvm.cpu()))
        canonical = graph_executor.GraphModule(prepared_lib["default"](tvm.cpu()))
        for filename in SAMPLES:
            activation = load_sample(APP_ROOT / "samples" / filename)
            original.set_input(INPUT_NAME, tvm.nd.array(activation))
            canonical.set_input(INPUT_NAME, tvm.nd.array(activation))
            original.run()
            canonical.run()
            expected = original.get_output(0).numpy()
            actual = canonical.get_output(0).numpy()
            assert expected.dtype == np.int8
            assert expected.shape == (1, 3)
            np.testing.assert_array_equal(actual, expected)
    finally:
        sys.path.remove(str(APP_ROOT))


def test_loaded_activation_is_one_fixed_int8_audio_window():
    sys.path.insert(0, str(APP_ROOT))
    try:
        from python.model import load_sample

        value = load_sample(APP_ROOT / "samples" / SAMPLES[0])
        assert value.shape == (1, 30, 1, 40)
        assert value.dtype == np.int8
    finally:
        sys.path.remove(str(APP_ROOT))


def test_custom_path_with_the_same_supported_model_contract_is_not_hash_rejected(tmp_path):
    sys.path.insert(0, str(APP_ROOT))
    try:
        from python.model import import_model

        original = APP_ROOT / "model/str_ww_ref_model.tflite"
        custom = tmp_path / "custom-model.tflite"
        custom.write_bytes(original.read_bytes() + b"\0")
        imported = import_model(custom)
        assert imported.model_sha256 != "3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b"
        assert imported.module["main"].ret_type.dtype == "int8"
        assert tuple(int(dim) for dim in imported.module["main"].ret_type.shape) == (1, 3)
    finally:
        sys.path.remove(str(APP_ROOT))
