"""Float feature input, model import, quantization, and routing contracts."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[3]
MODEL = APP_ROOT / "model/str_ww_ref_model_floag32.tflite"
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
        f"p=prepare_model({str(MODEL)!r}); "
        "assert p.routing.symbols == (); assert 'vta' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_float32_model_contract_and_deterministic_unquantized_features():
    sys.path.insert(0, str(APP_ROOT))
    try:
        from python.model import (
            INPUT_DTYPE, INPUT_NAME, INPUT_SHAPE, OUTPUT_DTYPE, OUTPUT_SHAPE,
            import_model, load_sample,
        )

        imported = import_model(MODEL)
        assert imported.model_sha256 == "c735ab47248df7648d9cb4397c0e7d161fe2e88ede17ad900f34a4163d89b267"
        assert imported.input_name == INPUT_NAME
        assert imported.input_shape == INPUT_SHAPE == (1, 30, 1, 40)
        assert imported.input_dtype == INPUT_DTYPE == "float32"
        assert imported.output_shape == OUTPUT_SHAPE == (1, 3)
        assert imported.module["main"].ret_type.dtype == OUTPUT_DTYPE == "float32"

        first = load_sample(APP_ROOT / "samples" / SAMPLES[0])
        assert first.shape == INPUT_SHAPE
        assert first.dtype == np.float32
        assert np.isfinite(first).all()
        assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
        np.testing.assert_array_equal(first, load_sample(APP_ROOT / "samples" / SAMPLES[0]))
    finally:
        sys.path.remove(str(APP_ROOT))


def test_relay_quantization_uses_reference_policy_and_keeps_float_io():
    sys.path.insert(0, str(APP_ROOT))
    try:
        from python.model import prepare_model

        prepared = prepare_model(MODEL)
        input_type = prepared.reference_module["main"].params[0].checked_type
        output_type = prepared.reference_module["main"].ret_type
        assert input_type.dtype == "float32"
        assert tuple(int(dim) for dim in input_type.shape) == (1, 30, 1, 40)
        assert output_type.dtype == "float32"
        assert tuple(int(dim) for dim in output_type.shape) == (1, 3)
    finally:
        sys.path.remove(str(APP_ROOT))


def test_real_partitioning_contains_source_convolutions():
    sys.path.insert(0, str(APP_ROOT))
    try:
        from python.model import prepare_model

        prepared = prepare_model(MODEL, use_vta=True)
        assert len(prepared.routing.symbols) == 4
        assert all(symbol.startswith("tvmgen_mlperf_streaming_wakeword_vta_main_")
                   for symbol in prepared.routing.symbols)
    finally:
        sys.path.remove(str(APP_ROOT))


def test_model_hash_tracks_custom_float_tflite_path(tmp_path):
    sys.path.insert(0, str(APP_ROOT))
    try:
        from python.model import import_model

        custom = tmp_path / "custom-float.tflite"
        custom.write_bytes(MODEL.read_bytes() + b"\0")
        imported = import_model(custom)
        assert imported.model_sha256 != "c735ab47248df7648d9cb4397c0e7d161fe2e88ede17ad900f34a4163d89b267"
        assert imported.module["main"].ret_type.dtype == "float32"
        assert tuple(int(dim) for dim in imported.module["main"].ret_type.shape) == (1, 3)
    finally:
        sys.path.remove(str(APP_ROOT))
