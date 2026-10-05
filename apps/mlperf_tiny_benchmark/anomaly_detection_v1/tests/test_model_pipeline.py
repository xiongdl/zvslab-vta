"""Model contract and one-window preprocessing tests."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "ad01_fp32.tflite"
SAMPLE_PATH = APP_ROOT / "samples" / "normal_id_01_00000000.wav"


def _model():
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    return importlib.import_module("python.model")


def test_float_model_contract_and_custom_matching_topology_are_supported(tmp_path):
    module = _model()
    imported = module.import_float_model(MODEL_PATH)
    assert imported.model_sha256 == module.MODEL_SHA256
    assert imported.input_shape == (1, 640)
    assert imported.output_shape == (1, 640)
    assert imported.input_dtype == imported.output_dtype == "float32"
    assert imported.tflite_operator_names == ("FULLY_CONNECTED",) * 10
    assert imported.dense_channel_pairs == (
        (640, 128), (128, 128), (128, 128), (128, 128), (128, 8),
        (8, 128), (128, 128), (128, 128), (128, 128), (128, 640),
    )
    changed_hash = tmp_path / "supported-copy.tflite"
    changed_hash.write_bytes(MODEL_PATH.read_bytes().replace(b"input_1", b"input_2", 1))
    copied = module.import_float_model(changed_hash)
    assert copied.model_sha256 != imported.model_sha256
    assert copied.input_name == "input_2"


def test_malformed_or_incompatible_models_fail_at_import_boundary(tmp_path):
    module = _model()
    path = tmp_path / "invalid.tflite"
    path.write_bytes(b"not a flatbuffer")
    with pytest.raises(ValueError, match="valid TFLite FlatBuffer"):
        module.import_float_model(path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("shape", "unexpected model input contract"),
        ("dtype", "unexpected model input contract"),
        ("topology", "unexpected TFLite operator topology"),
    ],
)
def test_flatbuffer_shape_dtype_and_operator_topology_are_checked(monkeypatch, mutation, message):
    module = _model()

    class Tensor:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

        def Name(self):
            return b"input_1"

        def ShapeAsNumpy(self):
            return self.shape

        def Type(self):
            return self.dtype

    class Graph:
        def InputsLength(self): return 1
        def OutputsLength(self): return 1
        def Inputs(self, _index): return 0
        def Outputs(self, _index): return 1
        def Tensors(self, index): return inputs[index]
        def OperatorsLength(self): return 1 if mutation == "topology" else 0
        def Operators(self, _index): return SimpleOperator()

    class SimpleOperator:
        def OpcodeIndex(self): return 0
        def Inputs(self, _index): return 0

    class Code:
        def BuiltinCode(self): return 0

    class Model:
        def Version(self): return 3
        def SubgraphsLength(self): return 1
        def Subgraphs(self, _index): return Graph()
        def OperatorCodes(self, _index): return Code()

    input_shape = (1, 639) if mutation == "shape" else (1, 640)
    input_type = int(module.tflite.TensorType.INT8) if mutation == "dtype" else int(module.tflite.TensorType.FLOAT32)
    inputs = [Tensor(input_shape, input_type), Tensor((1, 640), int(module.tflite.TensorType.FLOAT32))]
    monkeypatch.setattr(module.tflite.Model, "GetRootAsModel", lambda _bytes, _offset: Model())
    with pytest.raises(ValueError, match=message):
        module._flatbuffer_contract(b"fake model")


def test_log_mel_preprocessing_selects_exactly_first_finite_vector():
    module = _model()
    first, count = module.load_first_window(SAMPLE_PATH)
    repeated, repeated_count = module.load_first_window(SAMPLE_PATH)
    assert first.shape == (1, 640)
    assert first.dtype == np.float32
    assert count == repeated_count > 1
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, module.load_sample(SAMPLE_PATH)[:1])


def test_invalid_wav_is_rejected_before_model_compilation(tmp_path):
    module = _model()
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not a wave file")
    with pytest.raises(ValueError, match="valid mono PCM16 16 kHz WAV"):
        module.load_first_window(invalid)


def test_cpu_model_preparation_does_not_load_vta_or_require_vta_environment():
    code = (
        "import os,sys; os.environ.pop('VTA_BACKEND',None); "
        "os.environ.pop('VTA_CONFIG_FILE',None); from python.model import prepare_model; "
        f"p=prepare_model({str(MODEL_PATH)!r}); "
        "assert not any(n == 'vta' or n.startswith('vta.') for n in sys.modules); "
        "assert p.routing.symbols == (); assert p.routing.host_dense_count == 0; "
        "assert p.routing.host_convolution_count == 10"
    )
    import os
    import subprocess

    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env.pop("VTA_CONFIG_FILE", None)
    env["PYTHONPATH"] = ":".join(
        [str(APP_ROOT), str(APP_ROOT.parents[3] / "tvm" / "python")]
    )
    completed = subprocess.run(
        [str(APP_ROOT.parents[3] / ".envs/tvm-vta-env/bin/python"), "-c", code],
        cwd=APP_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
