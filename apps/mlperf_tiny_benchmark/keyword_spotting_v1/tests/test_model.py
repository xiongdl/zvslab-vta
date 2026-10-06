"""KWS input, fixed-point preparation, and model provenance contracts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "kws_ref_model_float32.tflite"
SAMPLES = sorted(APP_ROOT.joinpath("samples").glob("*.wav"))


@pytest.fixture(scope="module")
def model():
    package_name = "keyword_spotting_v1_test_app"
    if package_name not in sys.modules:
        package_path = APP_ROOT / "python" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package_name, package_path,
            submodule_search_locations=[str(package_path.parent)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.model")


def _executor(module, params):
    import tvm
    from tvm import relay

    with tvm.transform.PassContext(opt_level=3):
        library = relay.build(module, target="llvm", params=params)
    return tvm.contrib.graph_executor.GraphModule(library["default"](tvm.cpu(0)))


def _run(graph, input_name, sample):
    graph.set_input(input_name, sample)
    graph.run()
    return graph.get_output(0).numpy().copy()


def test_preprocessing_is_deterministic_and_preserves_float_features(model):
    sample = model.load_sample(SAMPLES[0])
    assert sample.shape == model.INPUT_SHAPE
    assert sample.dtype == np.float32
    np.testing.assert_array_equal(sample, model.load_sample(SAMPLES[0]))


def test_import_records_provenance_and_supported_model_contract(model):
    imported = model.import_model(MODEL_PATH)
    assert imported.model_sha256 == model.MODEL_SHA256
    assert imported.input_shape == (1, 49, 10, 1)
    assert imported.input_dtype == "float32"
    assert imported.output_shape == (1, 12)
    assert imported.output_dtype == "float32"

    assert imported.input_name == "serving_default_input_1:0"
    assert imported.output_name == "StatefulPartitionedCall:0"


def test_prepared_cpu_quantizes_float_model_for_all_committed_samples(model):
    imported = model.import_model(MODEL_PATH)
    prepared = model.prepare_model(MODEL_PATH, use_vta=False)
    prepared_graph = _executor(prepared.reference_module, imported.params)
    for sample_path in SAMPLES:
        sample = model.load_sample(sample_path)
        prepared_output = _run(prepared_graph, imported.input_name, sample)
        assert prepared_output.dtype == np.float32
        assert prepared_output.shape == model.OUTPUT_SHAPE
        assert np.isfinite(prepared_output).all(), sample_path.name


def test_real_partitioning_contains_actual_vta_convolution_work(model, monkeypatch):
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(APP_ROOT.parents[3] / "vta/config/vta_64mac.json"))
    prepared = model.prepare_model(MODEL_PATH, use_vta=True)
    assert len(prepared.routing.symbols) == 4
    assert prepared.routing.convolutions_per_partition == (1, 1, 1, 1)
    assert prepared.routing.composite_names == ("vta.qnn_conv2d",) * 4


def test_invalid_audio_is_rejected_before_model_execution(model, tmp_path):
    bad_audio = tmp_path / "stereo.wav"
    bad_audio.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="unable to read WAV"):
        model.load_sample(bad_audio)
