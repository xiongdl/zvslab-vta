"""KWS input, fixed-point preparation, and model provenance contracts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "kws_ref_model.tflite"
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


def test_preprocessing_is_deterministic_and_matches_int8_input_contract(model):
    sample = model.load_sample(SAMPLES[0])
    assert sample.shape == model.INPUT_SHAPE
    assert sample.dtype == np.int8
    np.testing.assert_array_equal(sample, model.load_sample(SAMPLES[0]))


def test_import_records_provenance_and_supported_model_contract(model, tmp_path):
    imported = model.import_model(MODEL_PATH)
    assert imported.model_sha256 == model.MODEL_SHA256
    assert imported.input_shape == (1, 49, 10, 1)
    assert imported.input_dtype == "int8"
    assert imported.output_shape == (1, 12)
    assert imported.output_dtype == "int8"

    custom = tmp_path / "same-contract.tflite"
    data = MODEL_PATH.read_bytes()
    custom.write_bytes(data.replace(b"input_1", b"input_2", 1))
    custom_import = model.import_model(custom)
    assert custom_import.model_sha256 != imported.model_sha256
    assert custom_import.input_name == "input_2"


def test_prepared_cpu_preserves_imported_qnn_outputs_for_all_committed_samples(model):
    imported = model.import_model(MODEL_PATH)
    prepared = model.prepare_model(MODEL_PATH, use_vta=False)
    original_graph = _executor(imported.module, imported.params)
    prepared_graph = _executor(prepared.reference_module, imported.params)
    for sample_path in SAMPLES:
        sample = model.load_sample(sample_path)
        original = _run(original_graph, imported.input_name, sample)
        prepared_output = _run(prepared_graph, imported.input_name, sample)
        assert original.dtype == prepared_output.dtype == np.int8
        np.testing.assert_array_equal(prepared_output, original, err_msg=sample_path.name)


def test_real_partitioning_reports_zero_coverage_without_invented_computation(model):
    prepared = model.prepare_model(MODEL_PATH, use_vta=True)
    assert prepared.routing.symbols == ()
    assert prepared.routing.convolutions_per_partition == ()
    assert "nn.conv2d" in prepared.routing.host_operator_names


def test_invalid_audio_is_rejected_before_model_execution(model, tmp_path):
    bad_audio = tmp_path / "stereo.wav"
    bad_audio.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="unable to read WAV"):
        model.load_sample(bad_audio)
