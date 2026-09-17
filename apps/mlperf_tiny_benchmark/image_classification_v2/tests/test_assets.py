# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Provenance and byte-level contracts for the MLPerf ResNet Large assets."""

import hashlib
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
VTA_ROOT = Path(__file__).resolve().parents[4]
REPOSITORY_ROOT = VTA_ROOT.parent
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"
MODEL_README_PATH = APP_ROOT / "model" / "README.md"
MLPERF_LICENSE_PATH = APP_ROOT / "LICENSE.mlperf-tiny"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
EXTRACTOR_PATH = REPOSITORY_ROOT / "scripts" / "extract_mlperf_resnet_samples.py"
LOCAL_TEST_BATCH = VTA_ROOT / "apps" / "mlperf_tiny_benchmark" / "cifar-10-batches-py" / "test_batch"

MODEL_SHA256 = "fb17ae9c1b6d0e5bd97f0f35024f207556261d7310b249716c87cc0628214b0e"
MLPERF_LICENSE_SHA256 = "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594"
EXPECTED_INPUT_NAME = "serving_default_input_5:0"
EXPECTED_INPUT_SHAPE = [1, 32, 32, 3]
EXPECTED_OUTPUT_NAME = "StatefulPartitionedCall:0"
EXPECTED_OUTPUT_SHAPE = [1, 10]
EXPECTED_TFLITE_OPERATORS = [
    "CONV_2D", "CONV_2D", "CONV_2D", "ADD",
    "CONV_2D", "CONV_2D", "CONV_2D", "ADD",
    "CONV_2D", "CONV_2D", "CONV_2D", "ADD",
    "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX",
]
EXPECTED_OPERATOR_CODES = [3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 1, 22, 9, 25]
EXPECTED_CONV_CHANNELS = [40, 40, 40, 80, 80, 80, 160, 160, 160]
TEST_BATCH_SHA256 = "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831"

EXPECTED_DEPENDENCIES = {"tflite": "2.10.0", "Pillow": "11.3.0"}
EXPECTED_SAMPLES = [
    {
        "filename": "00-airplane.png", "test_index": 3, "numeric_label": 0,
        "class_name": "airplane", "original_filename": "jetliner_s_001705.png",
        "png_sha256": "87b80873eac1a01a4c0adee060af11b95e2fa40060c7aa81e12ca3eca697eac0",
        "raw_rgb_sha256": "e5fcaa9fcb576b0b987f1f26526f5d22d3772f86e10879d3679bf5ee7275c22b",
    },
    {
        "filename": "01-automobile.png", "test_index": 6, "numeric_label": 1,
        "class_name": "automobile", "original_filename": "shooting_brake_s_000973.png",
        "png_sha256": "4e860efc4fdfbd16789f350b8c78f85596e35b28a85548328c57e92228c3ffd5",
        "raw_rgb_sha256": "b5f3208ae1e44485de30b7f1fe6ab78b71fbde7243549ee53320a8a0dc56ba1f",
    },
    {
        "filename": "02-bird.png", "test_index": 25, "numeric_label": 2,
        "class_name": "bird", "original_filename": "gamecock_s_000228.png",
        "png_sha256": "f39d098f01c6049cbef31025220bf4a0d92c009aa8610fa6cf09c52ffd6577be",
        "raw_rgb_sha256": "92329e7a23c326d350fa9ffbca4cc00a8a3bbb1f13ae34e612d27437432503c9",
    },
    {
        "filename": "03-cat.png", "test_index": 0, "numeric_label": 3,
        "class_name": "cat", "original_filename": "domestic_cat_s_000907.png",
        "png_sha256": "c485048fa353c1aa1714eea96c1a3839c4e48f91c69af5e34a6f1e7b46f6eda0",
        "raw_rgb_sha256": "af12e1241a1e7a0c9ba326825279bef5fe961ceb5f95212a99f9b1f22711f913",
    },
    {
        "filename": "04-deer.png", "test_index": 22, "numeric_label": 4,
        "class_name": "deer", "original_filename": "wapiti_s_001434.png",
        "png_sha256": "87c62e1a695418363bea6fba3836472c66506085b1b832fadb1ba762f2eb8eb3",
        "raw_rgb_sha256": "bf508685de95147ca5c33e7c0fc2b6ef4be13630e601579f991912e4057ede94",
    },
    {
        "filename": "05-dog.png", "test_index": 12, "numeric_label": 5,
        "class_name": "dog", "original_filename": "toy_spaniel_s_001592.png",
        "png_sha256": "7f40800e30e3197b9f42588cd5d3490ce6689c7937df4c588ce6d44fd1800322",
        "raw_rgb_sha256": "3dc597d07305887d229fe237847e86d84d38829dd481024383886b5091da35e9",
    },
    {
        "filename": "06-frog.png", "test_index": 4, "numeric_label": 6,
        "class_name": "frog", "original_filename": "green_frog_s_001658.png",
        "png_sha256": "05c9a2e401811fa7868d06aa1ea1a62581fc42950ef67948280106e4e2ce483f",
        "raw_rgb_sha256": "1a0f09360e4f0848b5ab172cef39ad2e363f044467b01234e57ea83854246e73",
    },
    {
        "filename": "07-horse.png", "test_index": 13, "numeric_label": 7,
        "class_name": "horse", "original_filename": "lippizan_s_000752.png",
        "png_sha256": "4f650f68f31eb599c6dd60f46ec631867dbe2635835cbf164eb148e42677aa5b",
        "raw_rgb_sha256": "0f4e4314cb4faf4744f289486e4e3ac79beab51a8abc9e4a3f44a602bfe73e7f",
    },
    {
        "filename": "08-ship.png", "test_index": 1, "numeric_label": 8,
        "class_name": "ship", "original_filename": "hydrofoil_s_000078.png",
        "png_sha256": "fd98ba9ab3c8b34a15e17c807dfab5cf6c8913e447979ae53774f48ba6ccd293",
        "raw_rgb_sha256": "df20a6f4102c83a7464ad1e47b60fb267ab02e970e13a081e95b223770f5a206",
    },
    {
        "filename": "09-truck.png", "test_index": 11, "numeric_label": 9,
        "class_name": "truck", "original_filename": "dustcart_s_000045.png",
        "png_sha256": "42e228989c5ab621ee9a6a212a8eacd7cb63a56d06c9b0755a21119c8b88c6a2",
        "raw_rgb_sha256": "fe16b75a074dc51896b1a7d6c5421d781e51d76aae559a1c42896652103eddbd",
    },
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_local_test_batch():
    with LOCAL_TEST_BATCH.open("rb") as stream:
        return pickle.load(stream, encoding="bytes")


def _batch_rgb(batch, index):
    chw = np.asarray(batch[b"data"][index], dtype="uint8").reshape(3, 32, 32)
    return np.transpose(chw, (1, 2, 0))


def test_model_is_the_exact_unmodified_mlperf_tiny_v14_large_float_artifact():
    assert _sha256(MODEL_PATH) == MODEL_SHA256
    assert _sha256(MLPERF_LICENSE_PATH) == MLPERF_LICENSE_SHA256

    provenance = MODEL_README_PATH.read_text(encoding="utf-8")
    required_provenance = [
        "MLPerf Tiny v1.4",
        "benchmark/training/image_classification/trained_models/pretrainedResnet_large_float.tflite",
        MODEL_SHA256,
        "byte-for-byte",
        "Apache-2.0",
        "LICENSE.mlperf-tiny",
        'calibrate_mode="global_scale"',
        "global_scale=8.0",
        "skip_conv_layers=[0]",
        f"Input tensor: `{EXPECTED_INPUT_NAME}`, float32 NHWC `{EXPECTED_INPUT_SHAPE}`",
        f"Output tensor: `{EXPECTED_OUTPUT_NAME}`, float32 `{EXPECTED_OUTPUT_SHAPE}`",
        f"Ordered operators: `{', '.join(EXPECTED_TFLITE_OPERATORS)}`",
        f"Convolution output channels: `{'/'.join(str(channel) for channel in EXPECTED_CONV_CHANNELS)}`",
    ]
    for expected in required_provenance:
        assert expected in provenance


def test_model_flatbuffer_has_the_approved_resnet8_large_topology():
    import tflite

    model = tflite.Model.GetRootAsModel(MODEL_PATH.read_bytes(), 0)
    assert model.Version() == 3
    assert model.SubgraphsLength() == 1

    graph = model.Subgraphs(0)
    assert graph.InputsLength() == 1
    assert graph.OutputsLength() == 1
    input_tensor = graph.Tensors(graph.Inputs(0))
    output_tensor = graph.Tensors(graph.Outputs(0))
    assert input_tensor.Name().decode("utf-8") == EXPECTED_INPUT_NAME
    assert output_tensor.Name().decode("utf-8") == EXPECTED_OUTPUT_NAME
    assert list(input_tensor.ShapeAsNumpy()) == EXPECTED_INPUT_SHAPE
    assert input_tensor.Type() == 0  # TensorType.FLOAT32
    assert list(output_tensor.ShapeAsNumpy()) == EXPECTED_OUTPUT_SHAPE
    assert output_tensor.Type() == 0

    operator_codes = []
    conv_channels = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode()
        operator_codes.append(code)
        if code == 3:  # BuiltinOperator.CONV_2D
            weight = graph.Tensors(operator.Inputs(1))
            conv_channels.append(int(weight.Shape(0)))

    assert operator_codes == EXPECTED_OPERATOR_CODES
    assert conv_channels == EXPECTED_CONV_CHANNELS


def test_setup_pins_only_the_two_approved_additional_dependencies():
    setup_text = (REPOSITORY_ROOT / "scripts" / "setup_tvm_vta_env.sh").read_text(
        encoding="utf-8"
    )
    for package, version in EXPECTED_DEPENDENCIES.items():
        assert re.search(
            rf'(?m)^\s*["\']?{re.escape(package)}=={re.escape(version)}["\']?\s*\\?$',
            setup_text,
        )

    forbidden = ["tensorflow", "tflite-runtime", "scikit-learn", "h5py", "cmsis-nn", "autotvm"]
    lowered = setup_text.lower()
    for package in forbidden:
        assert package not in lowered


def test_manifest_records_exact_selection_provenance_and_cifar_license_status():
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["selection"] == "first test_batch occurrence of numeric labels 0 through 9"
    assert manifest["samples"] == EXPECTED_SAMPLES
    assert sorted(sample["filename"] for sample in manifest["samples"]) == sorted(
        path.name for path in MANIFEST_PATH.parent.glob("*.png")
    )

    dataset = manifest["dataset"]
    assert dataset == {
        "name": "CIFAR-10",
        "version": "Python archive",
        "source_url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "archive": {
            "filename": "cifar-10-python.tar.gz",
            "md5": "c58f30108f718f92721af3b95e74349a",
        },
        "test_batch_sha256": TEST_BATCH_SHA256,
        "attribution": "Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton",
        "citation": (
            "Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. "
            "Technical report, 2009."
        ),
        "citation_url": "https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf",
        "license": {
            "status": "not-declared-by-source",
            "notice": (
                "The official CIFAR-10 distribution page does not declare an open-source license."
            ),
            "source_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        },
    }


def test_committed_pngs_match_exact_bytes_and_decode_as_lossless_rgb():
    from PIL import Image

    for sample in EXPECTED_SAMPLES:
        image_path = MANIFEST_PATH.parent / sample["filename"]
        assert _sha256(image_path) == sample["png_sha256"]
        with Image.open(image_path) as image:
            image.load()
            assert image.mode == "RGB"
            assert image.size == (32, 32)
            pixels = np.asarray(image)
        assert pixels.dtype == np.uint8
        assert pixels.shape == (32, 32, 3)
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == sample["raw_rgb_sha256"]


def test_committed_pngs_match_optional_local_test_batch_exactly():
    if not LOCAL_TEST_BATCH.exists():
        return

    from PIL import Image

    assert _sha256(LOCAL_TEST_BATCH) == TEST_BATCH_SHA256
    batch = _load_local_test_batch()
    for sample in EXPECTED_SAMPLES:
        index = sample["test_index"]
        assert int(batch[b"labels"][index]) == sample["numeric_label"]
        assert batch[b"filenames"][index].decode("utf-8") == sample["original_filename"]
        with Image.open(MANIFEST_PATH.parent / sample["filename"]) as image:
            actual = np.asarray(image.convert("RGB"))
        np.testing.assert_array_equal(actual, _batch_rgb(batch, index))


def test_extractor_recreates_committed_manifest_and_png_hashes(tmp_path):
    if not LOCAL_TEST_BATCH.exists():
        return

    subprocess.run(
        [
            sys.executable,
            str(EXTRACTOR_PATH),
            "--test-batch",
            str(LOCAL_TEST_BATCH),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )

    recreated_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert recreated_manifest == _manifest()
    for sample in EXPECTED_SAMPLES:
        assert _sha256(tmp_path / sample["filename"]) == sample["png_sha256"]
