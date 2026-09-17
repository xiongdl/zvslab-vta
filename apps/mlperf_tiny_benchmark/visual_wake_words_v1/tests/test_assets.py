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

"""Byte-level provenance and structural contracts for VWW assets."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
VTA_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = APP_ROOT / "model" / "vww_96_float.tflite"
MODEL_README_PATH = APP_ROOT / "model" / "README.md"
MLPERF_LICENSE_PATH = APP_ROOT / "LICENSE.mlperf-tiny"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
LOCAL_DATASET_ROOT = VTA_ROOT.parent / ".envs" / "vw_coco2014_96"

MODEL_SHA256 = "115bbc094d2119561320a21f01b6500a18bea8cc8589282ab007097bec8af38c"
EXPECTED_SAMPLES = [
    ("00-non-person-000000000009.jpg", "non_person", 0, "d8f0e1e6e7635f189ab52e3e98aef1f7d734814a1fbe41fdb2c5ff8cbfc6dcfc", "non_person/COCO_train2014_000000000009.jpg"),
    ("01-non-person-000000000025.jpg", "non_person", 0, "d635da6fef7b8968653bdfc1289d8cfa88b551010e8d85ed1f384f7a6e60a2b2", "non_person/COCO_train2014_000000000025.jpg"),
    ("02-non-person-000000000030.jpg", "non_person", 0, "c69bcd53f365ec472243f32a4ef9e0a00f389c1c10b3f892ece2392223f92fd6", "non_person/COCO_train2014_000000000030.jpg"),
    ("03-non-person-000000000034.jpg", "non_person", 0, "fb4b1a2d533b1935103e5083a7d17b22a7dabfdffb2ddf09f75a98f19337c27d", "non_person/COCO_train2014_000000000034.jpg"),
    ("04-non-person-000000000064.jpg", "non_person", 0, "21eab2520d5dd01d3a10ce6b322a882bc7c6856ac50b3fbf1317362dcf7f53a0", "non_person/COCO_train2014_000000000064.jpg"),
    ("05-person-000000000036.jpg", "person", 1, "068b7ad53d46c9c075b47df4505bd6369a14cd4bcbad06c1e679dc7dfc51156f", "person/COCO_train2014_000000000036.jpg"),
    ("06-person-000000000049.jpg", "person", 1, "23959ed66eb87f75c51895b174024ea8af0da9fc2e301c52045a8a01332a8bc2", "person/COCO_train2014_000000000049.jpg"),
    ("07-person-000000000077.jpg", "person", 1, "18f612634fd4d1636181b96bc7fed3eaf61b0e38d4bf19c26f4011db75130868", "person/COCO_train2014_000000000077.jpg"),
    ("08-person-000000000086.jpg", "person", 1, "c4cb9d7b9f7a6e8b2ffc17acbbdcd74d062b43849ef16bd124eca572df316254", "person/COCO_train2014_000000000086.jpg"),
    ("09-person-000000000110.jpg", "person", 1, "577704fec55b656ca27e4ce444b9767b9bd20e196b16952e400d52747363abda", "person/COCO_train2014_000000000110.jpg"),
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_model_is_the_authenticated_mlperf_tiny_v14_float_artifact():
    assert _sha256(MODEL_PATH) == MODEL_SHA256
    provenance = MODEL_README_PATH.read_text(encoding="utf-8")
    for expected in (
        "MLPerf Tiny v1.4",
        "benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite",
        MODEL_SHA256,
        "byte-for-byte",
        "Apache-2.0",
        "LICENSE.mlperf-tiny",
        'calibrate_mode="global_scale"',
        "global_scale=8.0",
        "skip_conv_layers=[0]",
    ):
        assert expected in provenance


def test_model_flatbuffer_has_the_approved_vww_contract():
    import tflite

    model = tflite.Model.GetRootAsModel(MODEL_PATH.read_bytes(), 0)
    assert model.Version() == 3
    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)
    assert graph.InputsLength() == 1
    assert graph.OutputsLength() == 1
    input_tensor = graph.Tensors(graph.Inputs(0))
    output_tensor = graph.Tensors(graph.Outputs(0))
    assert input_tensor.Name() == b"input_1"
    assert list(input_tensor.ShapeAsNumpy()) == [1, 96, 96, 3]
    assert input_tensor.Type() == int(tflite.TensorType.FLOAT32)
    assert output_tensor.Name() == b"Identity"
    assert list(output_tensor.ShapeAsNumpy()) == [1, 2]
    assert output_tensor.Type() == int(tflite.TensorType.FLOAT32)

    expected = []
    for index in range(13):
        expected.extend(
            [
                int(tflite.BuiltinOperator.CONV_2D),
                int(tflite.BuiltinOperator.DEPTHWISE_CONV_2D),
            ]
        )
    expected.extend(
        [
            int(tflite.BuiltinOperator.CONV_2D),
            int(tflite.BuiltinOperator.AVERAGE_POOL_2D),
            int(tflite.BuiltinOperator.RESHAPE),
            int(tflite.BuiltinOperator.FULLY_CONNECTED),
            int(tflite.BuiltinOperator.SOFTMAX),
        ]
    )
    actual = [
        int(model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode())
        for i in range(graph.OperatorsLength())
    ]
    assert actual == expected

    channels = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        if actual[index] == int(tflite.BuiltinOperator.CONV_2D):
            channels.append(int(graph.Tensors(operator.Inputs(1)).Shape(0)))
    assert channels == [8, 16, 32, 32, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256]


def test_manifest_records_balanced_selection_and_separate_dataset_provenance():
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["selection"] == "lexicographically first five JPEG files from each class directory"
    assert manifest["class_mapping"] == {"0": "non_person", "1": "person"}
    assert manifest["dataset"] == {
        "name": "Visual Wake Words COCO 2014-derived dataset",
        "root": "vw_coco2014_96",
        "source": "user-provided local dataset directory",
        "license": {
            "status": "not-declared-by-source",
            "notice": "The local COCO-derived image dataset license was not established by the source directory.",
        },
    }
    assert [
        (item["filename"], item["class_name"], item["label"], item["sha256"], item["source_relative_path"])
        for item in manifest["samples"]
    ] == EXPECTED_SAMPLES


def test_committed_jpegs_match_exact_bytes_and_decode_as_rgb():
    manifest = _manifest()
    assert sorted(path.name for path in MANIFEST_PATH.parent.glob("*.jpg")) == [item[0] for item in EXPECTED_SAMPLES]
    assert _sha256(MLPERF_LICENSE_PATH) == "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594"
    for item in manifest["samples"]:
        path = MANIFEST_PATH.parent / item["filename"]
        assert _sha256(path) == item["sha256"]
        with Image.open(path) as image:
            image.load()
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.size == (96, 96)
            pixels = np.asarray(image)
        assert pixels.dtype == np.uint8
        assert pixels.shape == (96, 96, 3)


def test_committed_jpegs_match_optional_local_dataset_exactly():
    if not LOCAL_DATASET_ROOT.exists():
        return
    for filename, _, _, digest, source_relative_path in EXPECTED_SAMPLES:
        source = LOCAL_DATASET_ROOT / source_relative_path
        assert source.is_file()
        assert _sha256(source) == digest
        assert (MANIFEST_PATH.parent / filename).read_bytes() == source.read_bytes()
