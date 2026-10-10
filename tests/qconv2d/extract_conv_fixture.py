#!/usr/bin/env python3
"""Extract the first quantized CONV_2D and one reproducible CIFAR sample."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import platform

import numpy as np
import tensorflow as tf

from fixture import quantize_multiplier


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(model: Path, cifar_batch: Path, sample_index: int, output_dir: Path) -> None:
    model = model.resolve()
    cifar_batch = cifar_batch.resolve()
    with cifar_batch.open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    pixels = np.asarray(batch[b"data"][sample_index], dtype=np.uint8)
    image = (pixels.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.int16) - 128).astype(np.int8)
    interpreter = tf.lite.Interpreter(
        model_path=str(model), experimental_delegates=[], experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    conv = next(op for op in interpreter._get_ops_details() if op["op_name"] == "CONV_2D")
    if list(conv["inputs"]).__len__() != 3:
        raise ValueError("first CONV_2D must have input, filter, and bias tensors")
    input_index, weight_index, bias_index = (int(index) for index in conv["inputs"])
    output_index = int(conv["outputs"][0])
    details = {int(item["index"]): item for item in interpreter.get_tensor_details()}
    input_detail, weight_detail = details[input_index], details[weight_index]
    bias_detail, output_detail = details[bias_index], details[output_index]
    if tuple(input_detail["shape"]) != (1, 32, 32, 3):
        raise ValueError(f"unexpected first convolution input shape: {input_detail['shape']}")
    input_quant = input_detail["quantization_parameters"]
    weight_quant = weight_detail["quantization_parameters"]
    output_quant = output_detail["quantization_parameters"]
    weight = interpreter.get_tensor(weight_index).astype(np.int8, copy=True)
    bias = interpreter.get_tensor(bias_index).astype(np.int32, copy=True)
    input_scale = float(input_quant["scales"][0])
    output_scale = float(output_quant["scales"][0])
    weight_scales = np.asarray(weight_quant["scales"], dtype=np.float64)
    multipliers_shifts = [quantize_multiplier(input_scale * float(scale) / output_scale)
                          for scale in weight_scales]
    multiplier = np.asarray([item[0] for item in multipliers_shifts], dtype=np.int32)
    shift = np.asarray([item[1] for item in multipliers_shifts], dtype=np.int32)
    interpreter.set_tensor(input_index, image[None, ...])
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_index).astype(np.int8, copy=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "fixture.npz", input=image[None, ...], weight=weight, bias=bias,
        multiplier=multiplier, shift=shift, tflite_output=tflite_output,
    )
    packed = b"".join(array.tobytes(order="C") for array in
                      (image[None, ...], weight, bias, multiplier, shift))
    (output_dir / "fixture.bin").write_bytes(packed)
    metadata = {
        "model": str(model), "model_sha256": sha256(model.read_bytes()),
        "input_source": str(cifar_batch), "input_sha256": sha256(image.tobytes()),
        "sample_index": sample_index, "input_preprocessing": "uint8 pixel values minus 128, cast int8",
        "tensor_indices": {"input": input_index, "weight": weight_index,
                           "bias": bias_index, "output": output_index},
        "operator": {"name": conv["op_name"], "index": int(conv["index"]),
                     "padding": "SAME", "stride": [1, 1], "dilation": [1, 1],
                     "fused_activation": "RELU"},
        "quantization": {
            "input_scale": input_scale, "input_zero_point": int(input_quant["zero_points"][0]),
            "weight_scale": weight_scales.tolist(),
            "weight_zero_point": np.asarray(weight_quant["zero_points"], dtype=np.int32).tolist(),
            "weight_quantized_dimension": int(weight_quant["quantized_dimension"]),
            "bias_scale": np.asarray(bias_detail["quantization_parameters"]["scales"],
                                      dtype=np.float64).tolist(),
            "bias_zero_point": np.asarray(bias_detail["quantization_parameters"]["zero_points"],
                                          dtype=np.int32).tolist(),
            "output_scale": output_scale,
            "output_zero_point": int(output_quant["zero_points"][0]),
        },
        "runtime": {"tensorflow": tf.__version__, "numpy": np.__version__,
                    "python": platform.python_version()},
        "fixture_sha256": sha256((output_dir / "fixture.npz").read_bytes()),
        "driver_fixture_sha256": sha256(packed),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cifar-batch", required=True, type=Path)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    extract(args.model, args.cifar_batch, args.sample_index, args.output_dir)


if __name__ == "__main__":
    main()
