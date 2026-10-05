"""Capture and serialize actual pre-schedule VTA workloads."""

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tvm
from tvm import autotvm, relay
from tvm.autotvm.task.dispatcher import DispatchContext
from tvm.relay.backend import te_compiler


@dataclass(frozen=True)
class TensorDescription:
    shape: tuple
    dtype: str


@dataclass(frozen=True)
class LayerCompute:
    occurrence: int
    symbol: str
    function: relay.Function
    compute: relay.Function
    compute_constants: tuple
    compute_sha256: str
    inputs: tuple
    output: TensorDescription
    input_layout: str
    kernel_layout: str
    output_layout: str
    constants: dict
    template: str
    workload: tuple
    config_space_size: int
    config_space_identity: str
    config_spaces: tuple


@dataclass(frozen=True)
class DeploymentCompute:
    model_id: str
    model_sha256: str
    geometry: dict
    geometry_sha256: str
    layers: tuple


def _tensor_description(value):
    checked = value if isinstance(value, relay.TensorType) else value.checked_type
    return TensorDescription(tuple(int(dim) for dim in checked.shape), str(checked.dtype))


def _layer_constants(func):
    composite = next(
        call.op
        for call in _outlined_calls(func)
        if call.op.attrs.get_str("Composite") == "vta.qnn_conv2d"
    )
    body = composite.body
    cast = body
    clip = cast.args[0]
    shifted = clip.args[0]
    biased = shifted.args[0]
    conv = biased.args[0] if isinstance(biased, relay.Call) and biased.op.name in ("add", "nn.bias_add") else biased
    bias = biased.args[1] if conv is not biased else None
    return {
        "bias": _constant_summary(bias),
        "shift": _constant_summary(shifted.args[1]),
        "clip_min": float(clip.attrs.a_min),
        "clip_max": float(clip.attrs.a_max),
        "output_dtype": str(cast.attrs.dtype),
        "weight_sha256": hashlib.sha256(tvm.ir.save_json(conv.args[1]).encode("utf-8")).hexdigest(),
    }, conv


def _constant_summary(expr):
    if not isinstance(expr, relay.Constant):
        raise ValueError("VTA deployment post-processing constants must be bound Relay constants")
    values = expr.data.numpy()
    flattened = values.reshape(-1)
    unique = set(flattened.tolist())
    if len(unique) == 1:
        value = next(iter(unique))
        return int(value) if isinstance(value, (int, bool)) else float(value)
    return [int(value) if isinstance(value, (int, bool)) else float(value) for value in flattened]


def _outlined_calls(func):
    calls = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, relay.Function):
            calls.append(node)

    relay.analysis.post_order_visit(func.body, visit)
    return calls


def _function_sha256(func):
    return hashlib.sha256(tvm.ir.save_json(func).encode("utf-8")).hexdigest()


def _geometry(config):
    return {
        "batch": config.batch,
        "block_in": config.block_in,
        "block_out": config.block_out,
        "input_dtype": config.input_dtype,
        "weight_dtype": config.weight_dtype,
        "accumulator_dtype": config.accumulator_dtype,
        "output_dtype": config.output_dtype,
        "target": config.target,
        "host_target": config.host_target,
        "model": config.model,
        "device_type": config.device_type,
    }


def _capture_layer(occurrence, function, compiler_config):
    from vta.relay import transform

    transform._validate_vta_function(function, compiler_config)
    constants, conv = _layer_constants(function)
    compute, compute_constants = transform.capture_vta_compute(function, compiler_config)
    class CaptureConfigSpace(DispatchContext):
        def __init__(self):
            super().__init__()
            self.queries = []

        def _query_inside(self, target, workload):
            space = autotvm.task.ConfigSpace()
            self.queries.append((target, workload, space))
            return space

        def update(self, target, workload, config):
            # TOPI may rewrite a workload after the initial template query;
            # the original queried ConfigSpace is the tuning contract.
            return None

    compiler = te_compiler.get()
    compiler.clear()
    capture_context = CaptureConfigSpace()
    try:
        with capture_context:
            from vta.relay import transform

            transform.lower_vta_compute(compute, compiler_config)
    finally:
        compiler.clear()
    unique = {
        (str(target), tuple(workload)): space
        for target, workload, space in capture_context.queries
    }
    conv_entries = [item for item in unique.items() if item[0][1][0] == "conv2d_packed.vta"]
    if len(conv_entries) != 1:
        raise ValueError(
            f"outlined VTA occurrence {occurrence} must query one conv2d template, got "
            f"{[workload[0] for _, workload in unique]}"
        )
    config_spaces = tuple(
        (key[1][0], key[1], key[0], space)
        for key, space in sorted(unique.items(), key=lambda item: repr(item[0][1]))
    )
    if any(len(space) <= 0 for _, _, _, space in config_spaces):
        raise ValueError(f"captured AutoTVM config space is empty for occurrence {occurrence}")
    unknown = {template for template, _, _, _ in config_spaces} - {
        "conv2d_packed.vta", "add.vta"
    }
    if unknown:
        raise ValueError(f"unsupported captured VTA schedule templates: {sorted(unknown)}")
    conv_key, conv_space = conv_entries[0]
    _, conv_workload = conv_key
    output_type = function.ret_type
    config_identity = {
        "spaces": [
            {
                "template": template,
                "workload": repr(workload),
                "target": target,
                "size": len(space),
                "entities": [str(space.get(index)) for index in range(len(space))],
            }
            for template, workload, target, space in config_spaces
        ],
    }
    data_layout = str(conv.attrs.data_layout)
    return LayerCompute(
        occurrence=occurrence,
        symbol=function.attrs.get_str("global_symbol"),
        function=function,
        compute=compute,
        compute_constants=tuple(compute_constants),
        compute_sha256=_function_sha256(function),
        inputs=tuple(_tensor_description(param) for param in function.params),
        output=_tensor_description(output_type),
        input_layout=data_layout,
        kernel_layout=str(conv.attrs.kernel_layout),
        output_layout=str(conv.attrs.out_layout) or data_layout,
        constants=constants,
        template=conv_workload[0],
        workload=conv_workload,
        config_space_size=math.prod(len(space) for _, _, _, space in config_spaces),
        config_space_identity=hashlib.sha256(
            json.dumps(config_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        config_spaces=config_spaces,
    )


def capture_deployment_compute(module, model_id, model_sha256, config=None):
    """Capture ordered outlined VTA layers and their actual AutoTVM spaces.

    `module` must be the prepared model's real partitioned Relay module. The
    computation exposed to tuning is legalized by the same VTA transform used
    by ordinary deployment and is captured before any schedule is selected.
    """
    if not isinstance(module, tvm.IRModule):
        raise TypeError("module must be a prepared tvm.IRModule")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(model_sha256, str) or not model_sha256:
        raise ValueError("model_sha256 must be a non-empty string")

    import vta
    from vta.relay import transform

    compiler_config = config or transform.VTACompilerConfig.from_env(vta.get_env())
    functions = transform._collect_vta_relay_functions(module)
    if not functions:
        raise ValueError("prepared module has no VTA deployment functions")
    outlined = relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols("vta")(module)
    rows = transform._global_vta_relay_functions(outlined)
    global_handles = {function.handle.value for _, function in rows}
    nested = [func for func in transform._collect_vta_relay_functions(outlined) if func.handle.value not in global_handles]
    if nested:
        raise ValueError("all nested Compiler='vta' functions must be directly outlineable")

    geometry = _geometry(compiler_config)
    geometry_json = json.dumps(geometry, sort_keys=True, separators=(",", ":"))
    layers = tuple(
        _capture_layer(index, function, compiler_config)
        for index, (_, function) in enumerate(rows)
    )
    return DeploymentCompute(
        model_id=model_id,
        model_sha256=model_sha256,
        geometry=geometry,
        geometry_sha256=hashlib.sha256(geometry_json.encode("utf-8")).hexdigest(),
        layers=layers,
    )


FORMAT = "resnet8_large-vta-workloads"
VERSION = 1
MAX_WORKLOAD_FILE_BYTES = 128 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUANTIZATION = {"calibrate_mode": "global_scale", "global_scale": 8.0, "skip_conv_layers": [0]}


@dataclass(frozen=True)
class WorkloadLayer:
    index: int
    symbol: str
    function: relay.Function
    compute_sha256: str
    inputs: tuple
    output: tuple
    activation: np.ndarray
    config_space_identity: str


@dataclass(frozen=True)
class WorkloadSnapshot:
    model_sha256: str
    input_sha256: str
    config_basename: str
    config_sha256: str
    config_bytes: bytes
    geometry: dict
    geometry_sha256: str
    tvm_version: str
    vta_version: str
    layers: tuple


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def portable_geometry(geometry):
    """Keep hardware dimensions while removing backend and host target names."""
    keys = (
        "batch", "block_in", "block_out", "input_dtype", "weight_dtype",
        "accumulator_dtype", "output_dtype",
    )
    return {key: geometry[key] for key in keys}


def portable_config_space_identity(layer):
    """Hash tunable entities while omitting backend-specific target model tags."""
    spaces = []
    for template, workload, target, space in layer.config_spaces:
        # ALU-only AutoTVM registration is backend-specific in VTA's TOPI
        # package; the shared GEMM config space is the portable tuning contract.
        if template != "conv2d_packed.vta":
            continue
        portable_target = re.sub(r"\s+-model=[^\s]+", "", str(target))
        spaces.append({
            "template": template,
            "workload": repr(workload),
            "target": portable_target,
            "size": len(space),
            "entities": [str(space.get(index)) for index in range(len(space))],
        })
    if not spaces:
        raise ValueError("VTA workload has no portable packed-convolution config space")
    return _sha256(_canonical({"spaces": spaces}))


def _require_hash(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _tensor(array):
    value = np.ascontiguousarray(array)
    dtype = value.dtype
    if dtype.hasobject or dtype.kind not in "biuf":
        raise ValueError(f"unsupported tensor dtype {dtype}")
    raw = value.tobytes(order="C")
    return {
        "shape": [int(dim) for dim in value.shape],
        "dtype": dtype.str,
        "byte_order": dtype.byteorder,
        "encoding": "base64",
        "data": base64.b64encode(raw).decode("ascii"),
        "sha256": _sha256(raw),
    }


def _decode_tensor(value, label):
    if not isinstance(value, dict) or value.get("encoding") != "base64":
        raise ValueError(f"{label} must use base64 tensor encoding")
    shape = value.get("shape")
    if (not isinstance(shape, list) or not shape
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape)):
        raise ValueError(f"{label} has an invalid shape")
    try:
        dtype = np.dtype(value.get("dtype"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} has an invalid dtype") from error
    if dtype.hasobject or dtype.kind not in "biuf":
        raise ValueError(f"{label} has an unsupported dtype")
    if value.get("byte_order") != dtype.byteorder:
        raise ValueError(f"{label} byte order does not match its dtype")
    encoded = value.get("data")
    if not isinstance(encoded, str):
        raise ValueError(f"{label} has no encoded bytes")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError(f"{label} has malformed base64 tensor bytes") from error
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"{label} tensor byte size does not match shape and dtype")
    if _sha256(raw) != _require_hash(value.get("sha256"), f"{label} tensor sha256"):
        raise ValueError(f"{label} tensor hash mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()


def make_layer_record(*, index, symbol, function, compute_sha256, inputs,
                      input_dtypes, output_shape, output_dtype, activation,
                      config_space_identity):
    """Create a serializable record from one captured outlined VTA layer."""
    if not isinstance(function, relay.Function):
        raise TypeError("workload function must be a Relay Function")
    if len(inputs) != len(input_dtypes):
        raise ValueError("workload input shapes and dtypes differ in length")
    function_json = tvm.ir.save_json(function)
    function_bytes = function_json.encode("utf-8")
    value = np.ascontiguousarray(activation)
    if len(inputs) != 1:
        raise ValueError("ResNet-8 Large VTA workloads must have exactly one activation input")
    if tuple(value.shape) != tuple(inputs[0]) or value.dtype.name != input_dtypes[0]:
        raise ValueError("captured activation tensor does not match the Relay input contract")
    return {
        "index": int(index),
        "symbol": str(symbol),
        "function": {"encoding": "tvm.ir.save_json", "data": function_json,
                     "sha256": _sha256(function_bytes)},
        "compute_sha256": _require_hash(compute_sha256, "compute sha256"),
        "inputs": [
            {"shape": [int(dim) for dim in shape], "dtype": str(dtype)}
            for shape, dtype in zip(inputs, input_dtypes)
        ],
        "output": {"shape": [int(dim) for dim in output_shape], "dtype": str(output_dtype)},
        "activation": _tensor(value),
        "config_space_identity": _require_hash(config_space_identity, "config space identity"),
    }


def make_document(*, model_sha256, input_sha256, config_bytes, config_basename,
                  geometry, tvm_version, vta_version, workloads):
    config_bytes = bytes(config_bytes)
    config_sha = _sha256(config_bytes)
    geometry_sha = _sha256(_canonical(geometry))
    return {
        "format": FORMAT,
        "version": VERSION,
        "model": {"sha256": _require_hash(model_sha256, "model sha256"),
                  "quantization": dict(_QUANTIZATION)},
        "input": {"sha256": _require_hash(input_sha256, "input sha256"),
                  "decoded_shape": [1, 32, 32, 3], "decoded_dtype": "float32"},
        "config": {"basename": Path(config_basename).name,
                   "raw_base64": base64.b64encode(config_bytes).decode("ascii"),
                   "sha256": config_sha, "geometry": geometry,
                   "geometry_sha256": geometry_sha},
        "compatibility": {"tvm_version": str(tvm_version), "vta_version": str(vta_version),
                          "relay_format": "tvm.ir.save_json"},
        "workloads": list(workloads),
    }


def _seal(document):
    value = dict(document)
    value.pop("snapshot_sha256", None)
    return _sha256(_canonical(value))


def write_workloads(document, path):
    """Write a sealed JSON snapshot with same-directory atomic replacement."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(document)
    value["snapshot_sha256"] = _seal(value)
    encoded = json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if len(encoded) > MAX_WORKLOAD_FILE_BYTES:
        raise ValueError("workloads file exceeds maximum supported size")
    fd, staging_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging_name, path)
    except Exception:
        try:
            os.unlink(staging_name)
        except FileNotFoundError:
            pass
        raise
    return path


def _load_function(value, index):
    if not isinstance(value, dict) or value.get("encoding") != "tvm.ir.save_json":
        raise ValueError(f"workload {index} has unsupported Relay serialization")
    encoded = value.get("data")
    if not isinstance(encoded, str):
        raise ValueError(f"workload {index} has no serialized Relay function")
    raw = encoded.encode("utf-8")
    if _sha256(raw) != _require_hash(value.get("sha256"), f"workload {index} Relay sha256"):
        raise ValueError(f"workload {index} Relay function hash mismatch")
    try:
        function = tvm.ir.load_json(encoded)
        if not isinstance(function, relay.Function):
            raise ValueError("serialized object is not a Relay Function")
        module = tvm.IRModule.from_expr(function)
        module = relay.transform.InferType()(module)
        restored = [item for item in module.functions.values() if isinstance(item, relay.Function)]
        if len(restored) != 1:
            raise ValueError("serialized workload must contain exactly one Relay Function")
        return restored[0]
    except Exception as error:
        raise ValueError(f"workload {index} Relay function cannot be restored") from error


def load_workloads(path, *, validate_config_space=True):
    """Validate and restore a snapshot, then optionally requery current VTA spaces."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"workloads file does not exist: {path}")
    if path.stat().st_size > MAX_WORKLOAD_FILE_BYTES:
        raise ValueError("workloads file exceeds maximum supported size")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("workloads file is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or document.get("format") != FORMAT or document.get("version") != VERSION:
        raise ValueError("unsupported workloads format or version")
    if document.get("snapshot_sha256") != _seal(document):
        raise ValueError("workloads snapshot integrity hash mismatch")
    model = document.get("model")
    source_input = document.get("input")
    config = document.get("config")
    compatibility = document.get("compatibility")
    if not all(isinstance(value, dict) for value in (model, source_input, config, compatibility)):
        raise ValueError("workloads provenance sections are missing")
    model_sha = _require_hash(model.get("sha256"), "model sha256")
    input_sha = _require_hash(source_input.get("sha256"), "input sha256")
    if source_input.get("decoded_shape") != [1, 32, 32, 3] or source_input.get("decoded_dtype") != "float32":
        raise ValueError("workloads decoded image tensor contract is unsupported")
    if model.get("quantization") != _QUANTIZATION:
        raise ValueError("workloads quantization policy does not match ResNet-8 Large")
    try:
        raw_config = base64.b64decode(config.get("raw_base64", ""), validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("workloads raw config is malformed base64") from error
    config_sha = _require_hash(config.get("sha256"), "config sha256")
    if _sha256(raw_config) != config_sha:
        raise ValueError("workloads config hash mismatch")
    if not isinstance(config.get("basename"), str) or Path(config["basename"]).name != config["basename"]:
        raise ValueError("workloads config basename is invalid")
    try:
        json.loads(raw_config.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("workloads raw config is not valid UTF-8 JSON") from error
    geometry = config.get("geometry")
    if not isinstance(geometry, dict) or _sha256(_canonical(geometry)) != config.get("geometry_sha256"):
        raise ValueError("workloads geometry hash mismatch")
    layers_value = document.get("workloads")
    if not isinstance(layers_value, list) or not layers_value:
        raise ValueError("workloads must contain at least one VTA layer")
    layers = []
    seen_indices = set()
    seen_symbols = set()
    if compatibility.get("relay_format") != "tvm.ir.save_json":
        raise ValueError("unsupported workloads Relay serialization format")
    for expected_index, entry in enumerate(layers_value):
        if not isinstance(entry, dict):
            raise ValueError("workload entry must be a JSON object")
        index = entry.get("index")
        symbol = entry.get("symbol")
        if (isinstance(index, bool) or not isinstance(index, int)
                or index != expected_index or index in seen_indices):
            raise ValueError("workloads must be ordered by contiguous unique nonnegative indices")
        if not isinstance(symbol, str) or not symbol or symbol in seen_symbols:
            raise ValueError("workload symbols must be unique non-empty strings")
        seen_indices.add(index)
        seen_symbols.add(symbol)
        function = _load_function(entry.get("function"), index)
        inputs = entry.get("inputs")
        output = entry.get("output")
        if not isinstance(inputs, list) or len(inputs) != len(function.params) or not isinstance(output, dict):
            raise ValueError(f"workload {index} tensor metadata does not match its Relay function")
        input_specs = []
        for param, spec in zip(function.params, inputs):
            if not isinstance(spec, dict):
                raise ValueError(f"workload {index} input metadata is invalid")
            expected_shape = tuple(int(dim) for dim in param.checked_type.shape)
            shape = spec.get("shape")
            if (not isinstance(shape, list) or any(isinstance(dim, bool) or not isinstance(dim, int) for dim in shape)
                    or tuple(shape) != expected_shape or spec.get("dtype") != str(param.checked_type.dtype)):
                raise ValueError(f"workload {index} input metadata differs from Relay function")
            input_specs.append((expected_shape, str(param.checked_type.dtype)))
        output_type = function.checked_type.ret_type
        output_shape = output.get("shape")
        if (not isinstance(output_shape, list)
                or any(isinstance(dim, bool) or not isinstance(dim, int) for dim in output_shape)
                or tuple(output_shape) != tuple(int(dim) for dim in output_type.shape)
                or output.get("dtype") != str(output_type.dtype)):
            raise ValueError(f"workload {index} output metadata differs from Relay function")
        activation = _decode_tensor(entry.get("activation"), f"workload {index} activation")
        if len(input_specs) != 1 or tuple(activation.shape) != input_specs[0][0] or activation.dtype.name != input_specs[0][1]:
            raise ValueError(f"workload {index} activation does not match Relay input")
        compute_sha = _require_hash(entry.get("compute_sha256"), f"workload {index} compute sha256")
        space_identity = _require_hash(entry.get("config_space_identity"), f"workload {index} config space identity")
        if compute_sha != entry["function"]["sha256"]:
            raise ValueError(f"workload {index} computation hash differs from serialized Relay")
        layers.append(WorkloadLayer(index, symbol, function, compute_sha, tuple(input_specs),
                                    (tuple(int(dim) for dim in output_type.shape), str(output_type.dtype)),
                                    activation, space_identity))

    if validate_config_space:
        _validate_current_environment(config_sha, geometry, compatibility, layers)
    return WorkloadSnapshot(model_sha, input_sha, str(config.get("basename")), config_sha,
                            raw_config, geometry, str(config.get("geometry_sha256")),
                            str(compatibility.get("tvm_version")),
                            str(compatibility.get("vta_version")), tuple(layers))


def _validate_current_environment(config_sha, geometry, compatibility, layers):
    import vta

    if compatibility.get("tvm_version") != tvm.__version__:
        raise ValueError("workloads TVM version is incompatible with this runtime")
    if compatibility.get("vta_version") != getattr(vta, "__version__", "source-tree"):
        raise ValueError("workloads VTA version is incompatible with this runtime")
    config_path = os.environ.get("VTA_CONFIG_FILE")
    if not config_path or _sha256(Path(config_path).read_bytes()) != config_sha:
        raise ValueError("workloads geometry config differs from VTA_CONFIG_FILE")
    current_config = vta.relay.transform.VTACompilerConfig.from_env(vta.get_env())
    if portable_geometry(_geometry(current_config)) != geometry:
        raise ValueError("workloads hardware geometry differs from current VTA geometry")
    for layer in layers:
        captured = _capture_layer(layer.index, layer.function, current_config)
        if portable_config_space_identity(captured) != layer.config_space_identity:
            raise ValueError(f"workload {layer.index} config-space identity mismatch")
