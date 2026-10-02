"""Capture actual pre-schedule VTA computations from a prepared Relay module."""

import hashlib
import json
import math
from dataclasses import dataclass

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
