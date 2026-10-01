"""Shared AutoTVM task for the complete VTA Conv arithmetic in Tiny graphs.

The extractor operates on the prepared mixed graph, rather than the imported
float graph, so every identity follows the actual VTA routing and symbol order.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import tvm
from tvm import autotvm, relay, te, topi

from vta.top.vta_conv2d import conv2d_packed, schedule_conv2d_packed


TASK_NAME = "mlperf_tiny_fused_conv2d.vta"


def _as_tuple(value):
    return tuple(_as_tuple(item) if isinstance(item, list) else item for item in value)


@dataclass(frozen=True)
class FusedOperatorIdentity:
    """Complete, deterministic identity for one deployed VTA Conv occurrence."""

    conv_workload: tuple
    bias_shape: tuple
    bias_dtype: str | None
    bias_values: tuple
    bias_axis: int | None
    shift: int
    clip_min: int
    clip_max: int
    output_dtype: str
    symbol: str
    occurrence: int

    def canonical_json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, serialized):
        value = json.loads(serialized)
        value["conv_workload"] = _as_tuple(value["conv_workload"])
        value["bias_shape"] = tuple(value["bias_shape"])
        value["bias_values"] = tuple(value["bias_values"])
        return cls(**value)

    @property
    def sha256(self):
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def task_args(self):
        return (
            *self.conv_workload[1:],
            self.bias_shape if self.bias_shape else (-1,),
            self.bias_dtype or "__none__",
            self.bias_axis if self.bias_axis is not None else -1,
            self.bias_values[0] if self.bias_shape == () else 0,
            self.shift,
            self.clip_min,
            self.clip_max,
            self.output_dtype,
            self.symbol,
            self.occurrence,
        )


def _constant(expr, label):
    if isinstance(expr, relay.Constant):
        return np.asarray(expr.data.numpy())
    if isinstance(expr, tvm.tir.IntImm):
        return np.asarray(int(expr), dtype="int32")
    raise ValueError(f"unsupported VTA fusion: {label} must be constant")


def _scalar(expr, label):
    value = _constant(expr, label)
    if value.size != 1:
        raise ValueError(f"unsupported VTA fusion: {label} must be scalar, got {value.shape}")
    return int(value.reshape(()))


def _conv_workload(conv):
    import vta

    env = vta.get_env()
    data_shape = tuple(int(dim) for dim in conv.args[0].checked_type.shape)
    kernel_shape = tuple(int(dim) for dim in conv.args[1].checked_type.shape)
    if len(data_shape) != 4 or len(kernel_shape) != 4:
        raise ValueError("unsupported VTA Conv: expected rank-4 NHWC/HWIO tensors")
    n, height, width, input_channels = data_shape
    kh, kw, kernel_inputs, output_channels = kernel_shape
    if input_channels != kernel_inputs:
        raise ValueError("VTA Conv input and kernel channels disagree")
    if n % env.BATCH or input_channels % env.BLOCK_IN or output_channels % env.BLOCK_OUT:
        raise ValueError("VTA Conv dimensions do not fit the active VTA packing")
    packed_data = (n // env.BATCH, input_channels // env.BLOCK_IN, height, width,
                   env.BATCH, env.BLOCK_IN)
    packed_kernel = (output_channels // env.BLOCK_OUT, input_channels // env.BLOCK_IN,
                     kh, kw, env.BLOCK_OUT, env.BLOCK_IN)
    return (
        TASK_NAME,
        ("TENSOR", packed_data, str(conv.args[0].checked_type.dtype)),
        ("TENSOR", packed_kernel, str(conv.args[1].checked_type.dtype)),
        tuple(int(x) for x in conv.attrs.strides),
        tuple(int(x) for x in conv.attrs.padding),
        tuple(int(x) for x in conv.attrs.dilation),
        f"NCHW{env.BATCH}n{env.BLOCK_IN}c",
        str(conv.attrs.out_dtype),
    )


def _extract_composite(expr, symbol, occurrence):
    cast = expr
    if not isinstance(cast, relay.Call) or cast.op.name != "cast":
        raise ValueError(f"{symbol}: unsupported VTA fusion; expected final cast")
    clip = cast.args[0]
    if not isinstance(clip, relay.Call) or clip.op.name != "clip":
        raise ValueError(f"{symbol}: unsupported VTA fusion; expected clip before cast")
    shift = clip.args[0]
    if not isinstance(shift, relay.Call) or shift.op.name != "right_shift":
        raise ValueError(f"{symbol}: unsupported VTA fusion; expected right_shift before clip")
    biased = shift.args[0]
    bias_shape, bias_dtype, bias_values, bias_axis = (), None, (), None
    conv = biased
    if isinstance(biased, relay.Call) and biased.op.name in ("add", "nn.bias_add"):
        # In all prepared Tiny graphs the MAC result is the left operand and
        # the other operand is a compile-time scalar or bias vector.
        conv, bias_expr = biased.args
        bias = _constant(bias_expr, "bias")
        bias_shape = tuple(int(dim) for dim in bias.shape)
        bias_dtype = str(bias.dtype)
        bias_values = tuple(int(value) for value in bias.reshape(-1))
        output_shape = tuple(int(dim) for dim in conv.checked_type.shape)
        if bias.size == 1:
            bias_shape = ()
        else:
            if biased.op.name == "nn.bias_add":
                axis = int(biased.attrs.axis)
            else:
                axis = len(output_shape) - bias.ndim
            axis = axis + len(output_shape) if axis < 0 else axis
            if (bias.ndim != 1 or axis < 0 or axis >= len(output_shape)
                    or output_shape[axis] != bias.shape[0]):
                raise ValueError(f"{symbol}: unsupported VTA fusion bias shape/axis {bias.shape}/{axis}")
            bias_axis = axis
    if not isinstance(conv, relay.Call) or conv.op.name != "nn.conv2d":
        raise ValueError(f"{symbol}: unsupported VTA fusion; expected nn.conv2d arithmetic")
    if len(conv.attrs.padding) not in (2, 4):
        raise ValueError(f"{symbol}: unsupported VTA Conv padding rank")
    return FusedOperatorIdentity(
        conv_workload=_conv_workload(conv),
        bias_shape=bias_shape,
        bias_dtype=bias_dtype,
        bias_values=bias_values,
        bias_axis=bias_axis,
        shift=_scalar(shift.args[1], "right_shift"),
        clip_min=int(clip.attrs.a_min),
        clip_max=int(clip.attrs.a_max),
        output_dtype=str(cast.attrs.dtype),
        symbol=symbol,
        occurrence=occurrence,
    )


def extract_fused_identities(prepared):
    """Extract every routed complete Conv fusion in deterministic symbol order."""
    functions = {}
    for gv in prepared.mixed_module.get_global_vars():
        func = prepared.mixed_module[gv]
        attrs = func.attrs
        if attrs is None or attrs.get("Compiler") != "vta":
            continue
        symbol = attrs.get_str("global_symbol") if "global_symbol" in attrs else None
        if not symbol or symbol in functions:
            raise ValueError(f"prepared VTA fusion has a missing or duplicate symbol: {symbol!r}")
        functions[symbol] = func
    symbols = tuple(prepared.routing.symbols)
    if len(symbols) != len(set(symbols)) or set(symbols) != set(functions):
        raise ValueError(
            "prepared VTA function symbols do not match routing: "
            f"routed={symbols}, functions={tuple(functions)}"
        )
    identities = []
    for occurrence, symbol in enumerate(symbols):
        calls = []

        def visit(node):
            if isinstance(node, relay.Call) and isinstance(node.op, relay.Function):
                if node.op.attrs and "Composite" in node.op.attrs:
                    calls.append(node)

        relay.analysis.post_order_visit(functions[symbol].body, visit)
        if len(calls) != 1:
            raise ValueError(f"{symbol}: expected exactly one VTA composite, found {len(calls)}")
        identities.append(_extract_composite(calls[0].op.body, symbol, occurrence))
    if not identities:
        raise ValueError("prepared graph contains no routed VTA Conv fusion")
    return identities


def host_inventory(prepared):
    """Return the prepared pipeline's explicit host-side operator inventory."""
    routing = prepared.routing
    return {
        "host_convolution_count": getattr(routing, "host_convolution_count", None),
        "host_depthwise_count": getattr(routing, "host_depthwise_count", None),
        "host_dense_count": getattr(routing, "host_dense_count", None),
        "host_operator_names": tuple(routing.host_operator_names),
    }


@autotvm.template(TASK_NAME)
def fused_conv2d_packed(data, kernel, strides, padding, dilation, layout, out_dtype,
                        bias_shape, bias_dtype, bias_axis, bias_scalar, shift, clip_min,
                        clip_max, output_dtype, symbol, occurrence):
    """Schedule the prepared Conv plus its exact bias/shift/clip/cast arithmetic."""
    cfg = autotvm.get_config()
    conv = conv2d_packed.__wrapped__(
        cfg, data, kernel, strides, padding, dilation, layout, out_dtype
    )
    value = conv
    if bias_dtype != "__none__":
        if bias_shape and bias_shape != (-1,):
            bias = te.placeholder(bias_shape, dtype=bias_dtype, name="bias")
            axis = int(bias_axis)
            value = te.compute(
                value.shape,
                lambda *i: value[i] + bias[i[axis]],
                name="fused_bias",
                tag=topi.tag.ELEMWISE,
            )
        else:
            value = te.compute(value.shape, lambda *i: value[i] + bias_scalar,
                               name="fused_bias", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: value[i] >> shift,
                       name="fused_right_shift", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: tvm.te.min(value[i], clip_max),
                       name="fused_clip_max", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: tvm.te.max(value[i], clip_min),
                       name="fused_clip_min", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: value[i].astype(output_dtype),
                       name="fused_cast", tag=topi.tag.ELEMWISE)
    schedule = schedule_conv2d_packed.__wrapped__(cfg, [value])
    inputs = [value, data, kernel]
    if bias_dtype != "__none__" and bias_shape and bias_shape != (-1,):
        inputs.append(bias)
    return schedule, inputs


def create_task(identity, target):
    """Create the schedule search task corresponding to one extracted identity."""
    import vta

    task = autotvm.task.create(TASK_NAME, args=identity.task_args(), target=target)
    task.target = vta.get_env().target
    return task


def conv_schedule_key(identity):
    """Return the Conv key emitted when Relay lowers this identity for VTA."""
    return ("conv2d_packed.vta", *identity.conv_workload[1:])


def lower_with_fused_config(prepared, identity, config):
    """Lower the actual prepared fusion while dispatching its Conv key to config."""
    import vta

    function = next(
        (
            prepared.mixed_module[global_var]
            for global_var in prepared.mixed_module.get_global_vars()
            if prepared.mixed_module[global_var].attrs
            and "Compiler" in prepared.mixed_module[global_var].attrs
            and prepared.mixed_module[global_var].attrs["Compiler"] == "vta"
            and prepared.mixed_module[global_var].attrs.get_str("global_symbol") == identity.symbol
        ),
        None,
    )
    if function is None:
        raise ValueError(f"fusion symbol {identity.symbol!r} is absent from the prepared model")
    context = autotvm.task.ApplyConfig(config)
    compiler = tvm.relay.backend.te_compiler.get()
    compiler.clear()
    try:
        with context:
            scheduled = vta.relay.transform._lower_to_scheduled_te(function)
    finally:
        compiler.clear()
    expected_key = conv_schedule_key(identity)
    if context.workload != expected_key:
        raise ValueError(
            "real fusion lowering used a different Conv schedule key: "
            f"expected {expected_key!r}, got {context.workload!r}"
        )
    return scheduled
