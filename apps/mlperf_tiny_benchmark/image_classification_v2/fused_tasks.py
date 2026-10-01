"""AutoTVM tasks for measuring the complete IC V2 VTA Conv fusion."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import tvm
from tvm import autotvm, relay, te, topi

from vta.top.vta_conv2d import conv2d_packed, schedule_conv2d_packed


TASK_NAME = "ic_v2_fused_conv2d.vta"


@dataclass(frozen=True)
class FusedConvIdentity:
    """Serializable Conv and postprocessing identity from an outlined function."""

    conv_workload: tuple
    bias: int | None
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
        return cls(**value)

    @property
    def sha256(self):
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def task_args(self):
        return (*self.conv_workload[1:], self.bias if self.bias is not None else "__none__",
                self.shift, self.clip_min,
                self.clip_max, self.output_dtype, self.symbol, self.occurrence)


def _as_tuple(value):
    return tuple(_as_tuple(item) if isinstance(item, list) else item for item in value)


def _int_constant(expr, label):
    if isinstance(expr, relay.Constant):
        value = np.asarray(expr.data.numpy())
        if value.size != 1:
            raise ValueError(f"IC V2 fused task requires scalar {label}, got shape {value.shape}")
        result = int(value.reshape(()))
    elif isinstance(expr, tvm.tir.IntImm):
        result = int(expr)
    else:
        raise ValueError(f"IC V2 fused task requires a constant {label}")
    return result


def _conv_workload(conv):
    """Translate one real NHWC/HWIO Conv into its packed VTA TOPI arguments."""
    import vta

    env = vta.get_env()
    data_shape = tuple(int(dim) for dim in conv.args[0].checked_type.shape)
    kernel_shape = tuple(int(dim) for dim in conv.args[1].checked_type.shape)
    if len(data_shape) != 4 or len(kernel_shape) != 4:
        raise ValueError("IC V2 fused Conv must use rank-4 NHWC/HWIO tensors")
    n, height, width, input_channels = data_shape
    kh, kw, kernel_inputs, output_channels = kernel_shape
    if input_channels != kernel_inputs:
        raise ValueError("IC V2 fused Conv input and kernel channels disagree")
    if n % env.BATCH or input_channels % env.BLOCK_IN or output_channels % env.BLOCK_OUT:
        raise ValueError("IC V2 fused Conv dimensions do not fit VTA packing")
    packed_data = (n // env.BATCH, input_channels // env.BLOCK_IN, height, width,
                   env.BATCH, env.BLOCK_IN)
    packed_kernel = (output_channels // env.BLOCK_OUT, input_channels // env.BLOCK_IN,
                     kh, kw, env.BLOCK_OUT, env.BLOCK_IN)
    strides = tuple(int(value) for value in conv.attrs.strides)
    padding = tuple(int(value) for value in conv.attrs.padding)
    dilation = tuple(int(value) for value in conv.attrs.dilation)
    data_layout = f"NCHW{env.BATCH}n{env.BLOCK_IN}c"
    out_dtype = str(conv.attrs.out_dtype)
    return (
        TASK_NAME,
        ("TENSOR", packed_data, str(conv.args[0].checked_type.dtype)),
        ("TENSOR", packed_kernel, str(conv.args[1].checked_type.dtype)),
        strides,
        padding,
        dilation,
        data_layout,
        out_dtype,
    )


def extract_fused_identities(prepared):
    """Extract ordered complete VTA Conv identities from a prepared mixed module."""
    from tvm import relay

    functions = {}
    for gv in prepared.mixed_module.get_global_vars():
        func = prepared.mixed_module[gv]
        attrs = func.attrs
        if attrs is None or "Compiler" not in attrs or attrs["Compiler"] != "vta":
            continue
        symbol = attrs.get_str("global_symbol") if "global_symbol" in attrs else None
        if not symbol:
            raise ValueError("outlined VTA fusion is missing its global_symbol")
        if symbol in functions:
            raise ValueError(f"prepared model contains duplicate VTA fusion symbol {symbol!r}")
        functions[symbol] = func

    expected_symbols = tuple(prepared.routing.symbols)
    if len(expected_symbols) != len(set(expected_symbols)):
        raise ValueError("prepared routing report contains duplicate VTA fusion symbols")
    missing = [symbol for symbol in expected_symbols if symbol not in functions]
    unexpected = [symbol for symbol in functions if symbol not in expected_symbols]
    if missing or unexpected:
        raise ValueError(
            "VTA fusion coverage does not match prepared deployment routing: "
            f"missing={missing}, unexpected={unexpected}"
        )

    identities = []
    for symbol in expected_symbols:
        func = functions[symbol]
        composite_calls = []

        def visit(node):
            if isinstance(node, relay.Call) and isinstance(node.op, relay.Function):
                if node.op.attrs and "Composite" in node.op.attrs:
                    composite_calls.append(node)

        relay.analysis.post_order_visit(func.body, visit)
        if len(composite_calls) != 1:
            raise ValueError(f"{symbol} must contain exactly one VTA composite")
        expr = composite_calls[0].op.body
        cast = expr
        if not isinstance(cast, relay.Call) or cast.op.name != "cast":
            raise ValueError(f"{symbol} fusion must end with cast")
        clip = cast.args[0]
        shift = clip.args[0]
        biased = shift.args[0]
        if not isinstance(clip, relay.Call) or clip.op.name != "clip":
            raise ValueError(f"{symbol} fusion must contain clip after right_shift")
        if not isinstance(shift, relay.Call) or shift.op.name != "right_shift":
            raise ValueError(f"{symbol} fusion must contain right_shift before clip")
        bias = None
        if isinstance(biased, relay.Call) and biased.op.name in ("add", "nn.bias_add"):
            conv = biased.args[0]
            bias = _int_constant(biased.args[1], "bias")
        else:
            conv = biased
        if not isinstance(conv, relay.Call) or conv.op.name != "nn.conv2d":
            raise ValueError(f"{symbol} fusion must contain nn.conv2d before postprocessing")
        occurrence = len(identities)
        identity = FusedConvIdentity(
            conv_workload=_conv_workload(conv),
            bias=bias,
            shift=_int_constant(shift.args[1], "right_shift"),
            clip_min=int(clip.attrs.a_min),
            clip_max=int(clip.attrs.a_max),
            output_dtype=str(cast.attrs.dtype),
            symbol=symbol,
            occurrence=occurrence,
        )
        identities.append(identity)
    if not identities:
        raise ValueError("prepared IC V2 graph contains no outlined VTA Conv fusions")
    return identities


@autotvm.template(TASK_NAME)
def fused_conv2d_packed(data, kernel, strides, padding, dilation, layout, out_dtype,
                        bias, shift, clip_min, clip_max, output_dtype, symbol, occurrence):
    """Build the real Conv then bias/shift/clip/cast sequence under VTA scheduling."""
    cfg = autotvm.get_config()
    if bias == "__none__":
        bias = None
    conv = conv2d_packed.__wrapped__(
        cfg, data, kernel, strides, padding, dilation, layout, out_dtype
    )
    value = apply_postprocessing(conv, bias, shift, clip_min, clip_max, output_dtype)
    schedule = schedule_conv2d_packed.__wrapped__(cfg, [value])
    return schedule, [value, data, kernel]


def apply_postprocessing(value, bias, shift, clip_min, clip_max, output_dtype):
    """Apply the deployment fusion's bias, shift, clip and cast in order."""
    if bias is not None:
        value = te.compute(value.shape, lambda *i: value[i] + bias,
                           name="ic_v2_bias", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: value[i] >> shift,
                       name="ic_v2_right_shift", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: tvm.te.min(value[i], clip_max),
                       name="ic_v2_clip_max", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: tvm.te.max(value[i], clip_min),
                       name="ic_v2_clip_min", tag=topi.tag.ELEMWISE)
    value = te.compute(value.shape, lambda *i: value[i].astype(output_dtype),
                       name="ic_v2_cast", tag=topi.tag.ELEMWISE)
    return value


def create_task(identity, target):
    """Create the importable AutoTVM task represented by one identity."""
    import vta

    task = autotvm.task.create(TASK_NAME, args=identity.task_args(), target=target)
    task.target = vta.get_env().target
    return task


def conv_schedule_key(identity):
    """Return the original Conv TOPI key used by real model lowering."""
    return ("conv2d_packed.vta", *identity.conv_workload[1:])


def lower_with_fused_config(prepared, identity, config):
    """Lower the real outlined fusion while dispatching its Conv key to config."""
    import vta

    func = next(
        (
            prepared.mixed_module[gv]
            for gv in prepared.mixed_module.get_global_vars()
            if prepared.mixed_module[gv].attrs
            and "Compiler" in prepared.mixed_module[gv].attrs
            and prepared.mixed_module[gv].attrs["Compiler"] == "vta"
            and prepared.mixed_module[gv].attrs.get_str("global_symbol") == identity.symbol
        ),
        None,
    )
    if func is None:
        raise ValueError(f"fusion symbol {identity.symbol!r} is absent from the prepared model")
    context = autotvm.task.ApplyConfig(config)
    # This helper is called once per model occurrence. Reusing TECompiler's
    # cache across calls can reuse a previous occurrence's shapes/schedule and
    # either select the wrong AutoTVM workload or exceed VTA local-buffer
    # bounds while lowering a later, larger fusion.
    compiler = tvm.relay.backend.te_compiler.get()
    compiler.clear()
    try:
        with context:
            scheduled = vta.relay.transform._lower_to_scheduled_te(func)
    finally:
        compiler.clear()
    if context.workload != conv_schedule_key(identity):
        raise ValueError(
            "real fusion lowering used a different Conv schedule key: "
            f"expected {conv_schedule_key(identity)!r}, got {context.workload!r}"
        )
    return scheduled
