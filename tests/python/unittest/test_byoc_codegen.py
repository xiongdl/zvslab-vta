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

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import tvm
import vta
from tvm import relay
from tvm.ir import CallingConv

from byoc_utils import make_qnn_conv2d_module
from vta.relay import partition_for_vta
from vta.relay.transform import lower_vta_function


def _partitioned_function(bias_kind=None):
    mod = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env(), bias_kind=bias_kind),
        mod_name="codegen",
    )
    return next(
        function
        for function in mod.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
    )


def _run_isolated_python(source):
    test_dir = str(Path(__file__).resolve().parent)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {test_dir!r})\n{textwrap.dedent(source)}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _tir_to_runtime_hook():
    hook = tvm.target.Target("vta").get_kind_attr("TIRToRuntime")
    assert hook is not None
    return hook


def _relay_to_tir_hook():
    hook = tvm.target.Target("vta").get_kind_attr("RelayToTIR")
    assert hook is not None
    return hook


def _vta_tir_module(function_count=1, host_kind="llvm"):
    primfunc = lower_vta_function(_partitioned_function())
    host = tvm.target.Target(host_kind)
    vta_target = tvm.target.Target("vta", host=host)
    functions = {}
    symbols = []
    for index in range(function_count):
        symbol = "tvmgen_native_vta_{}".format(index)
        functions[symbol] = primfunc.with_attr("global_symbol", symbol).with_attr(
            "target", vta_target
        )
        symbols.append(symbol)
    target = vta_target
    return tvm.IRModule(functions), target, symbols


def _llvm_function_body(llvm_source, symbol):
    definition = re.search(
        r'^define\b[^\n]*@"?{}"?\('.format(re.escape(symbol)),
        llvm_source,
        flags=re.MULTILINE,
    )
    assert definition is not None, symbol
    body_end = llvm_source.find("\n}", definition.end())
    assert body_end != -1, symbol
    return llvm_source[definition.start() : body_end]


@pytest.mark.parametrize("function_count", [1, 3])
def test_tir_to_runtime_returns_one_standard_llvm_module_with_every_symbol(function_count):
    mod, target, symbols = _vta_tir_module(function_count)

    runtime_module = _tir_to_runtime_hook()(mod, target)

    assert isinstance(runtime_module, tvm.runtime.Module)
    assert runtime_module.type_key == "llvm"
    assert runtime_module.handle.value is not None
    assert runtime_module.imported_modules == []
    llvm_source = runtime_module.get_source("ll")
    for symbol in symbols:
        assert runtime_module.implements_function(symbol, False)
        definitions = re.findall(
            r'^define\b[^\n]*@"?{}"?\('.format(re.escape(symbol)),
            llvm_source,
            flags=re.MULTILINE,
        )
        assert len(definitions) == 1, symbol


def test_modern_relay_to_tir_rebinds_every_vta_function_to_active_c_host():
    partitioned = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env()), mod_name="active_c_host"
    )
    active_target = tvm.target.Target("vta", host=tvm.target.Target("c"))

    with active_target:
        lowered = _relay_to_tir_hook()(partitioned)

    vta_functions = [
        function
        for function in lowered.functions.values()
        if isinstance(function, tvm.tir.PrimFunc)
        and function.attrs is not None
        and bool(function.attrs.get("vta.route_to_runtime", False))
    ]
    assert len(vta_functions) == 1
    function = vta_functions[0]
    target = function.attrs["target"]
    assert target.kind.name == "vta"
    assert target.host.kind.name == "c"
    assert target.host.kind.name != "llvm"
    assert function.attrs["global_symbol"]
    assert function.attrs["relay_attrs"].get_str("Compiler") == "vta"


def test_modern_relay_to_tir_rejects_missing_active_host():
    partitioned = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env()), mod_name="missing_active_host"
    )

    with tvm.target.Target("vta"):
        with pytest.raises(tvm.error.TVMError, match="<missing>.*llvm and c"):
            _relay_to_tir_hook()(partitioned)


def test_modern_relay_to_tir_rejects_unsupported_active_host_with_supported_kinds():
    partitioned = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env()), mod_name="unsupported_active_host"
    )
    active_target = tvm.target.Target("vta", host=tvm.target.Target("stackvm"))

    with active_target:
        with pytest.raises(tvm.error.TVMError, match="stackvm.*llvm and c"):
            _relay_to_tir_hook()(partitioned)


def test_tir_to_runtime_returns_one_standard_c_module_with_every_symbol(function_count=2):
    mod, target, symbols = _vta_tir_module(function_count, host_kind="c")

    runtime_module = _tir_to_runtime_hook()(mod, target)

    assert isinstance(runtime_module, tvm.runtime.Module)
    assert runtime_module.type_key == "c"
    source = runtime_module.get_source()
    assert source
    for symbol in symbols:
        assert runtime_module.implements_function(symbol, False)
        definitions = re.findall(
            r"^TVM_DLL .*\b{}\s*\([^)]*\)\s*\{{".format(re.escape(symbol)),
            source,
            flags=re.MULTILINE,
        )
        assert len(definitions) == 1, symbol
    assert "VTACheckConfig" in source
    assert source.find("VTACheckConfig") < source.find("VTATLSCommandHandle")


def test_c_host_partitioned_graph_build_exports_and_reloads_without_simulator():
    result = _run_isolated_python(
        """
        import os
        import sys
        import tempfile

        import tvm
        import vta
        from tvm import relay

        from byoc_utils import make_qnn_conv2d_module
        from vta.relay import partition_for_vta

        assert "vta.testing.simulator" not in sys.modules
        env = vta.get_env()
        partitioned = partition_for_vta(
            make_qnn_conv2d_module(env), mod_name="c_graph_build"
        )
        symbol = next(
            function.attrs.get_str("global_symbol")
            for function in partitioned.functions.values()
            if isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
        )
        target = tvm.target.Target("vta", host=tvm.target.Target("c"))
        with vta.build_config(config={"tir.disable_vectorize": True}):
            factory = relay.build(partitioned, target=target)
        runtime_module = factory.get_lib()
        assert runtime_module.type_key == "c"

        def find_source(module):
            sources = [module.get_source()]
            for imported in module.imported_modules:
                sources.extend(find_source(imported))
            return sources

        sources = find_source(runtime_module)
        assert any(symbol in source for source in sources)
        assert any("VTACheckConfig" in source for source in sources)
        assert any("VTABufferCPUPtr(void*, void*)" in source for source in sources)
        assert all("VTABufferCPUPtr(void*, int8_t*)" not in source for source in sources)
        with tempfile.TemporaryDirectory() as artifact_dir:
            artifact_path = os.path.join(artifact_dir, "c_graph_build.so")
            factory.export_library(artifact_path)
            reloaded = tvm.runtime.load_module(artifact_path)
            assert reloaded.get_function(symbol, True) is not None
        assert "vta.testing.simulator" not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


def test_tir_to_runtime_flattens_external_buffers_before_llvm_codegen():
    mod, target, symbols = _vta_tir_module()
    original_buffers = list(mod[symbols[0]].buffer_map.values())
    assert any(len(buffer.shape) > 1 for buffer in original_buffers)

    runtime_module = _tir_to_runtime_hook()(mod, target)

    assert runtime_module.implements_function(symbols[0], False)


def test_tir_to_runtime_injects_fingerprint_check_before_every_entry_activity():
    mod, target, symbols = _vta_tir_module(3)

    runtime_module = _tir_to_runtime_hook()(mod, target)
    llvm_source = runtime_module.get_source("ll")

    for symbol in symbols:
        body = _llvm_function_body(llvm_source, symbol)
        check_index = body.find("VTACheckConfig")
        assert check_index >= 0, symbol
        activity_indices = [
            body.find(runtime_symbol)
            for runtime_symbol in (
                "VTABufferAlloc",
                "VTATLSCommandHandle",
                "VTABufferCPUPtr",
                "VTALoadBuffer2D",
                "VTAStoreBuffer2D",
                "VTAUopPush",
                "VTAPushGEMMOp",
                "VTAPushALUOp",
                "VTADepPush",
                "VTADepPop",
                "VTASetDebugMode",
                "VTASynchronize",
            )
            if body.find(runtime_symbol) >= 0
        ]
        assert activity_indices, symbol
        assert check_index < min(activity_indices), symbol


def test_tir_to_runtime_does_not_recurse_through_public_tvm_build(monkeypatch):
    mod, target, symbols = _vta_tir_module()

    def unexpected_build(*args, **kwargs):
        raise AssertionError("native TIRToRuntime must not call public tvm.build")

    monkeypatch.setattr(tvm, "build", unexpected_build)

    runtime_module = _tir_to_runtime_hook()(mod, target)

    assert runtime_module.implements_function(symbols[0], False)


def test_tir_to_runtime_validates_every_function_before_codegen():
    mod, target, symbols = _vta_tir_module(2)
    malformed_symbol = symbols[1]
    malformed = mod[malformed_symbol].without_attr("global_symbol")
    mod.update_func(mod.get_global_var(malformed_symbol), malformed)
    before = tvm.ir.save_json(mod)
    llvm_builder = tvm.get_global_func("target.build.llvm")
    codegen_calls = []

    def tracking_llvm_builder(*args):
        codegen_calls.append(args)
        return llvm_builder(*args)

    tvm.register_func("target.build.llvm", tracking_llvm_builder, override=True)

    try:
        with pytest.raises(tvm.error.TVMError, match=malformed_symbol):
            _tir_to_runtime_hook()(mod, target)
    finally:
        tvm.register_func("target.build.llvm", llvm_builder, override=True)

    assert codegen_calls == []
    assert tvm.ir.save_json(mod) == before


def test_tir_to_runtime_rejects_duplicate_symbols():
    mod, target, symbols = _vta_tir_module(2)
    duplicate = mod[symbols[1]].with_attr("global_symbol", symbols[0])
    mod.update_func(mod.get_global_var(symbols[1]), duplicate)

    with pytest.raises(tvm.error.TVMError, match="duplicate.*{}".format(symbols[0])):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_requires_global_var_and_symbol_to_match():
    mod, target, symbols = _vta_tir_module()
    symbol = symbols[0]
    mismatched = mod[symbol].with_attr("global_symbol", "tvmgen_wrong_vta_symbol")
    mod.update_func(mod.get_global_var(symbol), mismatched)

    with pytest.raises(tvm.error.TVMError, match="{}.*global_symbol".format(symbol)):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_rejects_non_primfunc_and_empty_modules():
    _, target, _ = _vta_tir_module()
    relay_mod = tvm.IRModule.from_expr(relay.Function([], relay.const(0)))

    with pytest.raises(tvm.error.TVMError, match="PrimFunc"):
        _tir_to_runtime_hook()(relay_mod, target)
    with pytest.raises(tvm.error.TVMError, match="empty"):
        _tir_to_runtime_hook()(tvm.IRModule(), target)


def test_tir_to_runtime_rejects_wrong_target_and_missing_llvm_host():
    mod, target, symbols = _vta_tir_module()

    with pytest.raises(tvm.error.TVMError, match="vta target"):
        _tir_to_runtime_hook()(mod, tvm.target.Target("llvm"))
    with pytest.raises(tvm.error.TVMError, match="<missing>.*llvm and c"):
        _tir_to_runtime_hook()(mod, tvm.target.Target("vta"))

    symbol = symbols[0]
    wrong_function_target = mod[symbol].with_attr("target", tvm.target.Target("llvm"))
    mod.update_func(mod.get_global_var(symbol), wrong_function_target)
    with pytest.raises(tvm.error.TVMError, match="{}.*target".format(symbol)):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_rejects_unsupported_top_level_host_with_supported_kinds():
    mod, _, _ = _vta_tir_module()
    unsupported_target = tvm.target.Target("vta", host=tvm.target.Target("stackvm"))

    with pytest.raises(tvm.error.TVMError, match="stackvm.*llvm and c"):
        _tir_to_runtime_hook()(mod, unsupported_target)


def test_tir_to_runtime_rejects_unsupported_raw_function_host_with_supported_kinds():
    mod, target, symbols = _vta_tir_module()
    symbol = symbols[0]
    unsupported_function_target = tvm.target.Target(
        "vta", host=tvm.target.Target("stackvm")
    )
    malformed = mod[symbol].with_attr("target", unsupported_function_target)
    mod.update_func(mod.get_global_var(symbol), malformed)

    with pytest.raises(tvm.error.TVMError, match="stackvm.*llvm and c"):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_rejects_unsupported_packed_function_host_with_supported_kinds():
    mod, target, symbols = _vta_tir_module()
    symbol = symbols[0]
    malformed = mod[symbol].with_attrs(
        {
            "target": tvm.target.Target("stackvm"),
            "calling_conv": int(CallingConv.C_PACKED_FUNC),
        }
    )
    mod.update_func(mod.get_global_var(symbol), malformed)

    with pytest.raises(tvm.error.TVMError, match="stackvm.*llvm and c"):
        _tir_to_runtime_hook()(mod, target)


def test_tir_to_runtime_c_converts_integer_and_float_constants():
    target = tvm.target.Target("vta", host=tvm.target.Target("c"))
    functions = {}
    for symbol, dtype, values in (
        ("const_i32", "int32", [0x11223344, -7]),
        ("const_f32", "float32", [1.25, -3.5]),
    ):
        buffer_var = tvm.tir.Var(
            symbol + "_buffer",
            tvm.ir.PointerType(tvm.ir.PrimType(dtype), "global"),
        )
        data = tvm.nd.array(np.asarray(values, dtype=dtype))
        body = tvm.tir.AllocateConst(
            buffer_var,
            dtype,
            [len(values)],
            data,
            tvm.tir.Evaluate(tvm.tir.call_extern("int32", "VTASynchronize")),
        )
        attrs = tvm.ir.make_node("DictAttrs", global_symbol=symbol, target=target)
        functions[symbol] = tvm.tir.PrimFunc([], body, attrs=attrs)

    runtime_module = _tir_to_runtime_hook()(tvm.IRModule(functions), target)
    source = runtime_module.get_source()

    assert "287454020" in source
    assert "-7" in source
    assert "1.25" in source
    assert "-3.5" in source


def test_tir_to_runtime_rejects_malformed_runtime_calls():
    mod, target, symbols = _vta_tir_module()
    symbol = symbols[0]
    malformed = mod[symbol].with_body(
        tvm.tir.Evaluate(tvm.tir.call_extern("int32", "VTAUnknownRuntimeCall"))
    )
    mod.update_func(mod.get_global_var(symbol), malformed)

    with pytest.raises(tvm.error.TVMError, match="{}.*runtime call".format(symbol)):
        _tir_to_runtime_hook()(mod, target)


def test_compile_and_export_do_not_load_or_require_fsim():
    result = _run_isolated_python(
        """
        import os
        import sys
        import tempfile

        import tvm
        import vta
        from tvm import relay

        from byoc_utils import make_qnn_conv2d_module
        from vta.relay import partition_for_vta

        legacy_compiler_global = "relay.ext." + "vta"
        assert tvm.get_global_func("vta.relay._relay_to_tir", True) is not None
        assert "vta.testing.simulator" not in sys.modules
        assert tvm.get_global_func("vta.simulator.profiler_status", True) is None
        assert tvm.get_global_func(legacy_compiler_global, True) is None

        env = vta.get_env()
        partitioned = partition_for_vta(
            make_qnn_conv2d_module(env),
            mod_name="compile_without_fsim",
        )
        symbol = next(
            function.attrs.get_str("global_symbol")
            for function in partitioned.functions.values()
            if isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
        )
        with vta.build_config():
            factory = relay.build(
                partitioned,
                target=tvm.target.Target("vta", host=env.target_host),
            )
        assert factory.get_lib().get_function(symbol, True) is not None

        with tempfile.TemporaryDirectory() as artifact_dir:
            artifact_path = os.path.join(artifact_dir, "compile_without_fsim.tar")
            factory.export_library(artifact_path)
            assert os.path.isfile(artifact_path)
        assert "vta.testing.simulator" not in sys.modules
        assert tvm.get_global_func("vta.simulator.profiler_status", True) is None
        assert tvm.get_global_func(legacy_compiler_global, True) is None
        """
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_lowered_vta_function_builds_with_internal_constants(bias_kind):
    external = _partitioned_function(bias_kind)
    primfunc = lower_vta_function(external)
    symbol = external.attrs.get_str("global_symbol")

    module = tvm.build(
        tvm.IRModule({symbol: primfunc}),
        target=primfunc.attrs["target"],
    )

    assert isinstance(module, tvm.runtime.Module)
    assert module.handle.value is not None
    assert len(primfunc.params) == 2
