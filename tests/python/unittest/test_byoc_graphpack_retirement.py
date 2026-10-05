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

"""Migration contracts for the modern VTA Relay target cutover."""

import ast
import importlib.util
from pathlib import Path

import pytest
import tvm
import vta
from tvm import relay

from byoc_utils import make_qnn_conv2d_module


VTA_ROOT = Path(__file__).resolve().parents[3]
ACTIVE_RELAY_CONSUMERS = (
    ("detection-tutorial", (VTA_ROOT / "tutorials" / "frontend" / "deploy_detection.py",)),
    (
        "mlperf-resnet",
        (
            VTA_ROOT
            / "apps"
            / "mlperf_tiny_benchmark"
            / "image_classification_v1"
            / "python"
            / "model.py",
            VTA_ROOT
            / "apps"
            / "mlperf_tiny_benchmark"
            / "image_classification_v1"
            / "python"
            / "deployment.py",
        ),
    ),
)
ACTIVE_RELAY_DOCUMENTATION = (
    VTA_ROOT / "README.md",
    VTA_ROOT
    / "apps"
    / "mlperf_tiny_benchmark"
    / "image_classification_v1"
    / "README.md",
)
LEGACY_NAMES = {
    "EXTERNAL_COMPILER",
    "register_byoc",
    "graph_pack",
    "get_subgraph",
    "start_name",
    "stop_name",
    "start_name_idx",
    "stop_name_idx",
    "bitpack_start",
    "bitpack_end",
}
LEGACY_TEXT = LEGACY_NAMES | {"relay.ext.vta", "external compiler"}


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _calls(tree, dotted_name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == dotted_name
    ]


def _is_modern_vta_target(call):
    return (
        call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "vta"
        and any(keyword.arg == "host" for keyword in call.keywords)
    )


def _modern_target_bindings(trees):
    names = set()
    providers = set()
    for tree in trees:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _dotted_name(node.value.func) == "tvm.target.Target"
                and _is_modern_vta_target(node.value)
            ):
                names.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(child, ast.Return)
                and isinstance(child.value, ast.Call)
                and _dotted_name(child.value.func) == "tvm.target.Target"
                and _is_modern_vta_target(child.value)
                for child in ast.walk(node)
            ):
                providers.add(node.name)
    return names, providers


def _uses_modern_vta_target(node, names, providers):
    if isinstance(node, ast.Call):
        call_name = _dotted_name(node.func)
        return (
            call_name == "tvm.target.Target" and _is_modern_vta_target(node)
        ) or call_name in providers
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(_uses_modern_vta_target(item, names, providers) for item in node.elts)
    return False


def _source_contract(paths):
    sources = []
    trees = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        sources.append(source)
        trees.append(ast.parse(source, filename=str(path)))
    return "\n".join(sources), trees


@pytest.mark.parametrize(
    ("consumer_name", "paths"),
    ACTIVE_RELAY_CONSUMERS,
    ids=[item[0] for item in ACTIVE_RELAY_CONSUMERS],
)
def test_consumer_uses_explicit_capability_partition_and_modern_target(consumer_name, paths):
    source, trees = _source_contract(paths)
    identifiers = {
        node.id for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    keywords = {
        node.arg for tree in trees for node in ast.walk(tree) if isinstance(node, ast.keyword)
    }
    partition_calls = [
        call for tree in trees for call in _calls(tree, "vta.relay.partition_for_vta")
    ]
    relay_build_calls = [call for tree in trees for call in _calls(tree, "relay.build")]
    target_names, target_providers = _modern_target_bindings(trees)
    build_targets = [
        keyword.value
        for call in relay_build_calls
        for keyword in call.keywords
        if keyword.arg == "target"
    ]

    assert len(partition_calls) == 1, f"{consumer_name} must partition exactly once"
    assert relay_build_calls, f"{consumer_name} must build its partitioned Relay module"
    assert any(
        _uses_modern_vta_target(target, target_names, target_providers)
        for target in build_targets
    ), (
        f'{consumer_name} must pass Target("vta", host=...) to Relay compilation'
    )
    assert not LEGACY_NAMES.intersection(identifiers | attributes | keywords)
    assert not {legacy for legacy in LEGACY_TEXT if legacy in source}


@pytest.mark.parametrize("document", ACTIVE_RELAY_DOCUMENTATION, ids=lambda path: path.stem)
def test_active_relay_documentation_has_no_legacy_cutover_references(document):
    source = document.read_text(encoding="utf-8")

    assert not {legacy for legacy in LEGACY_TEXT if legacy in source}


def test_top_level_documentation_distinguishes_modern_and_low_level_targets():
    source = (VTA_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = source.lower()

    assert "target extension" in normalized
    assert "capability-based partition" in normalized
    assert "vta.relay.partition_for_vta" in source
    assert 'Target("vta"' in source
    assert "low-level" in normalized
    assert "ext_dev" in source


@pytest.mark.parametrize("mod_name", ("resnet18_v1", "yolov3_tiny"))
def test_consumer_fixture_creates_and_builds_a_vta_partition(mod_name):
    env = vta.get_env()
    partitioned = vta.relay.partition_for_vta(
        make_qnn_conv2d_module(env), mod_name=mod_name
    )

    assert any(
        isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        and function.attrs.get_str("Compiler") == "vta"
        for function in partitioned.functions.values()
    )

    with vta.build_config():
        factory = relay.build(
            partitioned,
            target=tvm.target.Target("vta", host=env.target_host),
        )
    assert factory.get_lib() is not None


def test_preserved_low_level_vta_interfaces_and_target_identity_remain_available():
    env = vta.get_env()

    assert all(callable(getattr(vta, name)) for name in ("build_config", "build", "lower"))
    assert env.target.kind.name == "ext_dev"
    assert env.target.device_name == "vta"
    low_level_target = tvm.target.vta(model=env.MODEL)
    assert low_level_target.kind.name == "ext_dev"
    assert low_level_target.device_name == "vta"


def test_legacy_registration_public_api_is_removed():
    assert not hasattr(vta, "register_byoc")


def test_legacy_external_compiler_constant_is_removed():
    assert not hasattr(vta.relay, "EXTERNAL_COMPILER")


def test_legacy_external_compiler_global_is_absent():
    assert tvm.get_global_func("relay.ext.vta", allow_missing=True) is None


def test_legacy_external_compiler_backend_is_removed():
    assert importlib.util.find_spec("vta.relay.backend") is None
    assert not (VTA_ROOT / "python" / "vta" / "relay" / "backend.py").exists()


def test_graphpack_public_api_and_implementation_are_removed():
    assert not hasattr(vta.top, "graph_pack")
    assert importlib.util.find_spec("vta.top.graphpack") is None
    assert not (VTA_ROOT / "python" / "vta" / "top" / "graphpack.py").exists()
