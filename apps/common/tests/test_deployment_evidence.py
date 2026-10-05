"""Tests for the graph-resident deployment evidence retained by all models."""

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "deployment_evidence.py"
SPEC = importlib.util.spec_from_file_location("common_deployment_evidence_contract", MODULE_PATH)
evidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evidence
SPEC.loader.exec_module(evidence)


def test_graph_node_resolution_uses_exact_function_symbols():
    graph = {"nodes": [
        {"op": "null", "name": "input"},
        {"op": "tvm_op", "name": "region_1",
         "attrs": {"func_name": "tvmgen_example_vta_main_1"}},
        {"op": "tvm_op", "name": "region_10",
         "attrs": {"func_name": "tvmgen_example_vta_main_10"}},
    ]}
    expected = [
        {"occurrence": 1, "symbol": "tvmgen_example_vta_main_1"},
        {"occurrence": 10, "symbol": "tvmgen_example_vta_main_10"},
    ]

    assert evidence.resolve_graph_nodes(graph, expected) == {1: 1, 10: 2}


def test_graph_node_resolution_rejects_missing_and_ambiguous_symbols():
    identity = {"occurrence": 0, "symbol": "vta_layer"}
    with pytest.raises(ValueError, match="maps to 0 nodes"):
        evidence.resolve_graph_nodes({"nodes": []}, [identity])

    graph = {"nodes": [
        {"op": "tvm_op", "name": "first", "attrs": {"func_name": "vta_layer"}},
        {"op": "tvm_op", "name": "second", "attrs": {"func_name": "vta_layer"}},
    ]}
    with pytest.raises(ValueError, match="maps to 2 nodes"):
        evidence.resolve_graph_nodes(graph, [identity])


def test_graph_resident_cycle_profile_clears_and_counts_each_node_once():
    graph = {"nodes": [
        {"op": "tvm_op", "name": "region_0", "attrs": {"func_name": "vta_0"}},
        {"op": "tvm_op", "name": "region_1", "attrs": {"func_name": "vta_1"}},
    ]}
    expected = [
        {"occurrence": 0, "symbol": "vta_0"},
        {"occurrence": 1, "symbol": "vta_1"},
    ]

    class Session:
        def __init__(self):
            self.clears = []
            self.reads = 0

        def clear_and_validate(self, simulator):
            self.clears.append(simulator)

        def read_stats(self, *, simulator):
            self.reads += 1
            assert simulator == "tsim"
            return {"cycle_count": 20 + self.reads}

        @staticmethod
        def validate_activity(stats):
            assert stats["cycle_count"] > 0

    class DebugGraph:
        def __init__(self):
            self.executed = []

        def _execute_node(self, node_index):
            self.executed.append(node_index)

    session, graph_executor = Session(), DebugGraph()
    rows = evidence.profile_graph_resident_nodes(
        graph, expected, graph_executor, session, "tsim"
    )

    assert session.clears == ["tsim", "tsim"]
    assert graph_executor.executed == [0, 1]
    assert [row["deployment_cycles"] for row in rows] == [21, 22]
    assert [row["counted_invocations"] for row in rows] == [1, 1]


def test_graph_resident_profile_rejects_nonpositive_or_noninteger_cycles():
    graph = {"nodes": [
        {"op": "tvm_op", "name": "region", "attrs": {"func_name": "vta_0"}}
    ]}
    identity = [{"occurrence": 0, "symbol": "vta_0"}]

    class Session:
        def clear_and_validate(self, _simulator):
            pass

        @staticmethod
        def read_stats(*, simulator):
            return {"cycle_count": True}

        @staticmethod
        def validate_activity(_stats):
            pass

    class DebugGraph:
        @staticmethod
        def _execute_node(_node_index):
            pass

    with pytest.raises(ValueError, match="invalid cycle_count"):
        evidence.profile_graph_resident_nodes(
            graph, identity, DebugGraph(), Session(), "tsim"
        )


def test_unified_deployment_gate_requires_strict_cycle_alignment_and_coverage():
    from common.deployment import validate_occurrence_rows

    expected = [{"occurrence": 0, "symbol": "vta_0"}]
    row = {
        "occurrence": 0,
        "symbol": "vta_0",
        "deployment_cycles": 109,
        "autotvm_cycles": 100,
    }
    assert validate_occurrence_rows([row], expected) == [row]

    with pytest.raises(ValueError, match="strict <10%"):
        validate_occurrence_rows([{**row, "deployment_cycles": 110}], expected)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        validate_occurrence_rows([], expected)
    with pytest.raises(ValueError, match="duplicate"):
        validate_occurrence_rows([row, row], expected)
