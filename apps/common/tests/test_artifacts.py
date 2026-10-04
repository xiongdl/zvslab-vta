from pathlib import Path

from common.artifacts import MODEL_IDS, inventory_artifacts


def _touch(path: Path, contents: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def test_inventory_covers_shared_and_selected_model_builds(tmp_path):
    benchmark = tmp_path / "mlperf_tiny_benchmark"
    shared_log = _touch(
        benchmark / "build" / "autotvm" / "image_classification_v2-fsim-20261002T123456Z.log", b"abc"
    )
    cache_lib = _touch(benchmark / MODEL_IDS[0] / "build" / "reference" / "model.dylib", b"12345")
    run_ledger = _touch(
        benchmark / MODEL_IDS[1] / "build" / "actual_compute_tuning" / "20261002T123456Z" / "occurrence-000.ledger.json",
        b"ledger",
    )
    unknown = _touch(benchmark / MODEL_IDS[0] / "build" / "user-data.txt", b"keep")

    result = inventory_artifacts(benchmark, model="all", tracked_paths=set())

    assert {item.path: (item.category, item.size_bytes) for item in result.items} == {
        shared_log.resolve(): ("tuning-runs", 3),
        cache_lib.resolve(): ("cache", 5),
        run_ledger.resolve(): ("tuning-runs", 6),
    }
    assert unknown.resolve() in result.unknown_paths
    assert result.total_bytes == 14


def test_inventory_scopes_model_and_preserves_unknown_and_tracked_files(tmp_path):
    benchmark = tmp_path / "mlperf_tiny_benchmark"
    selected = _touch(benchmark / MODEL_IDS[0] / "build" / "reference" / "model.dylib")
    other = _touch(benchmark / MODEL_IDS[1] / "build" / "reference" / "model.dylib")
    tracked = _touch(benchmark / MODEL_IDS[0] / "build" / "reference" / "graph.json")
    odd = _touch(benchmark / MODEL_IDS[0] / "build" / "reference" / "notes.md")
    shared_match = _touch(
        benchmark / "build" / "autotvm" / f"{MODEL_IDS[0]}-fsim-20261002T123456Z.log"
    )
    shared_other = _touch(
        benchmark / "build" / "autotvm" / f"{MODEL_IDS[1]}-fsim-20261002T123456Z.log"
    )

    result = inventory_artifacts(
        benchmark,
        model=MODEL_IDS[0],
        tracked_paths={tracked.resolve()},
    )

    assert {item.path for item in result.items} == {selected.resolve(), shared_match.resolve()}
    assert other.resolve() not in {item.path for item in result.items}
    assert shared_other.resolve() not in {item.path for item in result.items}
    assert tracked.resolve() in result.tracked_paths
    assert odd.resolve() in result.unknown_paths


def test_inventory_accepts_exact_model_ids_and_reports_missing_roots(tmp_path):
    assert len(MODEL_IDS) == 6
    result = inventory_artifacts(tmp_path / "absent", model="all", tracked_paths=set())
    assert result.items == ()
    assert result.total_bytes == 0
    assert result.errors == ()


def test_inventory_rejects_unknown_model(tmp_path):
    try:
        inventory_artifacts(tmp_path, model="unknown", tracked_paths=set())
    except ValueError as error:
        assert "model" in str(error)
    else:
        raise AssertionError("unknown model id must be rejected")


def test_inventory_does_not_follow_symlinked_build_root(tmp_path):
    benchmark = tmp_path / "mlperf_tiny_benchmark"
    external = tmp_path / "external"
    payload = _touch(external / "reference" / "model.dylib")
    linked_build = benchmark / MODEL_IDS[0] / "build"
    linked_build.parent.mkdir(parents=True)
    linked_build.symlink_to(external, target_is_directory=True)

    result = inventory_artifacts(benchmark, model=MODEL_IDS[0], tracked_paths=set())

    assert payload.resolve() not in {item.path for item in result.items}
    assert result.errors


def test_resnet_v1_inventory_limits_deployment_files_to_known_output_layouts(tmp_path):
    benchmark = tmp_path / "mlperf_tiny_benchmark"
    build = benchmark / "image_classification_v1" / "build"
    current = _touch(build / "vta_llvm" / "graph.json")
    matrix = _touch(build / "llvm-tsim" / "mixed" / "model.dylib")
    source = _touch(build / "vta_c" / "source" / "host.c")
    archived = _touch(build / "archive" / "llvm" / "graph.json")
    nested = _touch(build / "vta_llvm" / "archive" / "graph.json")
    unknown_source = _touch(build / "llvm-tsim" / "mixed" / "source" / "notes.md")

    result = inventory_artifacts(benchmark, model=MODEL_IDS[0], tracked_paths=set())

    assert {item.path for item in result.items} == {
        current.resolve(), matrix.resolve(), source.resolve()
    }
    assert set(result.unknown_paths) == {
        archived.resolve(), nested.resolve(), unknown_source.resolve()
    }
