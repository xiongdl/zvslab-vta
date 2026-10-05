"""Make clean removes only application-local generated state."""

import shutil
import subprocess
from pathlib import Path


SOURCE_APP = Path(__file__).resolve().parents[1]


def _fake_app(tmp_path):
    app = tmp_path / "repo" / "vta" / "apps" / "mlperf_tiny_benchmark" / "keyword_spotting_v1"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(SOURCE_APP / "Makefile", app / "Makefile")
    shutil.copy2(SOURCE_APP / "scripts" / "make_tasks.sh", app / "scripts" / "make_tasks.sh")
    return app


def _clean(app, output_dir):
    return subprocess.run(
        [
            "make", "clean",
            "CONFIG=/missing/geometry.json",
            "PYTHON=/missing/project/python",
            f"OUTPUT_DIR={output_dir}",
        ],
        cwd=app, capture_output=True, text=True, check=False,
    )


def test_clean_removes_build_and_python_caches_only(tmp_path):
    app = _fake_app(tmp_path)
    external_output = tmp_path / "custom-output"
    external_output.mkdir()
    output_sentinel = external_output / "sentinel.bin"
    output_sentinel.write_bytes(b"custom output")

    (app / "build").mkdir()
    (app / "build" / "sentinel.bin").write_bytes(b"generated")
    cache_roots = (app / "__pycache__", app / "python" / "__pycache__", app / "tests" / "__pycache__")
    for cache in cache_roots:
        cache.mkdir(parents=True)
        (cache / "module.cpython-311.pyc").write_bytes(b"cache")

    persistent = {
        app / "model" / "model.tflite": b"model",
        app / "samples" / "sample.png": b"sample",
        app / "tune" / "config" / "best.log": b"schedule",
        app / "tune" / "config" / "config.json": b"geometry snapshot",
    }
    for path, content in persistent.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    first = _clean(app, external_output)
    assert first.returncode == 0, first.stderr
    assert not (app / "build").exists()
    assert all(not cache.exists() for cache in cache_roots)
    assert output_sentinel.read_bytes() == b"custom output"
    assert {path: path.read_bytes() for path in persistent} == persistent

    second = _clean(app, external_output)
    assert second.returncode == 0, second.stderr
    assert {path: path.read_bytes() for path in persistent} == persistent


def test_clean_unlinks_local_build_symlink_without_following_external_caches(tmp_path):
    app = _fake_app(tmp_path)
    external = tmp_path / "external"
    (external / "build").mkdir(parents=True)
    (external / "build" / "sentinel.bin").write_bytes(b"outside build")
    (external / "__pycache__").mkdir()
    external_cache = external / "__pycache__" / "module.pyc"
    external_cache.write_bytes(b"outside cache")

    (app / "build").symlink_to(external / "build", target_is_directory=True)
    (app / "python").mkdir()
    (app / "python" / "external-cache").symlink_to(external, target_is_directory=True)

    result = _clean(app, tmp_path / "custom-output")
    assert result.returncode == 0, result.stderr
    assert not (app / "build").exists()
    assert (app / "python" / "external-cache").is_symlink()
    assert (external / "build" / "sentinel.bin").read_bytes() == b"outside build"
    assert external_cache.read_bytes() == b"outside cache"
