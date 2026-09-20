"""Authenticated, atomic Graph Executor artifact export and reload."""

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import tvm
from tvm import relay


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GraphArtifactBundle:
    """Published files and the validated module for one graph bundle."""

    artifact_dir: Path
    graph_path: Path
    params_path: Path
    library_path: Path
    manifest_path: Path
    graph_json: str
    params: bytes
    module: object
    manifest: object
    source_paths: tuple


def shared_library_suffix():
    return ".dylib" if sys.platform == "darwin" else ".so"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    return _sha256(path.read_bytes())


def _safe_artifact_dir(output_root, relative):
    root = Path(output_root).expanduser().resolve(strict=False)
    path = Path(relative)
    if not str(path) or path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError("artifact directory must be a safe non-empty relative path")
    parent = root
    for part in path.parts:
        parent /= part
        if parent.is_symlink():
            raise ValueError("artifact directory resolves through a symlink")
    candidate = root / path
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("artifact path must name a directory")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError("artifact directory escapes output root") from error
    return root, candidate


def _validate_metadata(name, role, model_sha256, host_codegen, simulator):
    if not isinstance(name, str) or not name:
        raise ValueError("artifact name must be non-empty")
    if role not in {"reference", "mixed"}:
        raise ValueError("artifact role must be reference or mixed")
    if not isinstance(model_sha256, str) or not _HASH_RE.fullmatch(model_sha256):
        raise ValueError("model SHA-256 must be 64 lowercase hexadecimal characters")
    if host_codegen not in {"llvm", "c"}:
        raise ValueError("host codegen must be llvm or c")
    if simulator not in {"fsim", "tsim"}:
        raise ValueError("simulator must be fsim or tsim")


def _symbols(values, label):
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} symbols must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} symbols must be unique")
    return result


def _module_type(module):
    value = getattr(module, "type_key", "unknown")
    return str(value() if callable(value) else value)


def _imports(module):
    value = getattr(module, "imported_modules", ())
    return tuple(value() if callable(value) else value)


def _source_format(module_type, host_codegen):
    lowered = module_type.lower()
    if lowered.startswith("llvm") or host_codegen == "llvm":
        return "ll", "ll"
    return "c", "c"


def _discover_sources(module, host_codegen):
    sources = []
    visited = set()

    def visit(current):
        if id(current) in visited:
            return
        visited.add(id(current))
        module_type = _module_type(current)
        requested, suffix = _source_format(module_type, host_codegen)
        try:
            source = current.get_source(requested)
        except Exception:
            source = None
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        if source is not None and not isinstance(source, str):
            raise RuntimeError("module source must be text")
        sources.append(
            {
                "module_type": module_type,
                "source_format": requested or suffix,
                "bytes": source.encode("utf-8") if source is not None else None,
                "available": source is not None,
                "reason": None if source is not None else "source_unavailable",
            }
        )
        for child in _imports(current):
            visit(child)

    visit(module)
    return sources


def _require_source(sources, host_codegen):
    expected = "ll" if host_codegen == "llvm" else "c"
    if not any(item["available"] and item["bytes"] and item["source_format"] == expected for item in sources):
        raise RuntimeError(f"{host_codegen} artifact has no inspectable host source")


def _safe_file(bundle_dir, relative):
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("manifest contains an invalid relative path")
    path = Path(relative)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise RuntimeError(f"manifest contains an unsafe relative path: {relative!r}")
    result = bundle_dir / path
    try:
        result.resolve(strict=False).relative_to(bundle_dir.resolve(strict=False))
    except ValueError as error:
        raise RuntimeError("manifest path escapes artifact directory") from error
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_symbols(module, expected, forbidden):
    checker = getattr(module, "implements_function", None)
    if (expected or forbidden) and not callable(checker):
        raise RuntimeError("reloaded artifact cannot validate its symbol contract")
    missing = [name for name in expected if not checker(name, True)]
    present = [name for name in forbidden if checker(name, True)]
    if missing:
        raise RuntimeError(f"reloaded artifact is missing expected VTA symbols: {missing}")
    if present:
        raise RuntimeError(f"reloaded artifact implements forbidden VTA symbols: {present}")


def _read_files(bundle_dir, manifest):
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("artifact manifest has no files map")
    paths = {}
    for name in ("graph", "params", "library"):
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"artifact manifest has an invalid {name} entry")
        path = _safe_file(bundle_dir, entry.get("path"))
        if not path.is_file() or _file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"artifact {name} hash mismatch or file missing: {path}")
        paths[name] = path
    try:
        graph_json = paths["graph"].read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("graph.json is not UTF-8") from error
    return paths, graph_json, paths["params"].read_bytes()


def _read_sources(bundle_dir, manifest):
    result = []
    for entry in manifest.get("sources", []):
        if not isinstance(entry, dict):
            raise RuntimeError("artifact source entry is invalid")
        if not entry.get("available"):
            continue
        path = _safe_file(bundle_dir, entry.get("path"))
        if not path.is_file() or _file_sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"artifact source hash mismatch or file missing: {path}")
        result.append(path)
    expected = "ll" if manifest.get("host_codegen") == "llvm" else "c"
    if not any(entry.get("available") and entry.get("source_format") == expected for entry in manifest.get("sources", [])):
        raise RuntimeError("artifact has no inspectable host source")
    return tuple(result)


def _bundle(bundle_dir, manifest, module, paths=None):
    if paths is None:
        paths, graph_json, params = _read_files(bundle_dir, manifest)
        sources = _read_sources(bundle_dir, manifest)
    else:
        graph_json, params, sources = paths
        paths = {"graph": bundle_dir / "graph.json", "params": bundle_dir / "params.bin", "library": bundle_dir / ("model" + shared_library_suffix())}
    return GraphArtifactBundle(
        artifact_dir=bundle_dir,
        graph_path=paths["graph"],
        params_path=paths["params"],
        library_path=paths["library"],
        manifest_path=bundle_dir / "manifest.json",
        graph_json=graph_json,
        params=params,
        module=module,
        manifest=_freeze(manifest),
        source_paths=tuple(sources),
    )


def _publish(stage, target):
    backup = None
    if target.exists() or target.is_symlink():
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(stage, target)
    except Exception:
        if backup is not None and not target.exists():
            os.replace(backup, target)
        raise
    return backup


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def export_graph_bundle(factory, output_root, relative_artifact_dir, *, artifact_name,
                        artifact_role, model_sha256, host_codegen, simulator,
                        expected_vta_symbols=(), forbidden_vta_symbols=(), metadata=None):
    """Export a hash-authenticated bundle and atomically reload it."""
    root, target = _safe_artifact_dir(output_root, relative_artifact_dir)
    _validate_metadata(artifact_name, artifact_role, model_sha256, host_codegen, simulator)
    expected = _symbols(expected_vta_symbols, "expected")
    forbidden = _symbols(forbidden_vta_symbols, "forbidden")
    if set(expected) & set(forbidden):
        raise ValueError("expected and forbidden symbol sets must be disjoint")
    root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    backup = None
    published = False
    try:
        graph_json = factory.get_graph_json()
        params = bytes(relay.save_param_dict(factory.get_params()))
        (stage / "graph.json").write_text(graph_json, encoding="utf-8")
        (stage / "params.bin").write_bytes(params)
        library_path = stage / ("model" + shared_library_suffix())
        factory.export_library(str(library_path))
        if not library_path.is_file():
            raise RuntimeError("factory did not export a library")
        sources = _discover_sources(factory.get_lib(), host_codegen)
        _require_source(sources, host_codegen)
        loaded = tvm.runtime.load_module(str(library_path))
        _validate_symbols(loaded, expected, forbidden)
        source_dir = stage / "source"
        source_dir.mkdir()
        source_entries = []
        source_paths = []
        for index, item in enumerate(sources):
            if item["available"]:
                source_path = source_dir / f"{index:02d}-module.{item['source_format']}"
                source_path.write_bytes(item["bytes"])
                source_paths.append(source_path)
                source_entries.append({
                    "available": True, "module_type": item["module_type"],
                    "path": source_path.relative_to(stage).as_posix(),
                    "reason": None, "sha256": _file_sha256(source_path),
                    "source_format": item["source_format"],
                })
            else:
                source_entries.append({
                    "available": False, "module_type": item["module_type"],
                    "path": None, "reason": item["reason"], "sha256": None,
                    "source_format": item["source_format"],
                })
        manifest = {
            "schema_version": 1,
            "artifact": {"name": artifact_name, "role": artifact_role},
            "model_sha256": model_sha256,
            "host_codegen": host_codegen,
            "simulator": simulator,
            "metadata": metadata or {},
            "files": {
                "graph": {"path": "graph.json", "sha256": _file_sha256(stage / "graph.json")},
                "params": {"path": "params.bin", "sha256": _file_sha256(stage / "params.bin")},
                "library": {"path": library_path.name, "sha256": _file_sha256(library_path)},
            },
            "sources": source_entries,
            "symbols": {"expected": list(expected), "forbidden": list(forbidden)},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        backup = _publish(stage, target)
        published = True
        result = load_graph_bundle(root, relative_artifact_dir)
        if backup is not None and backup.exists():
            _remove(backup)
        return result
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if published and (target.exists() or target.is_symlink()):
            _remove(target)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def load_graph_bundle(output_root, relative_artifact_dir):
    """Reload a bundle after validating paths, hashes, source, and symbols."""
    _, bundle_dir = _safe_artifact_dir(output_root, relative_artifact_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported artifact manifest schema")
    paths, graph_json, params = _read_files(bundle_dir, manifest)
    sources = _read_sources(bundle_dir, manifest)
    module = tvm.runtime.load_module(str(paths["library"]))
    symbols = manifest.get("symbols", {})
    _validate_symbols(module, tuple(symbols.get("expected", ())), tuple(symbols.get("forbidden", ())))
    return _bundle(bundle_dir, manifest, module, (graph_json, params, sources))
