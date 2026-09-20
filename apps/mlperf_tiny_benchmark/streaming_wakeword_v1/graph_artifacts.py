"""Authenticated, atomic Graph Executor artifacts for streaming wakeword."""

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
_SOURCE_TYPES = {"c", "cc", "cpp", "c_source"}


@dataclass(frozen=True)
class GraphArtifactBundle:
    """Validated paths, bytes, loaded module, and immutable manifest."""

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


def validate_output_root(output_root):
    """Resolve an artifact root and reject the repository environment tree."""
    root = Path(output_root).expanduser().resolve(strict=False)
    environment_directory = "." + "envs"
    if environment_directory in root.parts:
        raise ValueError(
            f"artifact output root must not be inside {environment_directory}"
        )
    return root


def _resolve_artifact_dir(output_root, relative_artifact_dir):
    root = validate_output_root(output_root)
    relative = Path(relative_artifact_dir)
    if not str(relative) or relative.is_absolute() or relative == Path("."):
        raise ValueError("artifact directory must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("artifact directory contains an unsafe path component")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("artifact directory resolves through a symlink")
    candidate = root / relative
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError("artifact directory resolves outside output root") from error
    if candidate.is_symlink():
        raise ValueError("artifact directory must not be a symlink")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("artifact path must name a directory")
    return root, candidate


def _validate_metadata(name, role, model_sha256, host_codegen, simulator):
    if not isinstance(name, str) or not name:
        raise ValueError("artifact name must be a non-empty string")
    if role not in {"reference", "mixed"}:
        raise ValueError("artifact role must be reference or mixed")
    if not isinstance(model_sha256, str) or not _HASH_RE.fullmatch(model_sha256):
        raise ValueError("model SHA-256 must be 64 lowercase hexadecimal characters")
    if not isinstance(host_codegen, str) or not host_codegen:
        raise ValueError("host codegen must be a non-empty string")
    if not isinstance(simulator, str) or not simulator:
        raise ValueError("simulator must be a non-empty string")


def _validate_metadata_payload(metadata, model_sha256):
    if not isinstance(metadata, dict):
        raise RuntimeError("artifact metadata must be an object")
    metadata_model = metadata.get("model_sha256")
    if metadata_model is not None and metadata_model != model_sha256:
        raise RuntimeError("artifact metadata model SHA-256 does not match manifest")
    input_contract = metadata.get("input")
    if input_contract is not None and (
        input_contract.get("shape") != [1, 30, 1, 40]
        or input_contract.get("dtype") != "int8"
    ):
        raise RuntimeError("artifact metadata has the wrong streaming input contract")
    output_contract = metadata.get("output")
    if output_contract is not None and (
        output_contract.get("shape") != [1, 3] or output_contract.get("dtype") != "int8"
    ):
        raise RuntimeError("artifact metadata has the wrong streaming output contract")
    labels = metadata.get("labels")
    if labels is not None and labels != ["Marvin", "Silence", "Unknown"]:
        raise RuntimeError("artifact metadata has the wrong class mapping")


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


def _module_imports(module):
    value = getattr(module, "imported_modules", ())
    return tuple(value() if callable(value) else value)


def _source_request(module_type, host_codegen):
    lowered = module_type.lower()
    if lowered.startswith("llvm"):
        return "ll"
    if lowered in _SOURCE_TYPES:
        return "c" if lowered == "c_source" else lowered
    return "ll" if host_codegen == "llvm" else "c"


def _discover_sources(module, host_codegen):
    discovered = []
    visited = set()

    def visit(current):
        if id(current) in visited:
            return
        visited.add(id(current))
        module_type = _module_type(current)
        source_format = _source_request(module_type, host_codegen)
        source = None
        reason = None
        try:
            source = current.get_source(source_format)
        except Exception:
            reason = "get_source_failed"
        if source is None and reason is None:
            reason = "source_unavailable"
        if source is not None:
            if isinstance(source, bytes):
                source = source.decode("utf-8")
            if not isinstance(source, str):
                raise RuntimeError("module source must be text")
            discovered.append(
                {
                    "module_type": module_type,
                    "source_format": source_format,
                    "bytes": source.encode("utf-8"),
                    "available": True,
                    "reason": None,
                }
            )
        else:
            discovered.append(
                {
                    "module_type": module_type,
                    "source_format": source_format,
                    "bytes": None,
                    "available": False,
                    "reason": reason,
                }
            )
        for child in _module_imports(current):
            visit(child)

    visit(module)
    return discovered


def _require_host_source(sources, host_codegen):
    expected = {"ll"} if host_codegen == "llvm" else {"c", "cc", "cpp"}
    if not any(
        item.get("available") and item.get("bytes") and item.get("source_format") in expected
        for item in sources
    ):
        raise RuntimeError(f"{host_codegen} artifact has no inspectable host source")


def _safe_relative_file(bundle_dir, relative):
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"manifest contains an invalid relative file path: {relative!r}")
    path = Path(relative)
    if path.is_absolute() or path == Path(".") or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise RuntimeError(f"manifest contains an unsafe relative file path: {relative!r}")
    current = bundle_dir
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"manifest file path must not traverse a symlink: {relative!r}")
    resolved = (bundle_dir / path).resolve(strict=False)
    try:
        resolved.relative_to(bundle_dir.resolve(strict=False))
    except ValueError as error:
        raise RuntimeError(f"manifest file escapes artifact directory: {relative!r}") from error
    return bundle_dir / path


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
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
        raise RuntimeError(f"reloaded artifact has forbidden VTA symbols: {present}")


def _read_files(bundle_dir, manifest):
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("artifact manifest has no files map")
    result = {}
    for name in ("graph", "params", "library"):
        entry = files.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError(f"artifact manifest has an invalid {name} file entry")
        path = _safe_relative_file(bundle_dir, entry["path"])
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"artifact file is missing or symlinked: {path}")
        expected = entry.get("sha256")
        if not isinstance(expected, str) or not _HASH_RE.fullmatch(expected):
            raise RuntimeError(f"artifact {name} has an invalid SHA-256")
        if _file_sha256(path) != expected:
            raise RuntimeError(f"artifact {name} hash mismatch: {path}")
        result[name] = path
    try:
        graph_json = result["graph"].read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("graph.json is not UTF-8") from error
    return result, graph_json, result["params"].read_bytes()


def _read_sources(bundle_dir, manifest):
    entries = manifest.get("sources")
    if not isinstance(entries, list):
        raise RuntimeError("artifact manifest has an invalid sources list")
    paths = []
    records = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("available"), bool):
            raise RuntimeError(f"artifact manifest has an invalid source entry at index {index}")
        if not entry["available"]:
            if entry.get("path") is not None or entry.get("sha256") is not None:
                raise RuntimeError(f"artifact manifest has unexpected source metadata at index {index}")
            records.append(entry)
            continue
        path = _safe_relative_file(bundle_dir, entry.get("path"))
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source file is missing or symlinked: {path}")
        expected = entry.get("sha256")
        if not isinstance(expected, str) or not _HASH_RE.fullmatch(expected):
            raise RuntimeError(f"invalid source SHA-256 at index {index}")
        if _file_sha256(path) != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
        paths.append(path)
        records.append(
            {
                "available": True,
                "bytes": path.read_bytes(),
                "source_format": entry.get("source_format"),
            }
        )
    _require_host_source(records, manifest.get("host_codegen", ""))
    return tuple(paths)


def _safe_module_name(module_type):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", module_type).strip(".")
    return value or "module"


def _manifest_for(stage, factory, artifact_name, artifact_role, model_sha256,
                  host_codegen, simulator, expected, forbidden, sources, metadata):
    source_dir = stage / "source"
    source_dir.mkdir()
    source_entries = []
    source_paths = []
    for index, source in enumerate(sources):
        if source["available"]:
            path = source_dir / f"{index:02d}-{_safe_module_name(source['module_type'])}.{source['source_format']}"
            path.write_bytes(source["bytes"])
            source_paths.append(path)
            source_entries.append(
                {
                    "available": True,
                    "module_type": source["module_type"],
                    "path": path.relative_to(stage).as_posix(),
                    "reason": None,
                    "sha256": _file_sha256(path),
                    "source_format": source["source_format"],
                }
            )
        else:
            source_entries.append(
                {
                    "available": False,
                    "module_type": source["module_type"],
                    "path": None,
                    "reason": source["reason"],
                    "sha256": None,
                    "source_format": source["source_format"],
                }
            )
    graph_path = stage / "graph.json"
    params_path = stage / "params.bin"
    library_path = stage / ("model" + shared_library_suffix())
    return {
        "artifact": {"name": artifact_name, "role": artifact_role},
        "files": {
            "graph": {"path": "graph.json", "sha256": _file_sha256(graph_path)},
            "params": {"path": "params.bin", "sha256": _file_sha256(params_path)},
            "library": {"path": library_path.name, "sha256": _file_sha256(library_path)},
        },
        "host_codegen": host_codegen,
        "metadata": metadata or {},
        "model_sha256": model_sha256,
        "schema_version": 1,
        "simulator": simulator,
        "sources": source_entries,
        "symbols": {
            "expected": list(expected),
            "expected_implemented": bool(expected),
            "forbidden": list(forbidden),
            "forbidden_absent": not bool(forbidden),
        },
    }, tuple(source_paths)


def _remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


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


def export_graph_bundle(factory, output_root, relative_artifact_dir, *, artifact_name,
                        artifact_role, model_sha256, host_codegen, simulator,
                        expected_vta_symbols=(), forbidden_vta_symbols=(), metadata=None):
    """Export, atomically publish, reload, and authenticate one bundle."""
    root, target = _resolve_artifact_dir(output_root, relative_artifact_dir)
    _validate_metadata(artifact_name, artifact_role, model_sha256, host_codegen, simulator)
    expected = _symbols(expected_vta_symbols, "expected")
    forbidden = _symbols(forbidden_vta_symbols, "forbidden")
    if set(expected) & set(forbidden):
        raise ValueError("expected and forbidden symbol sets must be disjoint")
    _validate_metadata_payload(metadata or {}, model_sha256)
    root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    backup = None
    published = False
    try:
        graph_json = factory.get_graph_json()
        if not isinstance(graph_json, str):
            raise RuntimeError("factory graph JSON must be a string")
        params = bytes(relay.save_param_dict(factory.get_params()))
        (stage / "graph.json").write_text(graph_json, encoding="utf-8")
        (stage / "params.bin").write_bytes(params)
        library_path = stage / ("model" + shared_library_suffix())
        factory.export_library(str(library_path))
        if not library_path.is_file() or library_path.is_symlink():
            raise RuntimeError(f"factory did not export a regular library: {library_path}")
        sources = _discover_sources(factory.get_lib(), host_codegen)
        _require_host_source(sources, host_codegen)
        staged_module = tvm.runtime.load_module(str(library_path))
        _validate_symbols(staged_module, expected, forbidden)
        manifest, _ = _manifest_for(
            stage, factory, artifact_name, artifact_role, model_sha256,
            host_codegen, simulator, expected, forbidden, sources, metadata,
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        backup = _publish(stage, target)
        published = True
        result = load_graph_bundle(root, relative_artifact_dir)
        if backup is not None and backup.exists():
            _remove_path(backup)
        return result
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if published and (target.exists() or target.is_symlink()):
            _remove_path(target)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def load_graph_bundle(output_root, relative_artifact_dir):
    """Reload every file hash, metadata, source, and symbol contract."""
    _, bundle_dir = _resolve_artifact_dir(output_root, relative_artifact_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"artifact manifest is missing or symlinked: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("artifact manifest is not valid UTF-8 JSON") from error
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported artifact manifest schema")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError("artifact manifest has no artifact identity")
    _validate_metadata(
        artifact.get("name"), artifact.get("role"), manifest.get("model_sha256"),
        manifest.get("host_codegen"), manifest.get("simulator"),
    )
    _validate_metadata_payload(manifest.get("metadata", {}), manifest["model_sha256"])
    files, graph_json, params = _read_files(bundle_dir, manifest)
    source_paths = _read_sources(bundle_dir, manifest)
    symbols = manifest.get("symbols")
    if not isinstance(symbols, dict):
        raise RuntimeError("artifact manifest has no symbol contract")
    expected = _symbols(symbols.get("expected", ()), "expected")
    forbidden = _symbols(symbols.get("forbidden", ()), "forbidden")
    if set(expected) & set(forbidden):
        raise RuntimeError("artifact symbol contract has overlapping symbols")
    module = tvm.runtime.load_module(str(files["library"]))
    _validate_symbols(module, expected, forbidden)
    return GraphArtifactBundle(
        artifact_dir=bundle_dir,
        graph_path=files["graph"],
        params_path=files["params"],
        library_path=files["library"],
        manifest_path=manifest_path,
        graph_json=graph_json,
        params=params,
        module=module,
        manifest=_freeze(manifest),
        source_paths=source_paths,
    )
