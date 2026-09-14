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

"""Export and reload self-contained Graph Executor deployment artifacts."""

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
    """Immutable handles and bytes for one published artifact bundle."""

    artifact_dir: Path
    graph_path: Path
    params_path: Path
    library_path: Path
    manifest_path: Path
    graph_json: str
    params: bytes
    module: tvm.runtime.Module
    manifest: object
    source_paths: tuple


def shared_library_suffix():
    """Return the platform suffix used by TVM's shared-library exporter."""
    return ".dylib" if sys.platform == "darwin" else ".so"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    return _sha256(path.read_bytes())


def _resolve_artifact_dir(output_root, relative_artifact_dir):
    """Resolve a safe artifact path without following an escape symlink."""
    root = Path(output_root).expanduser().resolve(strict=False)
    relative = Path(relative_artifact_dir)
    if not str(relative) or relative.is_absolute() or relative == Path("."):
        raise ValueError(
            "artifact directory must be a non-empty relative path below the output root"
        )
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("artifact directory contains an unsafe path component")

    parent = root
    for part in relative.parts:
        parent = parent / part
        if parent.is_symlink():
            raise ValueError(
                "artifact directory resolves through a symlink outside the output root"
            )
    candidate = root.joinpath(relative)
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError("artifact directory resolves outside the output root") from error
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("artifact directory must not be a symlink")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("artifact directory path must name a directory")
    return root, candidate


def _validate_metadata(artifact_name, artifact_role, model_sha256, host_codegen, simulator):
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError("artifact name must be a non-empty string")
    if artifact_role not in {"reference", "mixed"}:
        raise ValueError("artifact role must be reference or mixed")
    if not isinstance(model_sha256, str) or not _HASH_RE.fullmatch(model_sha256):
        raise ValueError("model SHA-256 must be 64 lowercase hexadecimal characters")
    if not isinstance(host_codegen, str) or not host_codegen:
        raise ValueError("host codegen must be a non-empty string")
    if not isinstance(simulator, str) or not simulator:
        raise ValueError("simulator must be a non-empty string")


def _symbols(symbols, label):
    values = tuple(symbols)
    if any(not isinstance(symbol, str) or not symbol for symbol in values):
        raise ValueError(f"{label} symbols must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} symbols must be unique")
    return values


def _module_type(module):
    value = getattr(module, "type_key", "unknown")
    return str(value() if callable(value) else value)


def _module_imports(module):
    imports = getattr(module, "imported_modules", ())
    return tuple(imports() if callable(imports) else imports)


def _source_request(module_type):
    lowered = module_type.lower()
    if lowered.startswith("llvm"):
        return "ll", "ll"
    if lowered in _SOURCE_TYPES:
        source_format = "c" if lowered == "c_source" else lowered
        return source_format, source_format
    return "", "txt"


def _discover_sources(module, host_codegen=None):
    """Walk the import tree once in deterministic pre-order."""
    discovered = []
    visited = set()
    codegen = host_codegen.lower() if isinstance(host_codegen, str) else ""

    def visit(current):
        identity = id(current)
        if identity in visited:
            return
        visited.add(identity)
        module_type = _module_type(current)
        requested_format, suffix = _source_request(module_type)
        if not requested_format and codegen == "llvm":
            requested_format, suffix = "ll", "ll"
        elif not requested_format and codegen == "c":
            requested_format, suffix = "c", "c"
        source = None
        reason = None
        try:
            source = current.get_source(requested_format)
        except Exception:  # Runtime modules legitimately have no source.
            # Keep exception details out of the persisted manifest: runtime and
            # staging paths are neither stable nor useful to artifact readers.
            reason = "get_source_failed"
        if source is None and reason is None:
            reason = "source_unavailable"
        if source is not None:
            if isinstance(source, bytes):
                source = source.decode("utf-8")
            if not isinstance(source, str):
                raise RuntimeError(f"module {module_type} returned a non-text source")
            source_bytes = source.encode("utf-8")
            discovered.append(
                {
                    "module_type": module_type,
                    "source_format": requested_format or suffix,
                    "bytes": source_bytes,
                    "available": True,
                    "reason": None,
                }
            )
        else:
            discovered.append(
                {
                    "module_type": module_type,
                    "source_format": requested_format or suffix,
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
    """Require a non-empty source in the language selected for the host."""
    codegen = host_codegen.lower()
    if codegen == "llvm":
        formats = {"ll"}
    elif codegen == "c":
        formats = {"c", "cc", "cpp"}
    else:
        return
    if not any(
        source.get("available")
        and source.get("bytes")
        and source.get("source_format") in formats
        for source in sources
    ):
        raise RuntimeError(f"{host_codegen} artifact has no inspectable host source")


def _safe_relative_file(bundle_dir, relative):
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"manifest contains an invalid relative file path: {relative!r}")
    path = Path(relative)
    if path == Path(".") or path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise RuntimeError(f"manifest contains an unsafe relative file path: {relative!r}")
    resolved = (bundle_dir / path).resolve(strict=False)
    try:
        resolved.relative_to(bundle_dir.resolve(strict=False))
    except ValueError as error:
        raise RuntimeError(f"manifest file escapes artifact directory: {relative!r}") from error
    return bundle_dir / path


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_symbols(module, expected, forbidden):
    checker = getattr(module, "implements_function", None)
    if (expected or forbidden) and not callable(checker):
        raise RuntimeError("reloaded artifact cannot validate its symbol contract")
    missing = [symbol for symbol in expected if not checker(symbol, True)]
    if missing:
        raise RuntimeError(f"reloaded artifact is missing expected VTA symbols: {missing}")
    present = [symbol for symbol in forbidden if checker(symbol, True)]
    if present:
        raise RuntimeError(f"reloaded artifact implements forbidden VTA symbols: {present}")
    return not missing, not present


def _manifest_for(stage, artifact_name, artifact_role, model_sha256, host_codegen, simulator,
                  expected, forbidden, sources):
    source_entries = []
    source_paths = []
    source_dir = stage / "source"
    source_dir.mkdir()
    for index, source in enumerate(sources):
        if source["available"]:
            suffix = source["source_format"] or "txt"
            source_path = source_dir / (
                f"{index:02d}-{_safe_module_name(source['module_type'])}.{suffix}"
            )
            source_path.write_bytes(source["bytes"])
            relative = source_path.relative_to(stage).as_posix()
            source_paths.append(source_path)
            source_entries.append(
                {
                    "available": True,
                    "module_type": source["module_type"],
                    "path": relative,
                    "reason": None,
                    "sha256": _file_sha256(source_path),
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
    files = {
        "graph": {"path": "graph.json", "sha256": _file_sha256(graph_path)},
        "library": {"path": library_path.name, "sha256": _file_sha256(library_path)},
        "params": {"path": "params.bin", "sha256": _file_sha256(params_path)},
    }
    manifest = {
        "artifact": {"name": artifact_name, "role": artifact_role},
        "files": files,
        "host_codegen": host_codegen,
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
    }
    return manifest, tuple(path for path in source_paths)


def _safe_module_name(module_type):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", module_type).strip(".")
    return value or "module"


def _read_validated_files(bundle_dir, manifest):
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("artifact manifest has no files map")
    loaded = {}
    for key in ("graph", "params", "library"):
        entry = files.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError(f"artifact manifest has an invalid {key} file entry")
        path = _safe_relative_file(bundle_dir, entry["path"])
        if not path.is_file():
            raise RuntimeError(f"artifact file is missing: {path}")
        actual = _file_sha256(path)
        if actual != entry.get("sha256"):
            raise RuntimeError(f"artifact {key} hash mismatch: {path}")
        loaded[key] = path
    graph_bytes = loaded["graph"].read_bytes()
    try:
        graph_json = graph_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("graph.json is not UTF-8") from error
    params = loaded["params"].read_bytes()
    return loaded, graph_json, params


def _read_validated_sources(bundle_dir, manifest):
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("artifact manifest has an invalid sources list")

    source_paths = []
    source_records = []
    for index, entry in enumerate(sources):
        if not isinstance(entry, dict):
            raise RuntimeError(f"artifact manifest has an invalid source entry at index {index}")
        available = entry.get("available")
        if not isinstance(available, bool):
            raise RuntimeError(f"artifact manifest has an invalid source availability at index {index}")
        if not available:
            if entry.get("path") is not None or entry.get("sha256") is not None:
                raise RuntimeError(
                    f"artifact manifest has unexpected source file metadata at index {index}"
                )
            continue

        try:
            path = _safe_relative_file(bundle_dir, entry.get("path"))
        except RuntimeError as error:
            raise RuntimeError(f"invalid source path at index {index}: {error}") from error
        if not path.is_file():
            raise RuntimeError(f"source file is missing: {path}")
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_sha256, str) or not _HASH_RE.fullmatch(expected_sha256):
            raise RuntimeError(f"invalid source SHA-256 at index {index}")
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"source hash mismatch: {path}")
        source_paths.append(path)
        source_records.append(
            {
                "available": True,
                "bytes": path.read_bytes(),
                "source_format": entry.get("source_format"),
            }
        )
    _require_host_source(source_records, manifest.get("host_codegen", ""))
    return tuple(source_paths)


def _result_from_files(bundle_dir, manifest, module):
    files, graph_json, params = _read_validated_files(bundle_dir, manifest)
    source_paths = _read_validated_sources(bundle_dir, manifest)
    return GraphArtifactBundle(
        artifact_dir=bundle_dir,
        graph_path=files["graph"],
        params_path=files["params"],
        library_path=files["library"],
        manifest_path=bundle_dir / "manifest.json",
        graph_json=graph_json,
        params=params,
        module=module,
        manifest=MappingProxyType(manifest),
        source_paths=source_paths,
    )


def _publish(stage, artifact_dir):
    backup = None
    if artifact_dir.exists() or artifact_dir.is_symlink():
        backup = artifact_dir.parent / f".{artifact_dir.name}.backup-{uuid.uuid4().hex}"
        os.replace(artifact_dir, backup)
    try:
        os.replace(stage, artifact_dir)
    except Exception:
        if backup is not None and not artifact_dir.exists():
            os.replace(backup, artifact_dir)
        raise
    return backup


def _remove_path(path):
    """Remove one exact artifact or staging path, including symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def export_graph_bundle(factory, output_root, relative_artifact_dir, *, artifact_name,
                        artifact_role, model_sha256, host_codegen, simulator,
                        expected_vta_symbols=(), forbidden_vta_symbols=()):
    """Export, validate, atomically publish, and reload a Graph bundle."""
    root, artifact_dir = _resolve_artifact_dir(output_root, relative_artifact_dir)
    _validate_metadata(artifact_name, artifact_role, model_sha256, host_codegen, simulator)
    expected = _symbols(expected_vta_symbols, "expected")
    forbidden = _symbols(forbidden_vta_symbols, "forbidden")
    if set(expected) & set(forbidden):
        raise ValueError("expected and forbidden symbol sets must be disjoint")

    root.mkdir(parents=True, exist_ok=True)
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{artifact_dir.name}.staging-", dir=artifact_dir.parent))
    backup = None
    published = False
    try:
        graph_json = factory.get_graph_json()
        if not isinstance(graph_json, str):
            raise RuntimeError("factory graph JSON must be a string")
        params = bytes(relay.save_param_dict(factory.get_params()))
        graph_path = stage / "graph.json"
        params_path = stage / "params.bin"
        graph_path.write_text(graph_json, encoding="utf-8")
        params_path.write_bytes(params)
        library_path = stage / ("model" + shared_library_suffix())
        factory.export_library(str(library_path))
        if not library_path.is_file():
            raise RuntimeError(f"factory did not export a library: {library_path}")

        library = factory.get_lib()
        sources = _discover_sources(library, host_codegen)
        _require_host_source(sources, host_codegen)
        staged_module = tvm.runtime.load_module(str(library_path))
        expected_implemented, forbidden_absent = _validate_symbols(
            staged_module, expected, forbidden
        )
        manifest, _ = _manifest_for(
            stage, artifact_name, artifact_role, model_sha256, host_codegen, simulator,
            expected, forbidden, sources,
        )
        manifest["symbols"]["expected_implemented"] = expected_implemented
        manifest["symbols"]["forbidden_absent"] = forbidden_absent
        _write_manifest(stage / "manifest.json", manifest)

        backup = _publish(stage, artifact_dir)
        published = True
        final_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        final_module = tvm.runtime.load_module(
            str(artifact_dir / ("model" + shared_library_suffix()))
        )
        _validate_symbols(final_module, expected, forbidden)
        result = _result_from_files(artifact_dir, final_manifest, final_module)
        if backup is not None:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        return result
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if published and (artifact_dir.exists() or artifact_dir.is_symlink()):
            _remove_path(artifact_dir)
        if backup is not None and backup.exists() and not artifact_dir.exists():
            os.replace(backup, artifact_dir)
        raise


def load_graph_bundle(output_root, relative_artifact_dir):
    """Reload and hash-validate a previously published Graph bundle."""
    _, artifact_dir = _resolve_artifact_dir(output_root, relative_artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported artifact manifest schema")
    files, _, _ = _read_validated_files(artifact_dir, manifest)
    module = tvm.runtime.load_module(str(files["library"]))
    symbols = manifest.get("symbols", {})
    _validate_symbols(
        module,
        tuple(symbols.get("expected", ())),
        tuple(symbols.get("forbidden", ())),
    )
    return _result_from_files(artifact_dir, manifest, module)
