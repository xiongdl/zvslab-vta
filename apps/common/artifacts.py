"""Inventory known generated MLPerf Tiny application files.

Only files with a known generated name in an owned build location are
classified. Anything unfamiliar remains visible as unknown and is preserved.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_IDS = (
    "image_classification_v1",
    "image_classification_v2",
    "anomaly_detection_v1",
    "keyword_spotting_v1",
    "streaming_wakeword_v1",
    "visual_wake_words_v1",
)
CATEGORIES = ("cache", "tuning-runs")

_RUN_ID = re.compile(r"^\d{8}T\d{6}(?:\.\d+)?Z$")
_OCCURRENCE_LEDGER = re.compile(r"^occurrence-\d+\.ledger\.json$")
_WORKLOAD_RECORD = re.compile(r"^workload-\d+-(?:fsim|tsim)\.(?:json|log)$")
_LEGACY_AUTOTVM = re.compile(
    r"^(?:" + "|".join(re.escape(model) for model in MODEL_IDS) + r")"
    r"-(?:fsim|tsim)-\d{8}T\d{6}(?:\.\d+)?Z\.(?:json|log)$"
)
_CACHE_ROOTS = {
    "reference",
    "mixed",
    "selected_deployment",
    "llvm-fsim",
    "llvm-tsim",
    "c-fsim",
    "c-tsim",
    "fsim",
    "tsim",
}
_CACHE_FILES = {"model.dylib", "graph.json", "manifest.json", "params.bin"}
_SOURCE_SUFFIXES = {".ll", ".c", ".cc", ".cpp", ".h", ".o", ".a", ".dylib", ".so"}


@dataclass(frozen=True)
class Artifact:
    path: Path
    model: str
    category: str
    size_bytes: int


@dataclass(frozen=True)
class Inventory:
    items: tuple[Artifact, ...]
    unknown_paths: tuple[Path, ...]
    errors: tuple[str, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)


def _is_tracked(path: Path, tracked_paths: set[Path]) -> bool:
    return path.resolve(strict=False) in tracked_paths


def _known_cache_file(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    # The VTA model builders store transient compiler products under these
    # named roots. Keep unknown roots and unknown file types untouched.
    root_found = any(part in _CACHE_ROOTS for part in parts[:-1])
    if len(parts) >= 2 and parts[0] == "actual_compute_tuning":
        root_found = root_found or parts[1] in {"activation-fsim", "activation-tsim"}
    if not root_found:
        return False
    name = parts[-1]
    if name in _CACHE_FILES:
        return True
    return "source" in parts and Path(name).suffix in _SOURCE_SUFFIXES


def _known_tuning_file(relative: Path, model: str) -> bool:
    parts = relative.parts
    name = relative.name
    if len(parts) >= 2 and parts[0] == "actual_compute_tuning":
        run_root = parts[1]
        if run_root == "seed" or _RUN_ID.fullmatch(run_root):
            return (
                name in {"manifest.json", "resume-manifest.json", "seed.json", "seed.log", "best.json", "best.log"}
                or bool(_OCCURRENCE_LEDGER.fullmatch(name))
                or bool(_WORKLOAD_RECORD.fullmatch(name))
                or name.startswith("candidate-") and name.endswith((".json", ".log"))
            )
    if len(parts) >= 2 and parts[0] == "two_stage_tuning" and _RUN_ID.fullmatch(parts[1]):
        return (
            name in {"manifest.json", "resume-manifest.json", "best.json", "best.log"}
            or bool(_WORKLOAD_RECORD.fullmatch(name))
            or bool(_OCCURRENCE_LEDGER.fullmatch(name))
        )
    return False


def _walk_owned(root: Path, errors: list[str]) -> Iterable[Path]:
    """Yield files without following symlinks; report all symlink entries."""
    if root.is_symlink():
        errors.append(f"symlinked owned root refused: {root}")
        return
    if not root.exists():
        return
    if not root.is_dir():
        errors.append(f"owned root is not a directory: {root}")
        return
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_dirs = []
        for dirname in dirnames:
            child = current_path / dirname
            if child.is_symlink():
                errors.append(f"symlink refused: {child}")
            else:
                retained_dirs.append(dirname)
        dirnames[:] = retained_dirs
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                errors.append(f"symlink refused: {path}")
            elif path.is_file():
                yield path


def inventory_artifacts(
    benchmark_root: str | Path,
    *,
    model: str = "all",
    categories: Iterable[str] | None = None,
    tracked_paths: Iterable[str | Path] | None = None,
) -> Inventory:
    """List known cache and tuning-run files under the benchmark build dirs.

    ``benchmark_root`` is ``vta/apps/mlperf_tiny_benchmark``. If tracked paths
    are not supplied, callers may pass the VTA repository's ``git ls-files``
    output so committed files are always excluded from cleanup candidates.
    """
    if model != "all" and model not in MODEL_IDS:
        raise ValueError(f"unknown model id: {model}")
    selected_categories = set(CATEGORIES if categories is None else categories)
    invalid_categories = selected_categories - set(CATEGORIES)
    if invalid_categories:
        raise ValueError(f"unknown artifact category: {', '.join(sorted(invalid_categories))}")

    base = Path(benchmark_root).absolute()
    if base.is_symlink():
        raise ValueError(f"symlinked benchmark root refused: {base}")
    tracked = {
        (Path(path) if Path(path).is_absolute() else base / Path(path)).resolve(strict=False)
        for path in (tracked_paths or ())
    }
    items: list[Artifact] = []
    unknown: list[Path] = []
    errors: list[str] = []

    owned_roots: list[tuple[str, Path, str]] = [("shared", base / "build", "shared")]
    models = MODEL_IDS if model == "all" else (model,)
    for model_id in models:
        owned_roots.append((model_id, base / model_id / "build", "model"))

    for owner, build_root, owner_kind in owned_roots:
        for path in _walk_owned(build_root, errors):
            relative = path.relative_to(build_root)
            category = None
            if owner_kind == "shared":
                if relative.parts and relative.parts[0] in {"autotvm", "autotvm-comparison"}:
                    shared_model = next(
                        (model_id for model_id in MODEL_IDS if path.name.startswith(f"{model_id}-")),
                        None,
                    )
                    if not _LEGACY_AUTOTVM.fullmatch(path.name):
                        category = None
                    elif model != "all" and shared_model != model:
                        continue
                    else:
                        category = "tuning-runs"
            else:
                if _known_cache_file(relative):
                    category = "cache"
                elif _known_tuning_file(relative, owner):
                    category = "tuning-runs"
            resolved = path.resolve(strict=False)
            if category is None or category not in selected_categories or _is_tracked(resolved, tracked):
                unknown.append(resolved)
                continue
            try:
                size = path.stat(follow_symlinks=False).st_size
            except OSError as error:
                errors.append(f"cannot stat {path}: {error}")
                continue
            items.append(Artifact(resolved, owner, category, size))

    return Inventory(
        tuple(sorted(items, key=lambda item: (item.model, item.category, str(item.path)))),
        tuple(sorted(set(unknown))),
        tuple(errors),
    )
