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

"""Validated per-occurrence useful MAC utilization estimates for VTA tasks."""

import hashlib
import importlib.util
import json
import math
import numbers
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_ROOT.parents[1] / "config" / "vta_64mac.json"
SUPPORTED_TEMPLATES = {"conv2d_packed.vta", "dense_packed.vta"}


def _load_tuner():
    """Load shared benchmark tuning identity and task extraction helpers."""
    name = "mlperf_tiny_autotvm_tuner"
    if name in sys.modules:
        return sys.modules[name]
    path = APP_ROOT / "autotvm_tuner.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared AutoTVM tuner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    """Return the SHA-256 identity of one existing artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workload_sha256(workload):
    """Match the canonical workload identity used by the benchmark tuner."""
    payload = json.dumps(workload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def peak_macs_per_cycle(config_path):
    """Derive VTA peak MAC lanes/cycle from log2 geometry fields."""
    path = Path(config_path).expanduser().resolve(strict=True)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"VTA geometry config is not valid JSON: {path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"VTA geometry config must be a JSON object: {path}")
    values = []
    for name in ("LOG_BATCH", "LOG_BLOCK"):
        value = config.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 30:
            raise ValueError(f"VTA geometry config has invalid {name}: {value!r}")
        values.append(value)
    log_batch, log_block = values
    return (1 << log_batch) * (1 << log_block) * (1 << log_block)


def _positive_integral(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integral value, got {value!r}")
    if isinstance(value, numbers.Integral):
        integer = int(value)
    elif isinstance(value, numbers.Real):
        floating = float(value)
        if not math.isfinite(floating) or not floating.is_integer():
            raise ValueError(f"{name} must be a positive integral value, got {value!r}")
        integer = int(floating)
    else:
        raise ValueError(f"{name} must be a positive integral value, got {value!r}")
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integral value, got {value!r}")
    return integer


def calculate_utilization(flop_count, best_trial_cycles, peak):
    """Convert AutoTVM FLOPs to logical MACs and calculate useful utilization."""
    flops = _positive_integral(flop_count, "AutoTVM FLOP count")
    if flops % 2:
        raise ValueError(f"AutoTVM FLOP count must be divisible by two, got {flops}")
    cycles = _positive_integral(best_trial_cycles, "best-trial TSIM cycles")
    peak = _positive_integral(peak, "peak MACs/cycle")
    macs = flops // 2
    ratio = macs / (cycles * peak)
    return {
        "flop_count": flops,
        "mac_count": macs,
        "best_trial_cycles": cycles,
        "peak_macs_per_cycle": peak,
        "useful_mac_utilization": ratio,
        "useful_mac_utilization_percent": 100.0 * ratio,
    }


def minimum_successful_cycles(records, *, expected_templates, expected_workloads):
    """Select the lowest valid successful TSIM trial cost for every workload."""
    best = {}
    observed = set()
    for measure_input, result in records:
        task = measure_input.task
        workload_id = workload_sha256(task.workload)
        observed.add(workload_id)
        if workload_id not in expected_workloads:
            raise ValueError(f"native AutoTVM log contains unexpected workload {workload_id}")
        expected_template = expected_templates.get(workload_id)
        if expected_template is None:
            raise ValueError(f"no extracted template mapping for workload {workload_id}")
        if task.name != expected_template:
            raise ValueError(
                f"native AutoTVM log template mismatch for workload {workload_id}: "
                f"expected {expected_template!r}, found {task.name!r}"
            )
        if result.error_no != 0:
            continue
        costs = getattr(result, "costs", None)
        if not costs:
            raise ValueError(f"successful TSIM trial has no cycle cost for workload {workload_id}")
        for raw_cost in costs:
            cycles = _positive_integral(raw_cost, "successful TSIM trial cycles")
            best[workload_id] = min(best.get(workload_id, cycles), cycles)

    missing_records = set(expected_workloads) - observed
    if missing_records:
        raise ValueError(f"native AutoTVM log is missing workloads: {sorted(missing_records)}")
    missing_success = set(expected_workloads) - set(best)
    if missing_success:
        raise ValueError(
            "native AutoTVM log has no successful cycle cost for workloads: "
            f"{sorted(missing_success)}"
        )
    return best


def rows_for_occurrences(model_id, occurrences, best_cycles, *, peak):
    """Build deterministic independent rows, retaining repeated workloads."""
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    normalized = []
    workload_facts = {}
    for occurrence in occurrences:
        template = occurrence.get("template")
        workload_id = occurrence.get("workload_sha256")
        flops = _positive_integral(occurrence.get("flop_count"), "AutoTVM FLOP count")
        if template not in SUPPORTED_TEMPLATES:
            raise ValueError(f"unsupported AutoTVM template for VTA layer: {template!r}")
        if not isinstance(workload_id, str) or len(workload_id) != 64:
            raise ValueError(f"invalid workload SHA-256 identity: {workload_id!r}")
        try:
            int(workload_id, 16)
        except ValueError as error:
            raise ValueError(f"invalid workload SHA-256 identity: {workload_id!r}") from error
        facts = (template, flops)
        prior = workload_facts.setdefault(workload_id, facts)
        if prior != facts:
            raise ValueError(f"ambiguous workload mapping for {workload_id}")
        normalized.append((template, workload_id, flops))

    if not normalized:
        raise ValueError(f"{model_id} has no supported VTA Conv/Dense layer occurrences")
    rows = []
    for ordinal, (template, workload_id, flops) in enumerate(normalized, start=1):
        if workload_id not in best_cycles:
            raise ValueError(f"missing successful best-trial cycles for workload {workload_id}")
        values = calculate_utilization(flops, best_cycles[workload_id], peak)
        rows.append(
            {
                "model_id": model_id,
                "layer_ordinal": ordinal,
                "layer_id": f"{model_id}:vta-layer-{ordinal:03d}",
                "template": template,
                "workload_sha256": workload_id,
                **values,
            }
        )
    return rows


def _load_model_occurrences(model_id):
    """Prepare the existing model graph and retain each extracted task occurrence."""
    tuner = _load_tuner()
    if model_id not in tuner.MODEL_PIPELINES:
        raise ValueError(f"unknown MLPerf Tiny model {model_id!r}")
    from tvm.autotvm.task.topi_integration import TaskExtractEnv

    pipeline = tuner._load_model_pipeline(model_id)
    _, model_dir, model_filename = tuner.MODEL_PIPELINES[model_id]
    model_path = APP_ROOT / model_id / model_dir / model_filename
    prepared = pipeline.prepare_model(model_path)
    prior_env = TaskExtractEnv.current
    prior_allow_duplicate = (
        prior_env.allow_duplicate if prior_env is not None else False
    )
    extract_env = TaskExtractEnv.get(allow_duplicate=True)
    original_get = TaskExtractEnv.__dict__["get"]
    # The shared tuner's extraction helper calls get() with its default False;
    # preserve this opt-in setting only while that existing extraction path runs.
    TaskExtractEnv.get = staticmethod(lambda allow_duplicate=False: extract_env)
    try:
        tasks, report = tuner.extract_model_tasks(prepared)
    finally:
        TaskExtractEnv.get = original_get
        extract_env.allow_duplicate = prior_allow_duplicate
    occurrences = []
    templates = {}
    for task in tasks:
        workload_id = tuner._task_workload_id(task)
        occurrence = {
            "template": task.name,
            "workload_sha256": workload_id,
            "flop_count": task.flop,
        }
        # Validate the FLOP count before accepting an incomplete or unsupported mapping.
        occurrence["flop_count"] = _positive_integral(
            occurrence["flop_count"], f"AutoTVM FLOP count for workload {workload_id}"
        )
        previous = templates.setdefault(workload_id, task.name)
        if previous != task.name:
            raise ValueError(f"ambiguous extracted template mapping for workload {workload_id}")
        occurrences.append(occurrence)
    return tuner, prepared, occurrences, report


def _validate_extracted_coverage(metadata, occurrences, report):
    sidecar_workloads = set(metadata["task_workloads"])
    extracted_workloads = {row["workload_sha256"] for row in occurrences}
    if extracted_workloads != sidecar_workloads:
        raise ValueError(
            "prepared graph workloads do not match the AutoTVM sidecar: "
            f"missing={sorted(extracted_workloads - sidecar_workloads)}, "
            f"unexpected={sorted(sidecar_workloads - extracted_workloads)}"
        )
    expected_report = {
        (entry["template"], entry["workload_sha256"])
        for entry in report.get("supported", [])
    }
    sidecar_report = {
        (entry["template"], entry["workload_sha256"])
        for entry in metadata["task_report"]["supported"]
    }
    if expected_report != sidecar_report:
        raise ValueError("prepared graph supported task templates do not match the AutoTVM sidecar")


def _load_native_records(log_path):
    tuner = _load_tuner()
    path = Path(log_path).expanduser().resolve(strict=True)
    try:
        records = list(tuner.autotvm.record.load_from_file(str(path)))
    except Exception as error:
        raise ValueError(f"native AutoTVM log cannot be decoded: {path}") from error
    if not records:
        raise ValueError(f"native AutoTVM log has no records: {path}")
    return records


def _validate_tsim_targets(records):
    for measure_input, _ in records:
        target = str(measure_input.target).lower()
        if "device=vta" not in target or "model=tsim_" not in target:
            raise ValueError(
                "native AutoTVM record target is not a VTA TSIM target: "
                f"{measure_input.target}"
            )


def build_rows(model_id, log_path, sidecar_path, *, config_path=None):
    """Validate one model's TSIM artifacts and calculate its VTA layer rows.

    The cycle value is an isolated AutoTVM workload measurement. It is not a
    per-layer profile from a full model execution.
    """
    tuner = _load_tuner()
    tuner.validate_backend("tsim")
    config_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve(strict=True)
    peak = peak_macs_per_cycle(config_path)
    tuner, prepared, occurrences, report = _load_model_occurrences(model_id)
    metadata = tuner.validate_tuning_artifacts(
        log_path,
        sidecar_path,
        model_id=model_id,
        model_sha256=prepared.imported.model_sha256,
        backend="tsim",
        config_path=config_path,
    )
    _validate_extracted_coverage(metadata, occurrences, report)
    expected_workloads = set(metadata["task_workloads"])
    expected_templates = {}
    for occurrence in occurrences:
        workload_id = occurrence["workload_sha256"]
        prior = expected_templates.setdefault(workload_id, occurrence["template"])
        if prior != occurrence["template"]:
            raise ValueError(f"ambiguous extracted template mapping for workload {workload_id}")
    records = _load_native_records(log_path)
    _validate_tsim_targets(records)
    best_cycles = minimum_successful_cycles(
        records,
        expected_templates=expected_templates,
        expected_workloads=expected_workloads,
    )
    rows = rows_for_occurrences(model_id, occurrences, best_cycles, peak=peak)
    log_path = Path(log_path).expanduser().resolve(strict=True)
    sidecar_path = Path(sidecar_path).expanduser().resolve(strict=True)
    config_sha256 = sha256_file(config_path)
    log_sha256 = sha256_file(log_path)
    sidecar_sha256 = sha256_file(sidecar_path)
    for row in rows:
        row.update(
            {
                "backend": "tsim",
                "model_sha256": prepared.imported.model_sha256,
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "log_path": str(log_path),
                "log_sha256": log_sha256,
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": sidecar_sha256,
            }
        )
    return rows
