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

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import numbers
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_ROOT.parents[1] / "config" / "vta_64mac.json"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build" / "autotvm" / "mac-utilization"
SUPPORTED_TEMPLATES = {"conv2d_packed.vta", "dense_packed.vta"}
CSV_FIELDS = (
    "model_id", "backend", "layer_ordinal", "layer_id", "template",
    "workload_sha256", "flop_count", "mac_count", "best_trial_cycles",
    "peak_macs_per_cycle", "useful_mac_utilization",
    "useful_mac_utilization_percent", "model_sha256", "config_path",
    "config_sha256", "log_path", "log_sha256", "sidecar_path", "sidecar_sha256",
)


def _model_order():
    """Return model identities in the aggregate tuner order."""
    return list(_load_tuner().MODEL_PIPELINES)


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
    tasks, report = _extract_tasks_with_duplicates(
        lambda: tuner.extract_model_tasks(prepared), task_extract_env=TaskExtractEnv
    )
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


def _extract_tasks_with_duplicates(extract, *, task_extract_env=None):
    """Temporarily preserve repeated tasks without leaking global extractor state."""
    if task_extract_env is None:
        from tvm.autotvm.task.topi_integration import TaskExtractEnv as task_extract_env

    prior_env = task_extract_env.current
    original_get = task_extract_env.__dict__["get"]
    extract_env = task_extract_env(allow_duplicate=True)
    task_extract_env.current = extract_env
    # extract_model_tasks calls get() with its deduplicating default. Keep this
    # opt-in scoped to extraction, then restore both the method and singleton.
    task_extract_env.get = staticmethod(lambda allow_duplicate=False: extract_env)
    try:
        return extract()
    finally:
        task_extract_env.get = original_get
        task_extract_env.current = prior_env


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


def build_parser():
    """Create the per-layer report command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Report useful-MAC utilization estimates from validated TSIM AutoTVM "
            "task records. These are isolated-task estimates, not full-model profiling."
        )
    )
    parser.add_argument("--model", required=True, help="one benchmark model id or all")
    parser.add_argument("--backend", required=True, choices=("tsim",))
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"VTA geometry JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--log", type=Path, help="single-model native TSIM AutoTVM log")
    parser.add_argument("--sidecar", type=Path, help="matching single-model JSON sidecar")
    parser.add_argument("--summary", type=Path, help="six-model TSIM aggregate summary")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def _read_json_object(path, description):
    path = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return path, value


def _summary_artifact_path(path, base_dir):
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base_dir / value
    return value.resolve(strict=True)


def load_aggregate_inputs(summary_path, config_path):
    """Validate aggregate model coverage and all declared sidecar identities."""
    summary_path, summary = _read_json_object(summary_path, "TSIM aggregate summary")
    model_order = _model_order()
    if summary.get("backend") != "tsim":
        raise ValueError("aggregate summary backend must be 'tsim'")
    if summary.get("model_order") != model_order:
        raise ValueError("aggregate summary model_order does not match the six benchmark models")
    results = summary.get("results")
    if not isinstance(results, dict):
        raise ValueError("aggregate summary results must be an object")
    missing = [model for model in model_order if model not in results]
    unexpected = sorted(set(results) - set(model_order))
    if missing or unexpected:
        raise ValueError(
            f"aggregate summary model coverage mismatch: missing models={missing}, "
            f"unexpected models={unexpected}"
        )
    if any(not isinstance(results[model], dict) or results[model].get("status") != "succeeded"
           for model in model_order):
        raise ValueError("aggregate summary has failed or incomplete model results")

    config_path = Path(config_path).expanduser().resolve(strict=True)
    config_hash = sha256_file(config_path)
    declared_config = summary.get("config_path")
    if not isinstance(declared_config, str) or Path(declared_config).expanduser().resolve() != config_path:
        raise ValueError("aggregate summary config_path does not match the requested geometry")
    if summary.get("config_sha256") != config_hash:
        raise ValueError("aggregate summary config_sha256 does not match the requested geometry")

    inputs = []
    for model_id in model_order:
        result = results[model_id]
        if not isinstance(result.get("log_path"), str) or not isinstance(
            result.get("sidecar_path"), str
        ):
            raise ValueError(f"aggregate summary has incomplete artifact paths for {model_id}")
        log_path = _summary_artifact_path(result["log_path"], summary_path.parent)
        sidecar_path = _summary_artifact_path(result["sidecar_path"], summary_path.parent)
        _, sidecar = _read_json_object(sidecar_path, f"{model_id} TSIM sidecar")
        if sidecar.get("model_id") != model_id or sidecar.get("backend") != "tsim":
            raise ValueError(f"aggregate summary {model_id} sidecar model/backend identity mismatch")
        if sidecar.get("log_path") != str(log_path):
            raise ValueError(f"aggregate summary {model_id} log/sidecar pairing mismatch")
        if sidecar.get("config_path") != str(config_path) or sidecar.get("config_sha256") != config_hash:
            raise ValueError(f"aggregate summary {model_id} sidecar geometry identity mismatch")
        for field in ("task_count", "trial_count", "task_report"):
            if result.get(field) != sidecar.get(field):
                raise ValueError(f"aggregate summary {model_id} {field} differs from its sidecar")
        inputs.append(
            {
                "model_id": model_id,
                "log_path": log_path,
                "sidecar_path": sidecar_path,
                "sidecar": sidecar,
                "summary_result": result,
            }
        )
    return summary_path, summary, inputs


def _validate_cli_inputs(args):
    model_order = _model_order()
    if args.model != "all" and args.model not in model_order:
        raise ValueError(f"unknown MLPerf Tiny model {args.model!r}")
    single_artifacts = args.log is not None or args.sidecar is not None
    if args.model == "all":
        if args.summary is None or single_artifacts:
            raise ValueError("--model all requires --summary and does not accept --log/--sidecar")
    elif args.summary is not None or args.log is None or args.sidecar is None:
        raise ValueError("single-model mode requires both --log and --sidecar, and no --summary")
    return model_order


def _model_summary(model_id, rows, unsupported, sidecar):
    workloads = {}
    for row in rows:
        workload = workloads.setdefault(
            row["workload_sha256"],
            {
                "template": row["template"],
                "workload_sha256": row["workload_sha256"],
                "flop_count": row["flop_count"],
                "mac_count": row["mac_count"],
                "best_trial_cycles": row["best_trial_cycles"],
                "row_count": 0,
            },
        )
        workload["row_count"] += 1
    first = rows[0]
    unsupported = sorted(
        unsupported,
        key=lambda entry: (entry.get("template", ""), entry.get("workload_sha256", "")),
    )
    return {
        "model_id": model_id,
        "model_sha256": first["model_sha256"],
        "row_count": len(rows),
        "workload_count": len(workloads),
        "workloads": sorted(workloads.values(), key=lambda item: item["workload_sha256"]),
        "artifact_identities": {
            name: {
                "path": first[f"{name}_path"],
                "sha256": first[f"{name}_sha256"],
            }
            for name in ("log", "sidecar")
        },
        "unsupported_task_coverage": {
            "count": len(unsupported),
            "entries": unsupported,
        },
        "task_count": sidecar["task_count"],
        "trial_count": sidecar["trial_count"],
    }


def _stable_run_stem(model_selector, identities):
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"mac-utilization-{model_selector}-tsim-{digest}"


def _write_reports(args, model_rows, model_sidecars, *, aggregate_identity=None):
    """Write paired reports after every selected artifact pair was validated."""
    all_rows = [row for model in model_rows for row in model_rows[model]]
    model_summaries = []
    identities = []
    for model_id, rows in model_rows.items():
        sidecar = model_sidecars[model_id]
        model_summary = _model_summary(
            model_id, rows, sidecar["task_report"]["unsupported"], sidecar
        )
        model_summaries.append(model_summary)
        identities.append(model_summary["artifact_identities"])
    identity = {"models": identities, "aggregate": aggregate_identity}
    stem = _stable_run_stem(args.model, identity)
    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    config_path = args.config.expanduser().resolve(strict=True)
    peak = peak_macs_per_cycle(config_path)
    summary = {
        "schema_version": 1,
        "model_selector": args.model,
        "model_order": list(model_rows),
        "backend": "tsim",
        "metric": {
            "name": "useful_mac_utilization",
            "formula": "logical_MACs / (best_successful_isolated_TSIM_task_cycles * peak_MACs_per_cycle)",
            "units": {
                "logical_mac_count": "MAC",
                "best_trial_cycles": "TSIM cycle",
                "peak_macs_per_cycle": "MAC/cycle",
                "useful_mac_utilization": "ratio",
                "useful_mac_utilization_percent": "percent",
            },
            "semantics": (
                "Isolated AutoTVM task estimate associated with a graph layer occurrence; "
                "not per-layer profiling inside full-model execution."
            ),
        },
        "geometry": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "peak_macs_per_cycle": peak,
        },
        "row_count": len(all_rows),
        "models": model_summaries,
        "unsupported_task_coverage": {
            "count": sum(item["unsupported_task_coverage"]["count"] for item in model_summaries),
            "by_model": {
                item["model_id"]: item["unsupported_task_coverage"] for item in model_summaries
            },
        },
        "aggregate_summary": aggregate_identity,
        "csv_path": str(csv_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    try:
        with csv_tmp.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        json_tmp.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        csv_tmp.replace(csv_path)
        json_tmp.replace(json_path)
    finally:
        csv_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
    return csv_path, json_path


def run_report(args):
    """Validate inputs, calculate all rows, and then write paired reports."""
    if args.backend != "tsim":
        raise ValueError("only TSIM task cycle records are supported")
    _load_tuner().validate_backend("tsim")
    model_order = _validate_cli_inputs(args)
    config_path = args.config.expanduser().resolve(strict=True)
    peak_macs_per_cycle(config_path)
    model_inputs = {}
    aggregate_identity = None
    if args.model == "all":
        summary_path, aggregate, inputs = load_aggregate_inputs(args.summary, config_path)
        model_inputs = {entry["model_id"]: entry for entry in inputs}
        aggregate_identity = {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        }
        selected_models = model_order
    else:
        log_path = Path(args.log).expanduser().resolve(strict=True)
        sidecar_path = Path(args.sidecar).expanduser().resolve(strict=True)
        _, sidecar = _read_json_object(sidecar_path, "TSIM sidecar")
        model_inputs[args.model] = {
            "model_id": args.model,
            "log_path": log_path,
            "sidecar_path": sidecar_path,
            "sidecar": sidecar,
        }
        selected_models = [args.model]

    model_rows = {}
    model_sidecars = {}
    for model_id in selected_models:
        entry = model_inputs[model_id]
        rows = build_rows(
            model_id, entry["log_path"], entry["sidecar_path"], config_path=config_path
        )
        sidecar = entry["sidecar"]
        if args.model == "all":
            result = entry["summary_result"]
            for field in ("task_count", "trial_count", "task_report"):
                if result.get(field) != sidecar.get(field):
                    raise ValueError(f"aggregate summary {model_id} {field} differs from its sidecar")
        model_rows[model_id] = rows
        model_sidecars[model_id] = sidecar

    # No output directory or report is created until every selected model pair
    # has passed graph, config, log, sidecar, TSIM-target, and workload checks.
    return _write_reports(
        args, model_rows, model_sidecars, aggregate_identity=aggregate_identity
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        csv_path, json_path = run_report(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
