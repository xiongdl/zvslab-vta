#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
app_dir="$(cd -- "${script_dir}/.." && pwd -P)"
repo_dir="$(cd -- "${app_dir}/../../../../" && pwd -P)"
invocation_dir="$PWD"
python_bin="${PYTHON:-${repo_dir}/.envs/tvm-vta-env/bin/python}"
config="${CONFIG:-${repo_dir}/vta/config/vta_64mac.json}"
model="${MODEL:-${app_dir}/model/ad01_fp32.tflite}"
input_wav="${INPUT:-${app_dir}/samples/normal_id_01_00000000.wav}"
target="${TARGET:-vta,llvm}"
simulator="${SIMULATOR:-fsim}"
workload="${WORKLOAD:--1}"
trial_batch="${TRIAL_BATCH:-100}"
min_successful="${MIN_SUCCESSFUL:-20}"
fsim_timeout="${FSIM_TIMEOUT:-60}"
tsim_timeout="${TSIM_TIMEOUT:-120}"

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$invocation_dir" "$1" ;;
    esac
}

config="$(absolute_path "$config")"
model="$(absolute_path "$model")"
input_wav="$(absolute_path "$input_wav")"
python_bin="$(absolute_path "$python_bin")"
pythonpath="${repo_dir}/tvm/python:${repo_dir}/vta/python:${app_dir}"

tune_dir() {
    local basename="${config##*/}"
    basename="${basename%.json}"
    printf '%s/tune/%s\n' "$app_dir" "$basename"
}

require_vta_runtime() {
    [[ -f "$config" ]] || { printf 'CONFIG is not a file: %s\n' "$config" >&2; exit 2; }
    [[ -x "$python_bin" ]] || { printf 'Python is not executable: %s\n' "$python_bin" >&2; exit 2; }
    [[ "$simulator" == fsim || "$simulator" == tsim ]] || {
        printf 'SIMULATOR must be fsim or tsim: %s\n' "$simulator" >&2
        exit 2
    }
}

run_python() {
    local backend="$1"
    local script="$2"
    shift 2
    (
        unset VTA_BACKEND
        export VTA_CONFIG_FILE="$config"
        export PYTHONPATH="$pythonpath"
        if [[ -n "$backend" ]]; then
            export VTA_BACKEND="$backend"
        fi
        "$python_bin" "$script" "$@"
    )
}

run_fsim() {
    local workloads="$1"
    local output_logs="$2"
    local timeout="$3"
    run_python fsim "$app_dir/tune.py" \
        --workloads "$workloads" --workload "$workload" --simulator fsim \
        --trial-batch "$trial_batch" --min-successful "$min_successful" \
        --timeout "$timeout" --output-logs "$output_logs"
}

run_tsim() {
    local workloads="$1"
    local input_logs="$2"
    local output_logs="$3"
    local timeout="$4"
    run_python tsim "$app_dir/tune.py" \
        --workloads "$workloads" --workload "$workload" --simulator tsim \
        --input-logs "$input_logs" --timeout "$timeout" \
        --output-logs "$output_logs"
}

case "${1:-}" in
    clean)
        rm -rf -- "$app_dir/build"
        find -P "$app_dir" -type d -name __pycache__ -prune -exec rm -rf -- {} +
        find -P "$app_dir" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
        ;;
    deploy)
        output_dir="${OUTPUT_DIR:-${app_dir}/build}"
        output_dir="$(absolute_path "$output_dir")"
        args=(--model "$model" --input "$input_wav" --target "$target" --output-dir "$output_dir")
        backend=""
        case "$target" in
            c|llvm)
                ;;
            vta,c|vta,llvm)
                require_vta_runtime
                backend="$simulator"
                args+=(--simulator "$simulator")
                ;;
            *)
                printf 'TARGET must be c, llvm, vta,c, or vta,llvm: %s\n' "$target" >&2
                exit 2
                ;;
        esac
        if [[ -n "${SCHEDULE:-}" ]]; then args+=(--schedule "$(absolute_path "$SCHEDULE")"); fi
        if [[ -n "${REPORT:-}" ]]; then args+=(--deployment-report "$(absolute_path "$REPORT")"); fi
        if [[ -n "${EXPORT_WORKLOADS:-}" ]]; then args+=(--export-workloads "$(absolute_path "$EXPORT_WORKLOADS")"); fi
        run_python "$backend" "$app_dir/deploy.py" "${args[@]}"
        ;;
    tune-fsim)
        require_vta_runtime
        [[ -n "${WORKLOADS:-}" ]] || { printf 'tune-fsim requires WORKLOADS=PATH\n' >&2; exit 2; }
        output_logs="${OUTPUT_LOGS:-$(tune_dir)/fsim.tmp}"
        run_fsim "$(absolute_path "$WORKLOADS")" "$(absolute_path "$output_logs")" "${TIMEOUT:-60}"
        ;;
    tune-tsim)
        require_vta_runtime
        [[ -n "${WORKLOADS:-}" ]] || { printf 'tune-tsim requires WORKLOADS=PATH\n' >&2; exit 2; }
        [[ -n "${INPUT_LOGS:-}" ]] || { printf 'tune-tsim requires INPUT_LOGS=PATH\n' >&2; exit 2; }
        output_logs="${OUTPUT_LOGS:-$(tune_dir)/best.log}"
        run_tsim "$(absolute_path "$WORKLOADS")" "$(absolute_path "$INPUT_LOGS")" \
            "$(absolute_path "$output_logs")" "${TIMEOUT:-120}"
        ;;
    tune)
        require_vta_runtime
        output_dir="${OUTPUT_DIR:-${app_dir}/build/tune}"
        output_dir="$(absolute_path "$output_dir")"
        mkdir -p "$output_dir"
        workloads="${WORKLOADS:-${output_dir}/workloads.json}"
        workloads="$(absolute_path "$workloads")"
        output_logs="$(tune_dir)"
        fsim_logs="${output_logs}/fsim.tmp"
        best_logs="${output_logs}/best.log"
        if [[ -z "${WORKLOADS:-}" ]]; then
            mkdir -p "${output_dir}/deploy"
            run_python fsim "$app_dir/deploy.py" \
                --model "$model" --input "$input_wav" \
                --target vta,llvm --simulator fsim \
                --output-dir "${output_dir}/deploy" --export-workloads "$workloads"
        fi
        run_fsim "$workloads" "$fsim_logs" "$fsim_timeout"
        run_tsim "$workloads" "$fsim_logs" "$best_logs" "$tsim_timeout"
        ;;
    *)
        printf 'Usage: make {deploy|tune-fsim|tune-tsim|tune}\n' >&2
        exit 2
        ;;
esac
