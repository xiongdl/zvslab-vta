#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vta_path="$(cd "${script_dir}/../.." && pwd)"
repo_root="$(cd "${vta_path}/.." && pwd)"
backend=""
alu_only=false
encoding_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) backend="${2:?--backend requires fsim or tsim}"; shift 2 ;;
    --alu-only) alu_only=true; shift ;;
    --encoding-only) encoding_only=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "${backend}" == fsim || "${backend}" == tsim ]] || { echo "--backend fsim|tsim is required" >&2; exit 2; }
export TVM_PATH="${TVM_PATH:-${repo_root}/tvm}"
export VTA_PATH="${vta_path}"
export VTA_CONFIG_FILE="${VTA_CONFIG_FILE:-${VTA_PATH}/config/vta_64mac.json}"
export VTA_BACKEND="${backend}"
export PYTHONPATH="${TVM_PATH}/python:${VTA_PATH}/python${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${repo_root}/.envs/tvm-vta-env/bin/python"
[[ -x "${python_bin}" ]] || { echo "Missing environment Python: ${python_bin}" >&2; exit 1; }
if [[ "${encoding_only}" == true ]]; then
  paths=("${script_dir}/test_encoding.py")
elif [[ "${alu_only}" == true ]]; then
  paths=("${script_dir}/test_runtime.py")
else
  paths=("${script_dir}/test_encoding.py" "${script_dir}/test_runtime.py")
fi
"${python_bin}" -m pytest -p no:cacheprovider -v "${paths[@]}"
