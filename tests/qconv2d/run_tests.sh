#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vta_path="$(cd "${script_dir}/../.." && pwd)"
repo_root="$(cd "${vta_path}/.." && pwd)"
backend=""
alu_only=false
encoding_only=false
conv_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) backend="${2:?--backend requires fsim or tsim}"; shift 2 ;;
    --alu-only) alu_only=true; shift ;;
    --conv-only) conv_only=true; shift ;;
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
  suite="encoding"
  paths=("${script_dir}/test_encoding.py")
elif [[ "${alu_only}" == true ]]; then
  suite="alu"
  paths=("${script_dir}/test_runtime.py" "${script_dir}/test_alu_requantize.py")
elif [[ "${conv_only}" == true ]]; then
  suite="conv"
  paths=("${script_dir}/test_conv_requantize.py")
else
  suite="all"
  paths=("${script_dir}/test_encoding.py" "${script_dir}/test_runtime.py"
         "${script_dir}/test_alu_requantize.py" "${script_dir}/test_conv_requantize.py")
fi
mkdir -p "${script_dir}/reports"
report_path="${script_dir}/reports/${backend}-${suite}.xml"
export VTA_QCONV_REPORT_DIR="${script_dir}/reports"
echo "Writing JUnit results to ${report_path}"
"${python_bin}" -m pytest -p no:cacheprovider -v --junitxml="${report_path}" "${paths[@]}"
