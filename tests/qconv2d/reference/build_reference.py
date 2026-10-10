#!/usr/bin/env python3
"""Build the pinned scalar CMSIS-NN requantization and convolution reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys


COMMIT = "13c97dbb6f781d4aab38ed34e6e441f42b79aff4"
TAG = "v8.0.0"
CONVOLUTION_SOURCES = [
    "Source/ConvolutionFunctions/arm_convolve_s8.c",
    "Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
    "Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c",
    "Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
    "Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c",
    "Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c",
]
PINNED_FILES = [
    "Include/arm_nnsupportfunctions.h",
    "Include/arm_nnfunctions.h",
    "Include/arm_nn_math_types.h",
    "Include/arm_nn_types.h",
    "Include/Internal/arm_nn_compiler.h",
    "Include/Internal/arm_nn_config.h",
    *CONVOLUTION_SOURCES,
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(path):
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def read_pin(cmsis_root):
    pin_path = cmsis_root / "UPSTREAM_PIN.md"
    if not pin_path.is_file():
        raise ValueError(f"CMSIS root must contain UPSTREAM_PIN.md: {pin_path}")
    pin = pin_path.read_text(encoding="utf-8")
    tag = re.search(r"^Tag: (.+)$", pin, re.MULTILINE)
    commit = re.search(r"^Commit: ([0-9a-f]{40})$", pin, re.MULTILINE)
    if not tag or not commit or tag.group(1) != TAG or commit.group(1) != COMMIT:
        raise ValueError(f"Expected official CMSIS-NN {TAG} commit {COMMIT}")
    blob_pins = {}
    for digest, relative in re.findall(r"^([0-9a-f]{40}) (\S+)$", pin, re.MULTILINE):
        blob_pins[relative] = digest
    for relative in PINNED_FILES:
        path = cmsis_root / relative
        if not path.is_file():
            raise ValueError(f"Pinned CMSIS-NN source is missing: {relative}")
        expected = blob_pins.get(relative)
        if not expected or git_blob_sha1(path) != expected:
            raise ValueError(f"CMSIS-NN source does not match its upstream blob pin: {relative}")
    return blob_pins


def compile_library(cmsis_root, output_path, wrapper, compiler, macros):
    command = [*compiler, "-std=c11", "-O2", "-fPIC"]
    if sys.platform == "darwin":
        command += ["-dynamiclib"]
    else:
        command += ["-shared"]
    command += [f"-I{cmsis_root / 'Include'}",
                f"-I{cmsis_root / 'Source/ConvolutionFunctions'}",
                f"-I{cmsis_root / 'Source/NNSupportFunctions'}"]
    command += [f"-D{macro}" for macro in macros]
    command += [str(wrapper), *(str(cmsis_root / path) for path in CONVOLUTION_SOURCES),
                "-o", str(output_path)]
    subprocess.run(command, check=True, text=True)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmsis-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    cmsis_root = args.cmsis_root.resolve()
    output_dir = args.output_dir.resolve()
    if not cmsis_root.is_dir():
        parser.error(f"--cmsis-root is not a directory: {cmsis_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    blob_pins = read_pin(cmsis_root)
    wrapper = Path(__file__).with_name("cmsis_reference.c").resolve()
    compiler = shlex.split(os.environ.get("CC", "clang"))
    if not compiler:
        parser.error("CC must name a compiler")
    compiler_path = shutil_which(compiler[0])
    if compiler_path is None:
        parser.error(f"C compiler is not available: {compiler[0]}")
    compiler[0] = compiler_path
    version = subprocess.run(compiler + ["--version"], check=True, capture_output=True, text=True)
    library_extension = ".dylib" if sys.platform == "darwin" else ".so"
    libraries = {}
    commands = {}
    macros = {"double": [], "single": ["CMSIS_NN_USE_SINGLE_ROUNDING"]}
    for mode, build_macros in macros.items():
        name = f"libcmsis_reference_{mode}{library_extension}"
        output_path = output_dir / name
        commands[mode] = compile_library(cmsis_root, output_path, wrapper, compiler, build_macros)
        libraries[mode] = name
    source_hashes = {relative: sha256(cmsis_root / relative) for relative in PINNED_FILES}
    manifest = {
        "version": "8.0.0",
        "commit": COMMIT,
        "tag": TAG,
        "build_macros": macros,
        "libraries": libraries,
        "source_sha256": source_hashes,
        "upstream_blob_sha1": {relative: blob_pins[relative] for relative in PINNED_FILES},
        "wrapper_sha256": sha256(wrapper),
        "compiler": {
            "command": compiler,
            "version": version.stdout.splitlines()[0] if version.stdout else version.stderr.splitlines()[0],
            "platform": platform.platform(),
        },
        "scalar_configuration": {
            "ARM_MATH_DSP": False,
            "ARM_MATH_MVEI": False,
            "CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY": False,
        },
        "source_files": PINNED_FILES,
        "convolution_sources": CONVOLUTION_SOURCES,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)


def shutil_which(command):
    from shutil import which
    return which(command)


if __name__ == "__main__":
    main()
