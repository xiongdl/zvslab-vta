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
"""VTA config tool"""
import argparse
import hashlib
import json
import os
import sys


ABI_SCHEMA_VERSION = 1
ABI_DEFINITION_KEYS = (
    "LOG_INP_WIDTH",
    "LOG_WGT_WIDTH",
    "LOG_ACC_WIDTH",
    "LOG_OUT_WIDTH",
    "LOG_BATCH",
    "LOG_BLOCK",
    "LOG_BLOCK_IN",
    "LOG_BLOCK_OUT",
    "LOG_UOP_BUFF_SIZE",
    "LOG_INP_BUFF_SIZE",
    "LOG_WGT_BUFF_SIZE",
    "LOG_ACC_BUFF_SIZE",
    "LOG_OUT_BUFF_SIZE",
)

SUPPORTED_BACKENDS = ("fsim", "tsim")
LEGACY_SIMULATOR_TARGETS = ("sim", "tsim")
GEOMETRY_CONFIG_KEYS = (
    "HW_VER",
    "LOG_INP_WIDTH",
    "LOG_WGT_WIDTH",
    "LOG_ACC_WIDTH",
    "LOG_BATCH",
    "LOG_BLOCK",
    "LOG_UOP_BUFF_SIZE",
    "LOG_INP_BUFF_SIZE",
    "LOG_WGT_BUFF_SIZE",
    "LOG_ACC_BUFF_SIZE",
)


class VTAConfigError(ValueError):
    """Raised when the canonical VTA config/backend contract is violated."""


def normalize_backend(backend=None):
    """Normalize the explicit VTA backend selector.

    Parameters
    ----------
    backend : str or None
        ``fsim`` or ``tsim``. If omitted, read ``VTA_BACKEND``.
    """
    selected = os.environ.get("VTA_BACKEND") if backend is None else backend
    if selected in SUPPORTED_BACKENDS:
        return selected
    raise VTAConfigError(
        "Unsupported VTA_BACKEND={!r}; supported values are fsim and tsim. "
        "Set VTA_BACKEND explicitly; legacy simulator names are not accepted."
        .format(selected)
    )


def validate_geometry_config(cfg):
    """Validate a geometry-only config for the canonical backend boundary."""
    target = cfg.get("TARGET")
    if target in LEGACY_SIMULATOR_TARGETS:
        suggested_backend = "fsim" if target == "sim" else "tsim"
        raise VTAConfigError(
            "Legacy TARGET={!r} is not supported for simulator configs; "
            "select the backend explicitly with VTA_BACKEND={} instead."
            .format(target, suggested_backend)
        )

    missing = [key for key in GEOMETRY_CONFIG_KEYS if key not in cfg]
    if missing:
        raise VTAConfigError(
            "Missing VTA geometry config fields: {}".format(", ".join(missing))
        )
    return cfg


def load_geometry_config(path, backend=None):
    """Load and validate a geometry-only config with an explicit backend."""
    normalize_backend(backend)
    with open(path, encoding="utf-8") as config_file:
        cfg = json.load(config_file)
    return validate_geometry_config(cfg)


def pkg_config(cfg):
    """Returns PkgConfig pkg config object."""
    pkg_config_py = os.path.join(
            os.path.dirname(os.path.abspath(os.path.expanduser(__file__))),
            "pkg_config.py"
    )
    libpkg = {"__file__": pkg_config_py}
    exec(compile(open(pkg_config_py, "rb").read(), pkg_config_py, "exec"), libpkg, libpkg)
    PkgConfig = libpkg["PkgConfig"]
    return PkgConfig(cfg)


def abi_definitions(cfg, backend=None):
    """Return the sorted compile definitions that form the VTA ABI."""
    if backend is not None:
        normalize_backend(backend)
    pkg = pkg_config(dict(cfg))
    return tuple(
        sorted("VTA_{}={}".format(key, getattr(pkg, key)) for key in ABI_DEFINITION_KEYS)
    )


def abi_fingerprint(definitions, schema_version=ABI_SCHEMA_VERSION):
    """Return the canonical 64-bit VTA ABI fingerprint."""
    normalized_definitions = sorted(
        definition[2:] if definition.startswith("-D") else definition
        for definition in definitions
    )
    descriptor = {
        "definitions": normalized_definitions,
        "schema_version": schema_version,
    }
    canonical_json = json.dumps(
        descriptor,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(canonical_json).digest()[:8], "big")


def gen_abi_header(definitions):
    """Return a reproducible C/C++ header containing the VTA ABI values."""
    fingerprint = abi_fingerprint(definitions)
    return (
        "#ifndef VTA_ABI_CONFIG_H_\n"
        "#define VTA_ABI_CONFIG_H_\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        "#define VTA_ABI_SCHEMA_VERSION {}\n"
        "#define VTA_ABI_FINGERPRINT UINT64_C(0x{:016x})\n"
        "\n"
        "#endif  // VTA_ABI_CONFIG_H_\n"
    ).format(ABI_SCHEMA_VERSION, fingerprint)


def write_abi_header(path, definitions):
    """Write the canonical ABI header, creating its parent directory."""
    output_path = os.path.abspath(os.path.expanduser(path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="ascii", newline="\n") as output_file:
        output_file.write(gen_abi_header(definitions))

def gen_target_name(pkg):
    """Emit target macro from config"""
    if pkg.TARGET == "pynq":
        return "VTA_TARGET_PYNQ"
    elif pkg.TARGET == "de10nano":
        return "VTA_TARGET_DE10_NANO"
    elif pkg.TARGET == "ultra96":
        return "VTA_TARGET_ULTRA96"
    elif pkg.TARGET == "zcu104":
        return "VTA_TARGET_ZCU104"
    else:
        return None

def gen_target_cflags(pkg):
    """Emit target cflags from config"""
    cflags_str = " ".join(pkg.cflags)
    target = gen_target_name(pkg)
    if target:
        cflags_str += " -D{}".format(target)
    return cflags_str

def calculate_num_wgt_uram(pkg):
    """Calculate number of weight uram from config"""
    if hasattr(pkg, 'num_wgt_mem_uram'):
        return pkg.num_wgt_mem_uram
    else:
        return 0

def gen_tcl_vivado(pkg, file):
    """Export variables to tcl file"""
    const_func = """proc const {name value} {
    uplevel 1 [list set $name $value]
    uplevel 1 [list trace var $name w {error constant ;#} ]
}"""
    with open(file, "w") as fo:
        fo.write(const_func)
        fo.write("\nconst CFLAGS \"{}\"".format(gen_target_cflags(pkg)))
        fo.write("\nconst TARGET {}".format(pkg.TARGET))
        fo.write("\nconst FPGA_DEVICE {}".format(pkg.fpga_device))
        fo.write("\nconst FPGA_FAMILY {}".format(pkg.fpga_family))
        fo.write("\nconst FPGA_BOARD {}".format(pkg.fpga_board))
        fo.write("\nconst FPGA_BOARD_REV {}".format(pkg.fpga_board_rev))
        fo.write("\nconst FPGA_PERIOD {}".format(pkg.fpga_per))
        fo.write("\nconst FPGA_FREQ {}".format(pkg.fpga_freq))
        fo.write("\nconst INP_MEM_AXI_RATIO {}".format(pkg.inp_mem_axi_ratio))
        fo.write("\nconst WGT_MEM_AXI_RATIO {}".format(pkg.wgt_mem_axi_ratio))
        fo.write("\nconst OUT_MEM_AXI_RATIO {}".format(pkg.out_mem_axi_ratio))
        fo.write("\nconst INP_MEM_BANKS {}".format(pkg.inp_mem_banks))
        fo.write("\nconst WGT_MEM_BANKS {}".format(pkg.wgt_mem_banks))
        fo.write("\nconst OUT_MEM_BANKS {}".format(pkg.out_mem_banks))
        fo.write("\nconst INP_MEM_WIDTH {}".format(pkg.inp_mem_width))
        fo.write("\nconst WGT_MEM_WIDTH {}".format(pkg.wgt_mem_width))
        fo.write("\nconst OUT_MEM_WIDTH {}".format(pkg.out_mem_width))
        fo.write("\nconst INP_MEM_DEPTH {}".format(pkg.inp_mem_depth))
        fo.write("\nconst WGT_MEM_DEPTH {}".format(pkg.wgt_mem_depth))
        fo.write("\nconst OUT_MEM_DEPTH {}".format(pkg.out_mem_depth))
        fo.write("\nconst NUM_WGT_MEM_URAM {}".format(calculate_num_wgt_uram(pkg)))
        fo.write("\nconst AXI_CACHE_BITS {}".format(pkg.axi_cache_bits))
        fo.write("\nconst AXI_PROT_BITS {}".format(pkg.axi_prot_bits))
        fo.write("\nconst IP_REG_MAP_RANGE {}".format(pkg.ip_reg_map_range))
        fo.write("\nconst FETCH_BASE_ADDR {}".format(pkg.fetch_base_addr))
        fo.write("\nconst LOAD_BASE_ADDR {}".format(pkg.load_base_addr))
        fo.write("\nconst COMPUTE_BASE_ADDR {}".format(pkg.compute_base_addr))
        fo.write("\nconst STORE_BASE_ADDR {}".format(pkg.store_base_addr))

def main():
    """Main funciton"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cfg", type=str, default="",
                        help="path to the config json")
    parser.add_argument("--cflags", action="store_true",
                        help="print the cflags")
    parser.add_argument("--defs", action="store_true",
                        help="print the macro defs")
    parser.add_argument("--sources", action="store_true",
                        help="print the source file paths")
    parser.add_argument("--update", action="store_true",
                        help="Print out the json option.")
    parser.add_argument("--ldflags", action="store_true",
                        help="print the ldflags")
    parser.add_argument("--cfg-json", action="store_true",
                        help="print all the config json")
    parser.add_argument("--save-cfg-json", type=str, default="",
                        help="save config json to file")
    parser.add_argument("--target", action="store_true",
                        help="print the target")
    parser.add_argument("--cfg-str", action="store_true",
                        help="print the configuration string")
    parser.add_argument("--abi-fingerprint", action="store_true",
                        help="print the canonical VTA ABI fingerprint")
    parser.add_argument("--backend-contract", action="store_true",
                        help="reject legacy simulator TARGET fields")
    parser.add_argument("--abi-header", type=str, default="",
                        help="write the canonical VTA ABI header")
    parser.add_argument("--get-inp-mem-banks", action="store_true",
                        help="returns number of input memory banks")
    parser.add_argument("--get-inp-mem-width", action="store_true",
                        help="returns input memory read/write port width")
    parser.add_argument("--get-inp-mem-depth", action="store_true",
                        help="returns input memory depth")
    parser.add_argument("--get-inp-mem-axi-ratio", action="store_true",
                        help="returns ratio between input element width and axi width")
    parser.add_argument("--get-wgt-mem-banks", action="store_true",
                        help="returns number of weight memory banks")
    parser.add_argument("--get-wgt-mem-width", action="store_true",
                        help="returns weight memory read/write port width")
    parser.add_argument("--get-wgt-mem-depth", action="store_true",
                        help="returns weight memory depth")
    parser.add_argument("--get-wgt-mem-axi-ratio", action="store_true",
                        help="returns ratio between weight element width and axi width")
    parser.add_argument("--get-out-mem-banks", action="store_true",
                        help="returns number of output memory banks")
    parser.add_argument("--get-out-mem-width", action="store_true",
                        help="returns output memory read/write port width")
    parser.add_argument("--get-out-mem-depth", action="store_true",
                        help="returns output memory depth")
    parser.add_argument("--get-out-mem-axi-ratio", action="store_true",
                        help="returns ratio between output element width and axi width")
    parser.add_argument("--get-num-wgt-mem-uram", action="store_true",
                        help="returns number of weight memory blocks to be implemented on URAM")
    parser.add_argument("--get-axi-cache-bits", action="store_true",
                        help="returns AXI system ARCACHE/AWCACHE hardcoded bit value")
    parser.add_argument("--get-axi-prot-bits", action="store_true",
                        help="returns AXI system ARPROT/AWPROT hardcoded bit value")
    parser.add_argument("--get-ip-reg-map-range", action="store_true",
                        help="returns ip register map address range")
    parser.add_argument("--get-fetch-base-addr", action="store_true",
                        help="returns fetch module base address")
    parser.add_argument("--get-load-base-addr", action="store_true",
                        help="returns load module base address")
    parser.add_argument("--get-compute-base-addr", action="store_true",
                        help="returns compute module base address")
    parser.add_argument("--get-store-base-addr", action="store_true",
                        help="returns store module base address")
    parser.add_argument("--get-fpga-dev", action="store_true",
                        help="returns FPGA device target")
    parser.add_argument("--get-fpga-board", action="store_true",
                        help="returns FPGA board")
    parser.add_argument("--get-fpga-board-rev", action="store_true",
                        help="returns FPGA board version")
    parser.add_argument("--get-fpga-family", action="store_true",
                        help="returns FPGA device family")
    parser.add_argument("--get-fpga-freq", action="store_true",
                        help="returns FPGA frequency")
    parser.add_argument("--get-fpga-per", action="store_true",
                        help="returns HLS target clock period")
    parser.add_argument("--export-tcl", type=str, default="",
                        help="export variables to tcl file")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Path to vta config
    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))

    path_list = [
        "vta_config.json", os.path.join(curr_path, "vta_config.json")
    ]

    if args.use_cfg:
        path_list = [args.use_cfg]

    ok_path_list = [p for p in path_list if os.path.exists(p)]
    if not ok_path_list:
        raise RuntimeError("Cannot find config in %s" % str(path_list))

    cfg = json.load(open(ok_path_list[0]))
    if args.backend_contract:
        validate_geometry_config(cfg)
    pkg = pkg_config(cfg)

    if args.target:
        print(pkg.TARGET)

    if args.defs:
        print(" ".join(pkg.macro_defs))

    if args.sources:
        print(" ".join(pkg.lib_source))

    if args.cflags:
        print(gen_target_cflags(pkg))

    if args.ldflags:
        print(" ".join(pkg.ldflags))

    if args.cfg_json:
        print(pkg.cfg_json)

    if args.save_cfg_json:
        with open(args.save_cfg_json, "w") as fo:
            fo.write(pkg.cfg_json)

    if args.cfg_str:
        print(pkg.TARGET + "_" + pkg.bitstream)

    if args.abi_fingerprint or args.abi_header:
        definitions = abi_definitions(cfg)

        if args.abi_fingerprint:
            print("{:016x}".format(abi_fingerprint(definitions)))

        if args.abi_header:
            write_abi_header(args.abi_header, definitions)

    if args.get_inp_mem_banks:
        print(pkg.inp_mem_banks)

    if args.get_inp_mem_width:
        print(pkg.inp_mem_width)

    if args.get_inp_mem_depth:
        print(pkg.inp_mem_depth)

    if args.get_inp_mem_axi_ratio:
        print(pkg.inp_mem_axi_ratio)

    if args.get_wgt_mem_banks:
        print(pkg.wgt_mem_banks)

    if args.get_wgt_mem_width:
        print(pkg.wgt_mem_width)

    if args.get_wgt_mem_depth:
        print(pkg.wgt_mem_depth)

    if args.get_wgt_mem_axi_ratio:
        print(pkg.wgt_mem_axi_ratio)

    if args.get_out_mem_banks:
        print(pkg.out_mem_banks)

    if args.get_out_mem_width:
        print(pkg.out_mem_width)

    if args.get_out_mem_depth:
        print(pkg.out_mem_depth)

    if args.get_out_mem_axi_ratio:
        print(pkg.out_mem_axi_ratio)

    if args.get_num_wgt_mem_uram:
        print(calculate_num_wgt_uram(pkg))

    if args.get_axi_cache_bits:
        print(pkg.axi_cache_bits)

    if args.get_axi_prot_bits:
        print(pkg.axi_prot_bits)

    if args.get_ip_reg_map_range:
        print(pkg.ip_reg_map_range)

    if args.get_fetch_base_addr:
        print(pkg.fetch_base_addr)

    if args.get_load_base_addr:
        print(pkg.load_base_addr)

    if args.get_compute_base_addr:
        print(pkg.compute_base_addr)

    if args.get_store_base_addr:
        print(pkg.store_base_addr)

    if args.get_fpga_dev:
        print(pkg.fpga_device)

    if args.get_fpga_family:
        print(pkg.fpga_family)

    if args.get_fpga_board:
        print(pkg.fpga_board)

    if args.get_fpga_board_rev:
        print(pkg.fpga_board_rev)

    if args.get_fpga_freq:
        print(pkg.fpga_freq)

    if args.get_fpga_per:
        print(pkg.fpga_per)

    if args.export_tcl:
        gen_tcl_vivado(pkg, args.export_tcl)

if __name__ == "__main__":
    main()
