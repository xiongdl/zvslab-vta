"""C-backed instruction layout and Python opcode contracts."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
from conftest import REPO_ROOT, VTA_ROOT, compile_probe


@pytest.mark.parametrize("log_block_in,log_block_out", [(3, 3), (3, 4)])
def test_rounding_field_preserves_existing_layout(tmp_path, compiler_flags, log_block_in, log_block_out):
    compiler, flags = compiler_flags
    cfg_path = VTA_ROOT / "config" / "vta_64mac.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(LOG_BLOCK_IN=log_block_in, LOG_BLOCK_OUT=log_block_out)
    variant = tmp_path / "config.json"
    variant.write_text(json.dumps(cfg))
    # Compile the C bitfield against actual generated geometry definitions.
    cflags = subprocess.run([sys.executable, str(VTA_ROOT / "config" / "vta_config.py"),
                             "--use-cfg=" + str(variant), "--cflags"], check=True,
                            capture_output=True, text=True).stdout
    flags.extend(shlex.split(cflags))
    probe_src = tmp_path / "encoding_probe.cc"
    probe_src.write_text(r'''
#include <vta/hw_spec.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  VTAAluInsn insn{};
  insn.opcode = VTA_OPCODE_ALU; insn.alu_opcode = std::atoi(argv[1]);
  insn.use_imm = 1; insn.imm = -1234; insn.rounding = std::atoi(argv[2]);
  uint64_t words[2] = {}; std::memcpy(words, &insn, sizeof(insn));
  std::printf("%zu %llu %llu %u %u %lld %u %u\n", sizeof(insn),
      (unsigned long long)words[0], (unsigned long long)words[1],
      VTA_ALU_OPCODE_BIT_OFFSET, VTA_ALU_ROUNDING_BIT_OFFSET,
      (long long)insn.imm, (unsigned)insn.alu_opcode, (unsigned)insn.rounding);
}
''')
    exe = compile_probe(probe_src, tmp_path / "encoding_probe", compiler, flags)
    expected_opcode_lsb, expected_round_lsb = (
        (104, 124) if log_block_out == 3 else (100, 120)
    )
    assert expected_opcode_lsb == expected_round_lsb - 20
    assert expected_round_lsb + 2 <= 126
    for opcode in (5, 6):
        for rounding in (0, 1, 2):
            output = subprocess.run([str(exe), str(opcode), str(rounding)], check=True,
                                    text=True, capture_output=True).stdout.split()
            size, low, high, opcode_lsb, rounding_lsb, imm, actual_opcode, actual_rounding = map(int, output)
            assert size == 16
            assert (actual_opcode, actual_rounding, imm) == (opcode, rounding, -1234)
            # These constants are independent expectations from inspecting the C bitfield bytes.
            assert (opcode_lsb, rounding_lsb) == (expected_opcode_lsb, expected_round_lsb)
            encoded = low | (high << 64)
            assert (encoded >> rounding_lsb) & 0b11 == rounding
            assert (encoded >> opcode_lsb) & 0b111 == opcode
            assert (encoded >> (opcode_lsb + 4)) & 0xffff == ((-1234) & 0xffff)
            assert encoded >> (rounding_lsb + 2) == 0


def test_python_opcode_and_rounding_constants():
    env = os.environ.copy()
    env["TVM_PATH"] = str(REPO_ROOT / "tvm")
    env["VTA_PATH"] = str(VTA_ROOT)
    env["VTA_CONFIG_FILE"] = str(VTA_ROOT / "config" / "vta_64mac.json")
    env["VTA_BACKEND"] = "fsim"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "tvm" / "python"), str(VTA_ROOT / "python")])
    result = subprocess.run([sys.executable, "-c", "import json,os; from vta.environment import Environment; cfg=json.load(open(os.environ['VTA_CONFIG_FILE'])); cfg['TARGET']='sim'; e=Environment(cfg).dev; print(e.ALU_OPCODE_RMUL,e.ALU_OPCODE_RSFT,e.ALU_ROUND_NONE,e.ALU_ROUND_UP,e.ALU_ROUND_AWAY)"],
                            check=True, capture_output=True, text=True, env=env)
    assert result.stdout.strip() == "5 6 0 1 2"
