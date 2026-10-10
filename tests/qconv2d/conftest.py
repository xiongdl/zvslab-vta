"""Shared paths and compiler helpers for the Q31 ALU tests."""
from pathlib import Path
import os
import shlex
import subprocess

import pytest

VTA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = VTA_ROOT.parent


@pytest.fixture(scope="session")
def compiler_flags():
    flags = shlex.split(os.environ.get("CXXFLAGS", ""))
    return os.environ.get("CXX", "c++"), flags


def compile_probe(source, output, compiler, flags=()):
    command = [compiler, *flags, "-std=c++17", "-I", str(VTA_ROOT / "include"),
               str(source), "-o", str(output)]
    subprocess.run(command, check=True, text=True, capture_output=True)
    return output
