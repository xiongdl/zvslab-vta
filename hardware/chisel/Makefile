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

VERILATOR ?= verilator
VERILATOR_ROOT ?= $(shell $(VERILATOR) --getenv VERILATOR_ROOT 2>/dev/null)
VERILATOR_INC_DIR ?= $(VERILATOR_ROOT)/include

CONFIG = DefaultTsimConfig
TOP = VTA
TOP_TEST = Test
BUILD_NAME = build
# Set USE_TRACE = 1 to generate a trace during simulation.
USE_TRACE = 0
# With USE_TRACE = 1, default trace format is VCD.
# Set USE_TRACE_FST = 1 to use the FST format.
# Note that although FST is around two orders of magnitude smaller than VCD
# it is also currently much slower to produce (verilator limitation). But if
# you are low on disk space it may be your only option.
USE_TRACE_FST = 0
# With USE_TRACE = 1, USE_TRACE_DETAILED = 1 will generate traces that also
# include non-interface internal signal names starting with an underscore.
# This will significantly increase the trace size and should only be used
# on a per need basis for difficult debug problems.
USE_TRACE_DETAILED = 0
USE_THREADS = 0
VTA_LIBNAME = libvta_hw
VTA_CONFIG_FILE ?=
ifeq ($(origin CXX), default)
  CXX := c++
endif
CXX_STANDARD ?= c++17
# A debug build with DEBUG = 1 is useful to trace the simulation with a
# debugger.
DEBUG = 0
# With DEBUG = 1, SANITIZE = 1 turns on address sanitizing to verify that
# the verilator build is sane. To be used if you know what you are doing.
SANITIZE = 0

CXX_MAJOR := $(shell $(CXX) -dumpversion | sed 's/\..*//')
CXX_HAS_ALIGN_NEW := $(shell [ $(CXX_MAJOR) -ge 7 ] && echo true)
CXX_IS_CLANG := $(shell $(CXX) --version 2>/dev/null | head -1 | grep -i clang)

CONFIG_TEST = $(TOP_TEST)$(CONFIG)

ifndef TVM_PATH
   TVM_PATH := $(abspath ../../../../)
endif

ifndef VTA_PATH
   VTA_PATH := $(abspath ../../)
endif

SBT ?= sbt
verilator_build_dir = $(VTA_PATH)/$(BUILD_NAME)/verilator
chisel_build_dir = $(VTA_PATH)/$(BUILD_NAME)/chisel
CHISEL_CONFIG_FILE ?= $(chisel_build_dir)/vta_geometry.properties
PYTHON ?= $(VTA_PATH)/../.envs/tvm-vta-env/bin/python
SBT_ENV = HOME=$(VTA_PATH)/hardware/chisel/dep/home \
	XDG_CACHE_HOME=$(VTA_PATH)/hardware/chisel/dep/xdg-cache \
	COURSIER_CACHE=$(VTA_PATH)/hardware/chisel/dep/coursier \
	SBT_OPTS='-Duser.home=$(VTA_PATH)/hardware/chisel/dep/home -Dvta.geometry.properties=$(abspath $(CHISEL_CONFIG_FILE))'

verilator_opt = --cc
verilator_opt += +define+RANDOMIZE_GARBAGE_ASSIGN
verilator_opt += +define+RANDOMIZE_REG_INIT
verilator_opt += +define+RANDOMIZE_MEM_INIT
verilator_opt += --x-assign unique
verilator_opt += --output-split 20000
verilator_opt += --output-split-cfuncs 20000
verilator_opt += --top-module ${TOP_TEST}
verilator_opt += -Mdir ${verilator_build_dir}
verilator_opt += -I$(chisel_build_dir)

ifeq ($(DEBUG), 0)
	cxx_flags = -O2 -Wall -fvisibility=hidden
else
	cxx_flags = -O0 -g -Wall
endif

cxx_flags += -std=$(CXX_STANDARD)
ifneq ($(CXX_IS_CLANG),)
	cxx_flags += -Wno-uninitialized
else
	cxx_flags += -Wno-maybe-uninitialized
endif
ifeq ($(CXX_HAS_ALIGN_NEW),true)
	cxx_flags += -faligned-new
endif
cxx_flags += -DVL_TSIM_NAME=V$(TOP_TEST)
cxx_flags += -DVL_PRINTF=printf
cxx_flags += -DVL_USER_FINISH
cxx_flags += -DVM_COVERAGE=0
cxx_flags += -DVM_SC=0
cxx_flags += -Wno-sign-compare
cxx_flags += -include V$(TOP_TEST).h
cxx_flags += -I$(verilator_build_dir)
cxx_flags += -I$(VERILATOR_INC_DIR)
cxx_flags += -I$(VERILATOR_INC_DIR)/vltstd
cxx_flags += -I$(VTA_PATH)/include
cxx_flags += -I$(TVM_PATH)/include
cxx_flags += -I$(TVM_PATH)/3rdparty/dlpack/include

ld_flags = -fPIC -shared

ifeq ($(SANITIZE), 1)
	ifeq ($(DEBUG), 1)
		cxx_flags += -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address
		ld_flags += -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address
	endif
endif

cxx_objs = $(verilator_build_dir)/tsim_device.o
verilated_model_lib = $(verilator_build_dir)/libV$(TOP_TEST).a
verilated_runtime_lib = $(verilator_build_dir)/libverilated.a
verilated_lib_stamp = $(verilator_build_dir)/.vta-verilated-libs

ifneq ($(USE_TRACE), 0)
	cxx_flags += -DVM_TRACE=1
	ifeq ($(USE_TRACE_FST), 1)
		cxx_flags += -DVM_TRACE_FST
		verilator_opt += --trace-fst
	else
		verilator_opt += --trace
	endif
	ifeq ($(USE_TRACE_DETAILED), 1)
		verilator_opt += --trace-underscore --trace-structs
	endif
	ifeq ($(USE_TRACE_FST), 1)
		cxx_flags += -DTSIM_TRACE_FILE=$(verilator_build_dir)/$(TOP_TEST).fst
		cxx_objs += $(verilator_build_dir)/verilated_fst_c.o
	else
		cxx_flags += -DTSIM_TRACE_FILE=$(verilator_build_dir)/$(TOP_TEST).vcd
		cxx_objs += $(verilator_build_dir)/verilated_vcd_c.o
	endif
else
	cxx_flags += -DVM_TRACE=0
endif

ifneq ($(USE_THREADS), 0)
	verilator_opt += --threads $(USE_THREADS)
	cxx_flags += -DVL_THREADED
endif

VPATH = $(VERILATOR_INC_DIR):$(verilator_build_dir):$(VTA_PATH)/hardware/dpi

# The following is to be consistent with cmake
ifeq ($(shell uname), Darwin)
	lib_path = $(VTA_PATH)/$(BUILD_NAME)/$(VTA_LIBNAME).dylib
	MACOS_SDK_PATH ?= $(shell xcrun --show-sdk-path 2>/dev/null)
	cxx_flags += -isysroot $(MACOS_SDK_PATH)
	ld_flags += -undefined dynamic_lookup
else
	lib_path = $(VTA_PATH)/$(BUILD_NAME)/$(VTA_LIBNAME).so
endif

default: lint lib

.PHONY: check_vta_config check_verilator deps
check_vta_config: $(CHISEL_CONFIG_FILE)
	@test -n "$(VTA_CONFIG_FILE)" || { echo "Error: VTA_CONFIG_FILE is required for hardware generation." >&2; exit 1; }
	@test -f "$(VTA_CONFIG_FILE)" || { echo "Error: VTA_CONFIG_FILE does not exist: $(VTA_CONFIG_FILE)" >&2; exit 1; }
	@test -s "$(CHISEL_CONFIG_FILE)" || { echo "Error: normalized Chisel geometry was not generated: $(CHISEL_CONFIG_FILE)" >&2; exit 1; }
	@$(PYTHON) $(VTA_PATH)/config/vta_config.py --use-cfg=$(VTA_CONFIG_FILE) --backend-contract --defs >/dev/null || { echo "Error: invalid geometry configuration: $(VTA_CONFIG_FILE)" >&2; exit 1; }

$(CHISEL_CONFIG_FILE): $(VTA_CONFIG_FILE) $(VTA_PATH)/config/vta_config.py
	@mkdir -p $(@D)
	$(PYTHON) $(VTA_PATH)/config/vta_config.py --use-cfg=$(VTA_CONFIG_FILE) --chisel-properties=$@

check_verilator:
	@command -v "$(VERILATOR)" >/dev/null 2>&1 || { \
		echo "Error: Verilator was not found in PATH." >&2; \
		exit 1; \
	}
	@test -f "$(VERILATOR_INC_DIR)/verilated.cpp" || { \
		echo "Error: Verilator include directory is invalid: $(VERILATOR_INC_DIR)" >&2; \
		exit 1; \
	}

deps:
	$(SBT_ENV) $(SBT) update test:compile

lint:
	$(SBT_ENV) $(SBT) scalastyle
	$(SBT_ENV) $(SBT) test:scalastyle

lib: check_vta_config $(lib_path)

$(verilator_build_dir)/%.o: %.cpp
	$(CXX) -fPIC $(cxx_flags) -c $^ -o $@

$(verilator_build_dir)/tsim_device.o: tsim_device.cc
	$(CXX) -fPIC $(cxx_flags) -c $^ -o $@

$(lib_path): $(verilated_lib_stamp) $(cxx_objs) | check_verilator
	$(CXX) $(ld_flags) $(cxx_flags) $(cxx_objs) $(verilated_model_lib) $(verilated_runtime_lib) -o $@

$(verilated_lib_stamp): $(verilator_build_dir)/V$(TOP_TEST).cpp
	$(MAKE) -C $(verilator_build_dir) -f V$(TOP_TEST).mk \
		CXX="$(CXX)" CXXFLAGS="-fPIC $(cxx_flags)" \
		libV$(TOP_TEST).a libverilated.a
	@test -f "$(verilated_model_lib)"
	@test -f "$(verilated_runtime_lib)"
	@touch $@

verilator: $(verilator_build_dir)/V$(TOP_TEST).cpp
$(cxx_objs): | $(verilator_build_dir)/V$(TOP_TEST).cpp
$(verilator_build_dir)/V$(TOP_TEST).cpp: $(chisel_build_dir)/$(TOP_TEST).$(CONFIG).sv | check_verilator
	$(VERILATOR) $(verilator_opt) $<

verilog: $(chisel_build_dir)/$(TOP).$(CONFIG).sv
$(chisel_build_dir)/$(TOP).$(CONFIG).sv: $(CHISEL_CONFIG_FILE)
	$(SBT_ENV) $(SBT) 'runMain vta.$(CONFIG) --target-dir $(chisel_build_dir) -o $(TOP).$(CONFIG)'

verilog_test: $(chisel_build_dir)/$(TOP_TEST).$(CONFIG).sv
$(chisel_build_dir)/$(TOP_TEST).$(CONFIG).sv: $(CHISEL_CONFIG_FILE)
	$(SBT_ENV) $(SBT) 'runMain vta.$(CONFIG_TEST) --target-dir $(chisel_build_dir) -o $(TOP_TEST).$(CONFIG)'

unittest: check_verilator
	$(SBT_ENV) $(SBT) test

clean:
	-rm -rf target project/target project/project test_run_dir

cleanall:
	-rm -rf $(VTA_PATH)/$(BUILD_NAME)/chisel
	-rm -rf $(VTA_PATH)/$(BUILD_NAME)/libvta_hw.so
	-rm -rf $(VTA_PATH)/$(BUILD_NAME)/libvta_hw.dylib
	-rm -rf $(VTA_PATH)/$(BUILD_NAME)/verilator
