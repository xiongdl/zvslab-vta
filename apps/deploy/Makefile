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

# Makefile Example to deploy TVM modules.
VTA_PATH ?= $(abspath ../..)
CUR_DIR=$(shell pwd)
DMLC_CORE=${TVM_PATH}/3rdparty/dmlc-core
PYTHON ?= $(abspath ../../../.envs/tvm-vta-env/bin/python)
VTA_CONFIG_FILE ?= $(VTA_PATH)/config/vta_64mac.json
VTA_BACKEND ?= fsim

ifeq ($(VTA_BACKEND),fsim)
	VTA_LIB=vta_fsim
else ifeq ($(VTA_BACKEND),tsim)
	VTA_LIB=vta_tsim
else
$(error Unsupported VTA_BACKEND=$(VTA_BACKEND) for this simulator deploy example; use fsim or tsim)
endif

PKG_CFLAGS = -std=c++14 -O0 -g -fPIC\
						 -I${TVM_PATH}/include\
						 -I${VTA_PATH}/include\
						 -I${DMLC_CORE}/include\
						 -I${TVM_PATH}/3rdparty/dlpack/include\
						 -I${TVM_PATH}/\

PKG_LDFLAGS = -L${TVM_PATH}/build -L${VTA_PATH}/build -L${CUR_DIR} -ldl -pthread -l${VTA_LIB} -ltvm_runtime

.PHONY: clean all

all:./build/deploy copylib

./build/deploy: ./build/deploy.o ./build/model/lib.so
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

./build/deploy.o: cpp_deploy.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

./build/model/lib.so: ./build/model/lib.o
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -shared

copylib: ${TVM_PATH}/build/libtvm_runtime.so ${VTA_PATH}/build/lib${VTA_LIB}.so
	@cp ${TVM_PATH}/build/libtvm_runtime.so ./build
	@cp ${VTA_PATH}/build/lib${VTA_LIB}.so ./build

clean:
	rm -rf  ./build/*.o ./build/deploy
