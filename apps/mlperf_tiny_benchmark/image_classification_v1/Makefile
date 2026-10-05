# ResNet-8 deployment and two-stage VTA tuning workflow.

APP_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_DIR := $(abspath $(APP_DIR)/../../../../)

ifeq ($(origin CONFIG),undefined)
CONFIG := $(REPO_DIR)/vta/config/vta_64mac.json
endif
ifeq ($(origin MODEL),undefined)
MODEL := $(APP_DIR)/model/pretrainedResnet.tflite
endif
ifeq ($(origin INPUT),undefined)
INPUT := $(APP_DIR)/samples/00-airplane.png
endif
ifeq ($(origin TARGET),undefined)
TARGET := vta,llvm
endif
ifeq ($(origin SIMULATOR),undefined)
SIMULATOR := fsim
endif
ifeq ($(origin PYTHON),undefined)
PYTHON := $(REPO_DIR)/.envs/tvm-vta-env/bin/python
endif

export CONFIG MODEL INPUT TARGET SIMULATOR PYTHON
export SCHEDULE OUTPUT_DIR REPORT EXPORT_WORKLOADS WORKLOADS WORKLOAD
export TRIAL_BATCH MIN_SUCCESSFUL TIMEOUT FSIM_TIMEOUT TSIM_TIMEOUT
export INPUT_LOGS OUTPUT_LOGS

deploy:
	@$(APP_DIR)/scripts/make_tasks.sh deploy

tune-fsim:
	@$(APP_DIR)/scripts/make_tasks.sh tune-fsim

tune-tsim:
	@$(APP_DIR)/scripts/make_tasks.sh tune-tsim

tune:
	@$(APP_DIR)/scripts/make_tasks.sh tune

clean:
	@$(APP_DIR)/scripts/make_tasks.sh clean

.PHONY: deploy tune-fsim tune-tsim tune clean
