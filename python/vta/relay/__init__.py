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

"""Relay integration for the VTA target extension."""

from .contract import COMPILER_NAME, VTACompilerConfig
from .device_plan import VTADevicePlan, plan_devices_for_vta as _plan_devices_for_vta
from .partition import partition_for_vta
from tvm import relay


def plan_devices_for_vta(module, host_target):
    """Return a heterogeneous VTA build plan with its host target attached."""
    plan = _plan_devices_for_vta(module, host_target)
    planned_module = plan.module
    for global_var, function in list(planned_module.functions.items()):
        if not isinstance(function, relay.Function):
            continue
        if function.attrs is None or "Compiler" not in function.attrs:
            continue
        if function.attrs.get_str("Compiler") != COMPILER_NAME:
            continue
        planned_module[global_var] = function.with_attr("vta.host_target", plan.targets[0])
    return VTADevicePlan(module=planned_module, targets=plan.targets)

__all__ = [
    "COMPILER_NAME",
    "VTACompilerConfig",
    "partition_for_vta",
    "VTADevicePlan",
    "plan_devices_for_vta",
]
