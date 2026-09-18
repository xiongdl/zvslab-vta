/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/relay/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

namespace tvm {

using FTVMTIRToRuntime = runtime::TypedPackedFunc<runtime::Module(IRModule, Target)>;

namespace vta {

transform::Pass RelayToTIR();

runtime::Module TIRToRuntime(IRModule mod, Target target);
void ValidateActiveVTAHost(const Target& target);

transform::Pass ModernRelayToTIR() {
  auto bind_vta_target = tir::transform::CreatePrimFuncPass(
      [](tir::PrimFunc func, IRModule module, transform::PassContext) {
        Optional<Target> lowered_target = func->GetAttr<Target>(tvm::attr::kTarget);
        if (!lowered_target.defined() || !lowered_target.value()->HasKey("vta")) {
          return func;
        }
        Target host(nullptr);
        Optional<Target> planned_host;
        Optional<DictAttrs> relay_attrs = func->GetAttr<DictAttrs>("relay_attrs");
        if (relay_attrs.defined()) {
          planned_host = relay_attrs.value().GetAttr<Target>("vta.host_target");
        }
        if (!planned_host.defined()) {
          planned_host = module->GetAttr<Target>("vta.host_target");
        }
        if (planned_host.defined()) {
          Target planned_vta_target = Target::WithHost(Target("vta"), planned_host.value());
          ValidateActiveVTAHost(planned_vta_target);
          host = planned_host.value();
        } else {
          Target active_target = Target::Current(true);
          ValidateActiveVTAHost(active_target);
          host = active_target->GetHost().value();
        }
        return WithAttrs(std::move(func),
                         {{tvm::attr::kTarget, Target::WithHost(Target("vta"), host)},
                          {"vta.route_to_runtime", Bool(true)}});
      },
      0, "vta.BindModernTarget", {});
  auto route_to_runtime = tir::transform::CreatePrimFuncPass(
      [](tir::PrimFunc func, IRModule, transform::PassContext) {
        if (!func->HasNonzeroAttr("vta.route_to_runtime")) {
          return func;
        }
        Target host = func->GetAttr<Target>(tvm::attr::kTarget).value();
        return WithAttrs(
            std::move(func),
            {{tvm::attr::kTarget, Target::WithHost(Target("vta"), host)},
             {tvm::attr::kCallingConv, Integer(CallingConv::kDeviceKernelLaunch)}});
      },
      0, "vta.RouteToRuntime", {});
  transform::Sequential pipeline = transform::Sequential(
      {RelayToTIR(), bind_vta_target, tir::transform::MakePackedAPI(), route_to_runtime},
      "vta.ModernRelayToTIR");
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [pipeline](IRModule mod, transform::PassContext pass_context) {
        return pipeline(std::move(mod), pass_context);
      };
  return transform::CreateModulePass(pass_func, 0, "vta.ModernRelayToTIR", {});
}

}  // namespace vta

TVM_REGISTER_TARGET_KIND("vta", kDLExtDev)
    .set_default_keys({"cpu"})
    .set_attr<Bool>("use_device_api", Bool(true))
    .set_attr<relay::transform::FTVMRelayToTIR>(attr::kRelayToTIR, vta::ModernRelayToTIR())
    .set_attr<FTVMTIRToRuntime>("TIRToRuntime", vta::TIRToRuntime);

}  // namespace tvm
