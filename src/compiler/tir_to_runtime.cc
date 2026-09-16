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

#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace vta {
namespace {

constexpr const char* kLegacyTEScheduleAttr = "from_legacy_te_schedule";

[[noreturn]] void FailValidation(const std::string& message) {
  TVMAPISetLastError(message.c_str());
  throw runtime::EnvErrorAlreadySet(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) {
    FailValidation(message);
  }
}

std::string HostKindName(const Optional<Target>& host) {
  if (!host.defined() || !host.value().defined()) {
    return "<missing>";
  }
  return host.value()->kind->name;
}

std::string UnsupportedHostMessage(const std::string& prefix, const Optional<Target>& host) {
  return prefix + " rejected host '" + HostKindName(host) +
         "'; supported host kinds are llvm and c";
}

bool StructurallyEqualTargets(const Target& lhs, const Target& rhs) {
  return StructuralEqual()(lhs, rhs);
}

}  // namespace

bool IsSupportedHostTarget(const Target& target) {
  return target.defined() &&
         (target->kind->name == "llvm" || target->kind->name == "c");
}

void ValidateActiveVTAHost(const Target& target) {
  Require(target.defined() && target->kind->name == "vta",
          "VTA RelayToTIR requires an active vta target");
  Optional<Target> host = target->GetHost();
  Require(host.defined() && IsSupportedHostTarget(host.value()),
          UnsupportedHostMessage("VTA RelayToTIR", host));
}

namespace {

bool IsRawVTAFunctionTarget(const Target& target) {
  return target->kind->name == "vta" || target->HasKey("vta");
}

class RuntimeCallValidator : public tir::StmtExprVisitor {
 public:
  explicit RuntimeCallValidator(std::string symbol) : symbol_(std::move(symbol)) {}

  void Validate(const tir::Stmt& body) {
    VisitStmt(body);
    Require(has_vta_activity_,
            "VTA PrimFunc " + symbol_ + " does not contain a recognized VTA runtime call");
  }

 private:
  void VisitExpr_(const tir::CallNode* call) final {
    if (call->op.same_as(tir::builtin::call_extern())) {
      Require(call->args.size() > 0,
              "VTA PrimFunc " + symbol_ + " contains a malformed runtime call");
      const auto* name = call->args[0].as<tir::StringImmNode>();
      Require(name != nullptr, "VTA PrimFunc " + symbol_ + " contains a malformed runtime call");
      static const std::unordered_set<std::string> kRuntimeCalls = {
          "VTABufferCPUPtr", "VTADepPop",        "VTADepPush",      "VTALoadBuffer2D",
          "VTASetDebugMode", "VTAStoreBuffer2D", "VTASynchronize",  "VTATLSCommandHandle",
          "VTAUopLoopBegin", "VTAUopLoopEnd",    "VTAUopPush"};
      Require(kRuntimeCalls.count(name->value),
              "VTA PrimFunc " + symbol_ + " contains unsupported runtime call " +
                  std::string(name->value));
      has_vta_activity_ = true;
    } else if (const auto* op = call->op.as<OpNode>()) {
      static const std::unordered_set<std::string> kVTAOps = {
          "tir.vta.command_handle", "tir.vta.coproc_sync", "tir.vta.coproc_dep_push",
          "tir.vta.coproc_dep_pop", "tir.vta.uop_push"};
      std::string op_name = op->name;
      if (op_name.rfind("tir.vta.", 0) == 0) {
        Require(kVTAOps.count(op_name),
                "VTA PrimFunc " + symbol_ + " contains unsupported runtime call " + op_name);
        has_vta_activity_ = true;
      }
    }
    tir::StmtExprVisitor::VisitExpr_(call);
  }

  std::string symbol_;
  bool has_vta_activity_{false};
};

class CHostCallRewriter : public tir::StmtExprMutator {
 private:
  PrimExpr VisitExpr_(const tir::CallNode* call) final {
    const auto* op = call->op.as<OpNode>();
    if (op != nullptr && op->name == "tir.vta.command_handle") {
      return tir::Call(DataType::Handle(), tir::builtin::call_extern(),
                       {tir::StringImm("VTATLSCommandHandle")});
    }
    if (op != nullptr && op->name == "tir.vta.uop_push") {
      Array<PrimExpr> args{tir::StringImm("VTAUopPush")};
      for (const PrimExpr& arg : call->args) {
        args.push_back(arg);
      }
      return tir::Call(call->dtype, tir::builtin::call_extern(), std::move(args));
    }
    if (call->op.same_as(tir::builtin::call_extern()) && call->args.size() > 0) {
      const auto* name = call->args[0].as<tir::StringImmNode>();
      if (name != nullptr) {
        static const std::unordered_map<std::string, size_t> kAddressArguments = {
            {"VTABufferCPUPtr", 2}, {"VTAWriteBarrier", 2}, {"VTAReadBarrier", 2},
            {"VTALoadBuffer2D", 2}, {"VTAStoreBuffer2D", 4}};
        auto it = kAddressArguments.find(name->value);
        if (it != kAddressArguments.end() && call->args.size() > it->second) {
          Array<PrimExpr> args;
          for (size_t index = 0; index < call->args.size(); ++index) {
            PrimExpr arg = call->args[index];
            if (index == it->second) {
              arg = tir::Cast(DataType::Handle(), arg);
            }
            args.push_back(std::move(arg));
          }
          return tir::Call(call->dtype, tir::builtin::call_extern(), std::move(args));
        }
      }
    }
    return tir::StmtExprMutator::VisitExpr_(call);
  }
};

IRModule LowerVTAOpsForC(IRModule mod) {
  mod = mod->ShallowCopy();
  for (const auto& [global_var, base_func] : mod->functions) {
    tir::PrimFunc prim_func = Downcast<tir::PrimFunc>(base_func);
    CHostCallRewriter rewriter;
    prim_func.CopyOnWrite()->body = rewriter(std::move(prim_func->body));
    mod->Update(global_var, std::move(prim_func));
  }
  return mod;
}

void ValidateModule(const IRModule& mod, const Target& target) {
  Require(target.defined() && target->kind->name == "vta",
          "VTA TIRToRuntime requires a vta target");
  Optional<Target> host = target->GetHost();
  Require(host.defined() && IsSupportedHostTarget(host.value()),
          UnsupportedHostMessage("VTA TIRToRuntime", host));
  Require(mod->functions.size() > 0, "VTA TIRToRuntime does not accept an empty module");

  std::unordered_set<std::string> symbols;
  for (const auto& [global_var, base_func] : mod->functions) {
    const auto* prim_func_node = base_func.as<tir::PrimFuncNode>();
    Require(prim_func_node != nullptr,
            "VTA TIRToRuntime requires every function to be a PrimFunc; " +
                std::string(global_var->name_hint) + " is not a PrimFunc");
    tir::PrimFunc prim_func = GetRef<tir::PrimFunc>(prim_func_node);
    Optional<String> global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    if (!global_symbol.defined()) {
      FailValidation("VTA PrimFunc " + std::string(global_var->name_hint) +
                     " is missing global_symbol");
    }
    std::string symbol = global_symbol.value();
    Require(symbols.insert(symbol).second,
            "VTA module contains duplicate global_symbol " + symbol);
  }

  for (const auto& [global_var, base_func] : mod->functions) {
    const auto* prim_func_node = base_func.as<tir::PrimFuncNode>();
    tir::PrimFunc prim_func = GetRef<tir::PrimFunc>(prim_func_node);
    Optional<String> global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    std::string symbol = global_symbol.value();
    Require(global_var->name_hint == symbol,
            "VTA PrimFunc " + std::string(global_var->name_hint) +
                " has mismatched global_symbol " + symbol);

    Optional<Target> function_target = prim_func->GetAttr<Target>(tvm::attr::kTarget);
    Require(function_target.defined(), "VTA PrimFunc " + symbol + " is missing target");
    auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
    bool is_packed = calling_conv.defined() &&
                     calling_conv.value()->value == static_cast<int>(CallingConv::kCPackedFunc);
    if (is_packed) {
      Require(IsSupportedHostTarget(function_target.value()),
              UnsupportedHostMessage("VTA PrimFunc " + symbol + " packed", function_target));
      Require(StructurallyEqualTargets(function_target.value(), host.value()),
              "VTA PrimFunc " + symbol + " packed host does not match selected host");
    } else {
      Require(IsRawVTAFunctionTarget(function_target.value()),
              "VTA PrimFunc " + symbol + " has invalid target");
      Optional<Target> function_host = function_target.value()->GetHost();
      Require(function_host.defined() && IsSupportedHostTarget(function_host.value()),
              UnsupportedHostMessage("VTA PrimFunc " + symbol + " target", function_host));
      Require(StructurallyEqualTargets(function_host.value(), host.value()),
              "VTA PrimFunc " + symbol + " host does not match selected host");
    }

    RuntimeCallValidator(symbol).Validate(prim_func->body);
  }
}

IRModule ForceFlattenExternalBuffers(IRModule mod) {
  mod = mod->ShallowCopy();
  for (const auto& [global_var, base_func] : mod->functions) {
    tir::PrimFunc prim_func = Downcast<tir::PrimFunc>(base_func);
    prim_func = WithoutAttr(std::move(prim_func), kLegacyTEScheduleAttr);
    if (prim_func->HasNonzeroAttr("vta.route_to_runtime")) {
      Target routed_target = prim_func->GetAttr<Target>(tvm::attr::kTarget).value();
      Target host = routed_target->GetHost().value();
      prim_func = WithAttrs(std::move(prim_func),
                            {{tvm::attr::kTarget, host},
                             {tvm::attr::kCallingConv, Integer(CallingConv::kCPackedFunc)}});
      prim_func = WithoutAttr(std::move(prim_func), "vta.route_to_runtime");
    }
    mod->Update(global_var, std::move(prim_func));
  }
  return tir::transform::FlattenBuffer()(std::move(mod));
}

IRModule InjectConfigChecks(IRModule mod) {
  mod = mod->ShallowCopy();
  for (const auto& [global_var, base_func] : mod->functions) {
    tir::PrimFunc prim_func = Downcast<tir::PrimFunc>(base_func);
    tir::Call check(DataType::Int(32), tir::builtin::call_extern(),
                    {tir::StringImm("VTACheckConfig"),
                     IntImm(DataType::Int(64),
                            static_cast<int64_t>(static_cast<uint64_t>(VTA_ABI_FINGERPRINT)))});
    tir::Stmt fail = tir::Evaluate(
        tir::Call(DataType::Int(32), tir::builtin::tvm_throw_last_error(), {}));
    tir::Stmt guarded_body = tir::SeqStmt(
        {tir::IfThenElse(check != IntImm(DataType::Int(32), 0), fail), prim_func->body});
    prim_func.CopyOnWrite()->body = std::move(guarded_body);
    mod->Update(global_var, std::move(prim_func));
  }
  return mod;
}

}  // namespace

runtime::Module TIRToRuntime(IRModule mod, Target target) {
  ValidateModule(mod, target);
  Target host = target->GetHost().value();

  IRModule lowered = ForceFlattenExternalBuffers(std::move(mod));
  lowered = tir::transform::MakePackedAPI()(std::move(lowered));
  if (host->kind->name == "c") {
    lowered = tir::transform::VectorizeLoop(false)(std::move(lowered));
    lowered = LowerVTAOpsForC(std::move(lowered));
  }
  lowered = transform::Sequential({tir::transform::BindTarget(host),
                                   tir::transform::LowerTVMBuiltin(),
                                   tir::transform::LowerCustomDatatypes(),
                                   tir::transform::LowerIntrin(),
                                   tir::transform::LowerDeviceStorageAccessInfo(),
                                   tir::transform::CombineContextCall()})(std::move(lowered));
  lowered = InjectConfigChecks(std::move(lowered));
  return codegen::Build(std::move(lowered), host);
}

}  // namespace vta
}  // namespace tvm
