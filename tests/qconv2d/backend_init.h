/* Initialize the selected production simulator backend in this process. */
#pragma once

#include <tvm/runtime/c_runtime_api.h>
#include <vta/runtime.h>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <cstdlib>

inline void CheckTVMStatus(int status, const char* operation) {
  if (status == 0) return;
  const char* detail = TVMGetLastError();
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (detail ? detail : "unknown TVM runtime error"));
}

inline void CheckBackendABI(int64_t fingerprint = VTA_ABI_FINGERPRINT) {
  const int status = VTACheckConfig(fingerprint);
  if (status == 0) return;
  const char* detail = TVMGetLastError();
  throw std::runtime_error(std::string("selected VTA backend ABI mismatch: ") +
                           (detail ? detail : "configuration fingerprint check failed"));
}

inline void InitializeSimulatorBackend() {
  const char* backend = std::getenv("VTA_BACKEND");
  if (!backend || std::string(backend) == "fsim") return;
  if (std::string(backend) != "tsim") {
    throw std::runtime_error("VTA_BACKEND must be fsim or tsim");
  }

  const char* vta_path = std::getenv("VTA_PATH");
  if (!vta_path || !*vta_path) {
    throw std::runtime_error("VTA_PATH is required to initialize TSIM");
  }
#if defined(__APPLE__)
  const char* suffix = "dylib";
#else
  const char* suffix = "so";
#endif
  const std::string hardware_path = std::string(vta_path) + "/build/libvta_hw." + suffix;
  static void* hardware_library = nullptr;
  static TVMModuleHandle hardware_module = nullptr;
  static TVMFunctionHandle initialize = nullptr;
  if (hardware_module) return;

  hardware_library = dlopen(hardware_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!hardware_library) {
    const char* detail = dlerror();
    throw std::runtime_error("unable to load TSIM hardware library " + hardware_path +
                             ": " + (detail ? detail : "unknown loader error"));
  }
  CheckTVMStatus(TVMModLoadFromFile(hardware_path.c_str(), "vta-tsim", &hardware_module),
                 "loading TSIM hardware module");
  CheckTVMStatus(TVMFuncGetGlobal("vta.tsim.init", &initialize),
                 "looking up vta.tsim.init");
  TVMValue argument{};
  argument.v_handle = hardware_module;
  int type_code = kTVMModuleHandle;
  TVMValue result{};
  int result_type = kTVMNullptr;
  CheckTVMStatus(TVMFuncCall(initialize, &argument, &type_code, 1, &result, &result_type),
                 "initializing TSIM hardware module");
}
