cmake_minimum_required(VERSION 3.18)

project(vta LANGUAGES CXX)

set(TVM_PATH "" CACHE PATH "Path to the TVM source tree")
set(VTA_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Path to the VTA source tree")
set(VTA_CONFIG_FILE "${VTA_PATH}/config/vta_64mac.json" CACHE STRING
    "Path to the VTA configuration JSON file")
set(VTA_BACKEND "all" CACHE STRING "VTA backend to build: fsim, tsim, or all")

if(NOT VTA_BACKEND MATCHES "^(fsim|tsim|all)$")
  message(FATAL_ERROR "VTA_BACKEND must be fsim, tsim, or all")
endif()

if(NOT TVM_PATH)
  message(FATAL_ERROR "TVM_PATH is required and must point to the TVM source tree")
endif()

file(REAL_PATH "${TVM_PATH}" TVM_PATH)
file(REAL_PATH "${VTA_PATH}" VTA_PATH)
if(NOT EXISTS "${VTA_PATH}/config/vta_config.py")
  message(FATAL_ERROR "VTA_PATH must point to the VTA source tree: ${VTA_PATH}")
endif()
if(NOT IS_ABSOLUTE "${VTA_CONFIG_FILE}")
  message(FATAL_ERROR "VTA_CONFIG_FILE must be an absolute path")
endif()
if(NOT EXISTS "${VTA_CONFIG_FILE}")
  message(FATAL_ERROR "VTA_CONFIG_FILE does not exist: ${VTA_CONFIG_FILE}")
endif()

set(TVM_INCLUDE_DIR "${TVM_PATH}/include")
set(TVM_WORKSPACE_POOL_HEADER
    "${TVM_INCLUDE_DIR}/../src/runtime/workspace_pool.h")

foreach(required_path
        "${TVM_INCLUDE_DIR}/tvm/runtime/registry.h"
        "${TVM_WORKSPACE_POOL_HEADER}"
        "${TVM_PATH}/3rdparty/dlpack/include/dlpack/dlpack.h"
        "${TVM_PATH}/3rdparty/dmlc-core/include/dmlc/logging.h")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "Required TVM dependency was not found: ${required_path}")
  endif()
endforeach()

find_library(
  TVM_LIBRARY
  NAMES tvm
  PATHS "${TVM_PATH}/build"
  NO_DEFAULT_PATH
  REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${VTA_PATH}/config/vta_config.py"
          "--use-cfg=${VTA_CONFIG_FILE}" --backend-contract --defs
  RESULT_VARIABLE VTA_CONFIG_DEFS_RESULT
  OUTPUT_VARIABLE VTA_DEFINITIONS_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT VTA_CONFIG_DEFS_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to read the VTA compile definitions")
endif()
separate_arguments(VTA_DEFINITIONS NATIVE_COMMAND "${VTA_DEFINITIONS_RAW}")
list(TRANSFORM VTA_DEFINITIONS REPLACE "^-D" "")

set(VTA_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
set(VTA_ABI_HEADER "${VTA_GENERATED_INCLUDE_DIR}/vta/abi_config.h")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${VTA_PATH}/config/vta_config.py"
          "--use-cfg=${VTA_CONFIG_FILE}" "--abi-header=${VTA_ABI_HEADER}"
  RESULT_VARIABLE VTA_ABI_HEADER_RESULT)
if(NOT VTA_ABI_HEADER_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to generate the VTA ABI configuration header")
endif()
set_property(
  DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${VTA_CONFIG_FILE}"
  "${VTA_PATH}/config/vta_config.py"
  "${VTA_PATH}/config/pkg_config.py")

set(
  VTA_RUNTIME_SOURCES
  "${VTA_PATH}/src/runtime/device_api.cc"
  "${VTA_PATH}/src/runtime/runtime.cc"
  "${VTA_PATH}/src/vmem/virtual_memory.cc")

function(configure_vta_runtime target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(
    ${target}
    PRIVATE
      DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
      ${VTA_DEFINITIONS})
  target_include_directories(
    ${target}
    PRIVATE
      "${VTA_GENERATED_INCLUDE_DIR}"
      "${VTA_PATH}/include"
      "${TVM_INCLUDE_DIR}"
      "${TVM_PATH}/3rdparty/dlpack/include"
      "${TVM_PATH}/3rdparty/dmlc-core/include")
  if(MSVC)
    target_compile_options(${target} PRIVATE "/FI${VTA_ABI_HEADER}")
  else()
    target_compile_options(${target} PRIVATE -include "${VTA_ABI_HEADER}")
  endif()
  target_link_libraries(${target} PRIVATE "${TVM_LIBRARY}")
  set_target_properties(
    ${target}
    PROPERTIES
      CXX_EXTENSIONS OFF
      BUILD_RPATH "${TVM_PATH}/build")
endfunction()

add_library(
  tvm_vta_ext SHARED
  "${VTA_PATH}/src/compiler/target.cc"
  "${VTA_PATH}/src/compiler/relay_to_tir.cc"
  "${VTA_PATH}/src/compiler/tir_to_runtime.cc")
configure_vta_runtime(tvm_vta_ext)
set_target_properties(
  tvm_vta_ext
  PROPERTIES
    OUTPUT_NAME "tvm-vta-ext"
    PREFIX "lib")

if(VTA_BACKEND STREQUAL "fsim" OR VTA_BACKEND STREQUAL "all")
  add_library(
    vta_fsim SHARED
    ${VTA_RUNTIME_SOURCES}
    "${VTA_PATH}/src/sim/sim_driver.cc"
    "${VTA_PATH}/src/sim/sim_tlpp.cc")
  configure_vta_runtime(vta_fsim)
  target_compile_definitions(vta_fsim PRIVATE USE_FSIM_TLPP)
endif()

if(VTA_BACKEND STREQUAL "tsim" OR VTA_BACKEND STREQUAL "all")
  set(VERILATOR_ROOT "" CACHE PATH "Path to the Verilator installation")
  if(NOT VERILATOR_ROOT)
    find_program(VERILATOR_EXECUTABLE NAMES verilator)
    if(NOT VERILATOR_EXECUTABLE)
      message(FATAL_ERROR
              "TSIM requires VERILATOR_ROOT or verilator available in PATH")
    endif()
    execute_process(
      COMMAND "${VERILATOR_EXECUTABLE}" -getenv VERILATOR_ROOT
      RESULT_VARIABLE VERILATOR_ROOT_RESULT
      OUTPUT_VARIABLE VERILATOR_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT VERILATOR_ROOT_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to query VERILATOR_ROOT from ${VERILATOR_EXECUTABLE}")
    endif()
  endif()
  file(REAL_PATH "${VERILATOR_ROOT}" VERILATOR_ROOT)
  set(VERILATOR_INCLUDE_DIR "${VERILATOR_ROOT}/include")
  foreach(required_path
          "${VERILATOR_INCLUDE_DIR}/verilated.cpp"
          "${VERILATOR_INCLUDE_DIR}/verilated_dpi.cpp"
          "${VERILATOR_INCLUDE_DIR}/verilated_threads.cpp"
          "${VERILATOR_INCLUDE_DIR}/vltstd/svdpi.h")
    if(NOT EXISTS "${required_path}")
      message(FATAL_ERROR "Required Verilator dependency was not found: ${required_path}")
    endif()
  endforeach()

  find_package(Threads REQUIRED)
  add_library(
    vta_tsim
    SHARED
    ${VTA_RUNTIME_SOURCES}
    "${VTA_PATH}/src/tsim/tsim_driver.cc"
    "${VTA_PATH}/src/dpi/module.cc"
    "${VERILATOR_INCLUDE_DIR}/verilated_threads.cpp")
  configure_vta_runtime(vta_tsim)
  target_include_directories(
    vta_tsim
    PRIVATE
      "${VERILATOR_INCLUDE_DIR}"
      "${VERILATOR_INCLUDE_DIR}/vltstd")
  target_link_libraries(vta_tsim PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
  if(APPLE)
    target_link_options(
      vta_tsim
      PRIVATE
        "-Wl,-U,__Z15vl_time_stamp64v"
        "-Wl,-U,__Z13sc_time_stampv")
  endif()
endif()
