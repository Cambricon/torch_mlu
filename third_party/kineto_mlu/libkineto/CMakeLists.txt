cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# function to extract filelists from libkineto_defs.bzl file
find_package(PythonInterp)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('libkineto_defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

project(kineto_mlu VERSION 0.1 LANGUAGES CXX C)

set(KINETO_LIBRARY_TYPE "shared" CACHE STRING
  "Type of library (shared) to build")
set_property(CACHE KINETO_LIBRARY_TYPE PROPERTY STRINGS shared)
option(KINETO_MLU_BUILD_TESTS "Build kineto unit tests" ON)

set(LIBKINETO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIBKINETO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIBKINETO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBKINETO_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(LIBPYTORCH_DIR "$ENV{PYTORCH_WHEEL_DIR}/include")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Set LIBKINETO_NOCNPAPI to explicitly disable CNPAPI
# Otherwise, CNPAPI is disabled if not found
IF (NOT CNPAPI_INCLUDE_DIR OR NOT CNPAPI_LIBRARY)
    set(LIBKINETO_NOCNPAPI ON CACHE BOOL "" FORCE)
endif()

# Define file lists
if (LIBKINETO_NOCNPAPI)
    get_filelist("get_libkineto_cpu_only_srcs(with_api=False)" LIBKINETO_SRCS)
    message(INFO " CNPAPI unavailable or disabled - not building MLU profilers")
else()
    get_filelist("get_libkineto_cnpapi_srcs(with_api=False)" LIBKINETO_SRCS)
endif()
get_filelist("get_libkineto_public_headers()" LIBKINETO_PUBLIC_HEADERS)
get_filelist("get_libkineto_api_srcs()" LIBKINETO_API_SRCS)

add_library(kineto_base OBJECT ${LIBKINETO_SRCS})
add_library(kineto_api OBJECT ${LIBKINETO_API_SRCS})

# Make libraries depend on libkineto_defs.bzl
add_custom_target(libkineto_defs.bzl DEPENDS libkineto_defs.bzl)
add_dependencies(kineto_base libkineto_defs.bzl)

set_target_properties(kineto_base kineto_api PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      CXX_VISIBILITY_PRESET hidden)

set(KINETO_COMPILE_OPTIONS "-DKINETO_NAMESPACE=libkineto_mlu")
list(APPEND KINETO_COMPILE_OPTIONS "-DFMT_HEADER_ONLY")
if(NOT MSVC)
  list(APPEND KINETO_COMPILE_OPTIONS "-std=c++17")
  list(APPEND KINETO_COMPILE_OPTIONS "-Wno-error=pragmas")
else()
  list(APPEND KINETO_COMPILE_OPTIONS "/std:c++17")
  list(APPEND KINETO_COMPILE_OPTIONS "-DWIN32_LEAN_AND_MEAN")
  list(APPEND KINETO_COMPILE_OPTIONS "-DNOGDI")
endif()
if (NOT LIBKINETO_NOCNPAPI)
    list(APPEND KINETO_COMPILE_OPTIONS "-DHAS_CNPAPI")
endif()

target_compile_options(kineto_base PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_compile_options(kineto_api PRIVATE "${KINETO_COMPILE_OPTIONS}")

if(NOT TARGET fmt)
  if(NOT FMT_SOURCE_DIR)
    set(FMT_SOURCE_DIR "${LIBKINETO_THIRDPARTY_DIR}/fmt"
      CACHE STRING "fmt source directory from submodules")
  endif()

  # Build FMT.
  # FMT and some other libraries use BUILD_SHARED_LIBS to control
  # the library type.
  # Save and restore the value after configuring FMT
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
  set(FMT_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  add_subdirectory("${FMT_SOURCE_DIR}" "${LIBKINETO_BINARY_DIR}/fmt")
  set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
endif()

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "Kineto: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
message(STATUS "Kineto: FMT_INCLUDE_DIR = ${FMT_INCLUDE_DIR}")

message(INFO " CNPAPI_INCLUDE_DIR = ${CNPAPI_INCLUDE_DIR}")

target_include_directories(kineto_base PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CNPAPI_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBPYTORCH_DIR}>)

target_include_directories(kineto_api PUBLIC
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBPYTORCH_DIR}>)

if(KINETO_LIBRARY_TYPE STREQUAL "default")
  add_library(kineto_mlu
    $<TARGET_OBJECTS:kineto_base>
    $<TARGET_OBJECTS:kineto_api>)
elseif(KINETO_LIBRARY_TYPE STREQUAL "static")
  add_library(kineto_mlu STATIC
    $<TARGET_OBJECTS:kineto_base>
    $<TARGET_OBJECTS:kineto_api>)
elseif(KINETO_LIBRARY_TYPE STREQUAL "shared")
  add_library(kineto_mlu SHARED
    $<TARGET_OBJECTS:kineto_base>)
  set_property(TARGET kineto_base PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(kineto_mlu PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
else()
  message(FATAL_ERROR "Unsupported library type ${KINETO_LIBRARY_TYPE}")
endif()

if(NOT LIBKINETO_NOCNPAPI)
    target_link_libraries(kineto_mlu "${CNPAPI_LIBRARY}")
endif()
target_link_libraries(kineto_mlu $<BUILD_INTERFACE:fmt::fmt-header-only>)
add_dependencies(kineto_mlu fmt::fmt-header-only)

set(KINETO_LIBRARY "$ENV{PYTORCH_WHEEL_DIR}/lib")
target_link_libraries(kineto_mlu "${KINETO_LIBRARY}/libkineto.so")
install(TARGETS kineto_mlu DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../../../torch_mlu/csrc/lib")

if(${KINETO_MLU_BUILD_TESTS})
  add_subdirectory(test)
endif()
