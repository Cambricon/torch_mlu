cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(kineto_mlu VERSION 0.1 LANGUAGES CXX C)

set(LIBKINETO_INCLUDE_DIR "$ENV{PYTORCH_WHEEL_DIR}/include/kineto")
set(LIBKINETO_MLU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIBKINETO_MLU_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

include_directories(${LIBKINETO_INCLUDE_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(src)
add_library(kineto_mlu OBJECT ${PROFILER_SRCS})

set_target_properties(kineto_mlu PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO)

set(KINETO_COMPILE_OPTIONS "-DFMT_HEADER_ONLY")
list(APPEND KINETO_COMPILE_OPTIONS "-std=c++17")
list(APPEND KINETO_COMPILE_OPTIONS "-Wno-error=pragmas")
target_compile_options(kineto_mlu PRIVATE "${KINETO_COMPILE_OPTIONS}")

if(NOT TARGET fmt)
  if(NOT FMT_SOURCE_DIR)
    set(FMT_SOURCE_DIR "${LIBKINETO_MLU_THIRDPARTY_DIR}/fmt"
      CACHE STRING "fmt source directory from submodules")
  endif()

  # Build FMT.
  # FMT and some other libraries use BUILD_SHARED_LIBS to control
  # the library type.
  # Save and restore the value after configuring FMT
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
  set(FMT_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  add_subdirectory("${FMT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/fmt")
  set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
endif()

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "Kineto: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
message(STATUS "Kineto: FMT_INCLUDE_DIR = ${FMT_INCLUDE_DIR}")
message(INFO " CNPAPI_INCLUDE_DIR = ${CNPAPI_INCLUDE_DIR}")

include_directories(${FMT_INCLUDE_DIR})
include_directories(${CNPAPI_INCLUDE_DIR})

set_property(TARGET kineto_mlu PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(kineto_mlu "${CNPAPI_LIBRARY}")
target_link_libraries(kineto_mlu "${TORCH_LIBRARIES}")
target_link_libraries(kineto_mlu $<BUILD_INTERFACE:fmt::fmt-header-only>)
add_dependencies(kineto_mlu fmt::fmt-header-only)

option(KINETO_MLU_BUILD_TESTS "Build kineto unit tests" ON)
if(${KINETO_MLU_BUILD_TESTS})
  add_subdirectory(test)
endif()
