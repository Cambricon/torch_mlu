# FindTorchMLU
# -------
#
# Finds the Torch MLU library
#
#  TORCH_MLU_LIBRARIES
#  TORCH_MLU_INCLUDE_DIRS
#  TORCH_MLU_LIBRARY_DIRS
#

include(FindPackageHandleStandardArgs)

get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules $ENV{NEUWARE_HOME}/cmake/modules)

set(TORCH_MLU_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../../csrc)
set(TORCH_MLU_LIBRARY_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../../csrc/lib)
set(TORCH_MLU_LIBRARIES "")
list(APPEND TORCH_MLU_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../../csrc/api/include/)

# torch_mlu
find_package(CNRT)
if (CNRT_FOUND)
    list(APPEND TORCH_MLU_INCLUDE_DIRS ${CNRT_INCLUDE_DIRS})
    list(APPEND TORCH_MLU_LIBRARIES ${CNRT_LIBRARIES})
endif()

# Find cnnl header files and libs
find_package(CNNL)
if (CNNL_FOUND)
    list(APPEND TORCH_MLU_INCLUDE_DIRS ${CNNL_INCLUDE_DIRS})
    list(APPEND TORCH_MLU_LIBRARIES ${CNNL_LIBRARIES})
endif()

# Find bangc files and libs
find_package(BANG)
if (BANG_FOUND)
    set(BANG_CNCC_FLAGS "-Wall -Werror -fPIC -std=c++11 -pthread")
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3 -DNDEBUG")
    if ("${CMAKE_C_COMPILER}" MATCHES ".*aarch64-linux-gnu-.*" AND "${CMAKE_CXX_COMPILER}" MATCHES ".*aarch64-linux-gnu-.*")
      set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --target=aarch64-linux-gnu")
      set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_220")

      include_directories($ENV{CROSS_TOOLCHAIN_PATH}/../aarch64-linux-gnu/libc/usr/include)
      include_directories($ENV{CROSS_TOOLCHAIN_PATH}/../aarch64-linux-gnu/include/c++/6.2.1)
      include_directories($ENV{CROSS_TOOLCHAIN_PATH}/../aarch64-linux-gnu/include/c++/6.2.1/aarch64-linux-gnu)
    else()
      set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_220"
                                               "--bang-mlu-arch=mtp_270"
                                               "--bang-mlu-arch=mtp_372"
                                               "--bang-mlu-arch=mtp_290")
    endif()
    set(BANG_CNCC_EXECUTABLE "$ENV{NEUWARE_HOME}/bin/cncc")
endif()

# Find cndrv header files and libs
find_package(CNDRV)
if (CNDRV_FOUND)
    list(APPEND TORCH_MLU_INCLUDE_DIRS ${CNDRV_INCLUDE_DIRS})
    list(APPEND TORCH_MLU_LIBRARIES ${CNDRV_LIBRARIES})
endif()

find_library(TORCH_ATEN_LIBRARY aten_mlu PATHS "${TORCH_MLU_LIBRARY_DIRS}")
find_library(TORCH_MLU_LIBRARY torch_mlu PATHS "${TORCH_MLU_LIBRARY_DIRS}")
find_package_handle_standard_args(TorchMLU DEFAULT_MSG TORCH_MLU_LIBRARY)
list(APPEND TORCH_MLU_LIBRARIES ${TORCH_MLU_LIBRARY} ${TORCH_ATEN_LIBRARY})

include_directories(${TORCH_MLU_INCLUDE_DIRS})
include_directories($ENV{NEUWARE_HOME}/lib/clang/*/include)
link_directories(${TORCH_MLU_LIBRARY_DIRS})
link_directories($ENV{NEUWARE_HOME}/lib)
