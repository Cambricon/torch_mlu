cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(torch_mlu CXX C)
set(CMAKE_CXX_STANDARD 17)
set(_rpath_portable_origin $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_DEPENDENT_DIR ${PROJECT_SOURCE_DIR}/../../cmake/)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-as-needed")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  else()
    # Please note this is required in order to ensure compatibility between gcc 9 and gcc 7
    # This could be removed when all Linux PyTorch binary builds are compiled by the same toolchain again
    string(APPEND CMAKE_CXX_FLAGS " -fabi-version=11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
endif()

# Include common functions
include(${CMAKE_DEPENDENT_DIR}/modules/Utils.cmake)

# Set the cmake path of Modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/modules
    ${NEUWARE_HOME}/cmake/modules)

message(STATUS "the neuware home path is ${NEUWARE_HOME}")
set(DEPENDENT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
set(DEPENDENT_LIBRARY_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib)
list(APPEND DEPENDENT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/api/include)
set(DEPENDENT_LIBRARIES "")

# Init compile definitions for torch_mlu_python
set(TORCH_MLU_PYTHON_COMPILE_DEFINITIONS)

if (DEFINED ENV{TEST_COVERAGE})
  list(APPEND TORCH_MLU_PYTHON_COMPILE_DEFINITIONS TEST_COVERAGE)
endif()

# Set the include dir of neuware
include_directories(${NEUWARE_HOME}/include)
include_directories(${NEUWARE_HOME}/lib/clang/*/include)
include_directories(${NEUWARE_HOME}/lib/clang/*/include)
link_directories(${NEUWARE_HOME}/lib)

# Set the llvm libs
if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
  link_directories(${NEUWARE_HOME}/lib/llvm-mm/lib)
else()
  link_directories(${NEUWARE_HOME}/lib/llvm-mm-cxx11-old-abi/lib)
endif()

find_package(pybind11)
if(pybind11_FOUND)
    message(STATUS "System pybind11 found")
else()
    message(STATUS "Using third_party/pybind11.")
    set(pybind11_INCLUDE_DIRS ${PYTORCH_WHEEL_DIR}/include)
    # install(DIRECTORY ${pybind11_INCLUDE_DIRS}
    #         DESTINATION ${CMAKE_INSTALL_PREFIX}
    #         FILES_MATCHING PATTERN "*.h")
endif()
message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
list(APPEND DEPENDENT_INCLUDE_DIRS ${pybind11_INCLUDE_DIRS})

# Set bangc compile flags
if (USE_BANG)
    # Find bang files and libs
    find_package(BANG)
    if(NOT BANG_FOUND)
    	message(FATAL_ERROR "BANG cannot be found !!!!!")
    endif()
	# cncc gflags
    set(BANG_CNCC_FLAGS "-Wall -Werror -fPIC -std=c++11 -pthread")

    set(cmake_build_type "")
    string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type)
    if(${cmake_build_type} MATCHES "debug")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
    else()
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3 -DNDEBUG")
    endif()
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" " --bang-mlu-arch=mtp_372"
                                             " --bang-mlu-arch=mtp_592")
endif()

# Set cndev header files and libs
find_package(CNDEV)
if(CNDEV_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNDEV_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNDEV_LIBRARIES})
endif()

# Set cndrv header files and libs
find_package(CNDRV)
if(CNDRV_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNDRV_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNDRV_LIBRARIES})
endif()

# Find cnrt header files and libs
find_package(CNRT)
if (CNRT_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNRT_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNRT_LIBRARIES})
endif()


# Find cnnl header files and libs
find_package(CNNL)
if (CNNL_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNNL_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNNL_LIBRARIES})
endif()

# Find mlu_op header files and libs
find_package(MLUOP)
if (USE_MLUOP AND MLUOP_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MLUOP_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MLUOP_LIBRARIES})
    add_definitions("-DUSE_MLUOP")
endif()

# Find cncl header files and libs
find_package(CNCL)
if (USE_CNCL AND CNCL_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNCL_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNCL_LIBRARIES})
endif()

# Find CNPAPI header files and libs
find_package(CNPAPI)
message(STATUS " DEPENDENT_INCLUDE_DIRS " ${DEPENDENT_INCLUDE_DIRS})
message(STATUS " CNPAPI_INCLUDE_DIRS " ${CNPAPI_INCLUDE_DIRS})
if (USE_PROFILE AND CNPAPI_FOUND)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/kineto_mlu/libkineto/include)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNPAPI_INCLUDE_DIRS})
    add_definitions("-DUSE_PROFILE")
endif()

# Find cnnl_extra header files and libs
find_package(CNNLExtra)
if (CNNLExtra_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${CNNLExtra_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${CNNLExtra_LIBRARIES})
endif()

find_package(Torch REQUIRED)
if (NOT TORCH_FOUND)
    message(FATAL_ERROR "Error! Could not find Torch")
endif()
list(APPEND DEPENDENT_LIBRARIES ${TORCH_LIBRARIES})

if(USE_PYTHON)
  # Check that Python works
  set(PYTHON_VERSION)
  execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" "--version"
      RESULT_VARIABLE _exitcode OUTPUT_VARIABLE PYTHON_VERSION)
  if(NOT _exitcode EQUAL 0)
    message(FATAL_ERROR "The Python executable ${PYTHON_EXECUTABLE} cannot be run. Make sure that it is an absolute path.")
  endif()
  if(PYTHON_VERSION)
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)" PYTHON_VERSION ${PYTHON_VERSION})
  endif()

  # These should fill in the rest of the variables, like versions, but resepct
  # the variables we have set like PYTHON_EXECUTABLE, PYTHON_INCLUDE_DIR and
  # PYTHON_LIARARY in setup.py
  set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION} 3.8 3.7)
  if(${CMAKE_VERSION} VERSION_LESS 3.27)
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)
  else()
    find_package(Python3 COMPONENTS Interpreter Development)
    set(PYTHONLIBS_VERSION_STRING ${Python3_VERSION})
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
  endif()

  message(STATUS "${PYTHONLIBS_VERSION_STRING}: " ${Python3_VERSION})
  message(STATUS "cmake version: " ${CMAKE_VERSION})

  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 2 has reached end-of-life and is no longer supported by PyTorch.")
  endif()
  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3.7)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 3.6 is no longer supported by PyTorch.")
  endif()
  list(APPEND DEPENDENT_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
endif()

# Set the header files' path of Pytorch
list(APPEND DEPENDENT_INCLUDE_DIRS ${PYTORCH_WHEEL_DIR}/include
	                           ${PYTORCH_WHEEL_DIR}/include/torch/csrc/api/include)

# Set include and link directories
include_directories(${DEPENDENT_INCLUDE_DIRS})
link_directories(${DEPENDENT_LIBRARY_DIRS})

# Set the basic source files
set(BASIC_SRCS "")

# Set Aten Sources
add_subdirectory(aten)

# Set Framework Sources
add_subdirectory(framework)

# Build Bangc Kernel
if (USE_BANG)
    # Set the header files' path of bangc kernel
    # Build bangc op
    file(GLOB_RECURSE bang_op_files FOLLOW_SYMLINKS "${CMAKE_CURRENT_SOURCE_DIR}/aten/operators/bang/internal/*.mlu")
    bang_add_library(bangc SHARED "${bang_op_files}")
endif()

# Compile torch_mlu_python Library
if(NOT TORCH_MLU_INSTALL_LIB_DIR)
    set(TORCH_MLU_INSTALL_LIB_DIR lib)
endif()

# Compile torch_mlu Library
file(GLOB_RECURSE API_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/api/*.cpp)
file(GLOB_RECURSE UTIL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)

list(APPEND BASIC_SRCS ${API_SRCS} ${UTIL_SRCS})
add_library(torch_mlu SHARED ${BASIC_SRCS})

# Checks if compiler supports -fvisibility=hidden and -fvisibility-inlines-hidden
check_cxx_compiler_flag("-fvisibility=hidden" COMPILER_SUPPORTS_HIDDEN_VISIBILITY)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY)

# Enable hidden visibility if compiler supports it.
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(torch_mlu  PRIVATE -fvisibility=hidden)
endif()

# Enable hidden visibility of inline function if compiler support it.
if(${COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY})
  target_compile_options(torch_mlu  PRIVATE -fvisibility-inlines-hidden)
endif()

target_link_libraries(torch_mlu ${DEPENDENT_LIBRARIES})

if(USE_BANG)
    target_link_libraries(torch_mlu ${DEPENDENT_LIBRARIES} bangc)
    install(TARGETS bangc DESTINATION "${TORCH_MLU_INSTALL_LIB_DIR}")
endif()

install(TARGETS torch_mlu DESTINATION "${TORCH_MLU_INSTALL_LIB_DIR}")

# Compile test
if (BUILD_TEST)
    message("---- Build Torch MLU Test.")
    add_subdirectory(test)
endif()

set(DEPENDENT_SRCS "")
file(GLOB_RECURSE DEPENDENT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/python/*.cpp)

if (USE_PYTHON)
    # Enable hidden visibility of inline function if compiler support it.
    add_library(torch_mlu_python SHARED ${DEPENDENT_SRCS})
    if(${COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY})
      target_compile_options(torch_mlu_python PRIVATE -fvisibility-inlines-hidden)
    endif()
    target_link_libraries(torch_mlu_python torch_mlu ${DEPENDENT_LIBRARIES})
    if(USE_MLUOP AND MLUOP_FOUND)
        target_link_libraries(torch_mlu_python ${MLUOP_LIBRARIES})
    endif()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/python/ProcessGroupCNCL.cpp PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")

    if(USE_PROFILE AND CNPAPI_FOUND)
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/kineto_mlu/libkineto" kineto_mlu.out)
        target_link_libraries(torch_mlu_python kineto_mlu)
    endif()

    target_compile_definitions(torch_mlu_python PRIVATE ${TORCH_MLU_PYTHON_COMPILE_DEFINITIONS})

    install(TARGETS torch_mlu_python DESTINATION "${TORCH_MLU_INSTALL_LIB_DIR}")
endif()

include(${CMAKE_DEPENDENT_DIR}/Summary.cmake)
torch_mlu_print_configuration_summary()
