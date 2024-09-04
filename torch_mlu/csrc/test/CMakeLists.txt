cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(TEST_INCLUDE_DIRS "")

if(NOT GTEST_SHARED_INSTALL_LIB_DIR)
  set(GTEST_SHARED_INSTALL_LIB_DIR lib)
endif()

message(STATUS "CMAKE_PREFIX_PATH for gtest: " ${CMAKE_PREFIX_PATH})

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
  if(${CMAKE_VERSION} VERSION_LESS 3.12)
    # Deprecated since version 3.12, and Changed in version 3.27
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)
  else()
    find_package(Python3 COMPONENTS Interpreter Development)
    set(PYTHONLIBS_VERSION_STRING ${Python3_VERSION})
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
  endif()

  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 2 has reached end-of-life and is no longer supported by PyTorch.")
  endif()
  if(${PYTHONLIBS_VERSION_STRING} VERSION_LESS 3.7)
    message(FATAL_ERROR
      "Found Python libraries version ${PYTHONLIBS_VERSION_STRING}. Python 3.6 is no longer supported by PyTorch.")
  endif()
endif()

# Build the dynamic library that generates Google test.
set(CMAKE_GOOGLE_TEST_PATH "$ENV{TORCH_MLU_SOURCE_PATH}/third_party/googletest")
list(APPEND TEST_INCLUDE_DIRS ${CMAKE_GOOGLE_TEST_PATH}
                              ${CMAKE_GOOGLE_TEST_PATH}/include)
include_directories(${TEST_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB GTEST_SRC_DIR ${CMAKE_GOOGLE_TEST_PATH}/src/gtest-all.cc
                        ${CMAKE_GOOGLE_TEST_PATH}/src/gtest_main.cc)
add_library(gtest_shared SHARED ${GTEST_SRC_DIR})

link_directories("$ENV{PYTORCH_WHEEL_DIR}/lib")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set(GTEST_SRCS "")
  file(GLOB_RECURSE BASIC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cnnl/*.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp")
list(APPEND GTEST_SRCS ${BASIC_SRCS})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no-as-needed")
foreach(test_src ${GTEST_SRCS})
  get_filename_component(test_name ${test_src} NAME_WE)
  get_filename_component(test_dir ${test_src} DIRECTORY)

  STRING(REGEX REPLACE ".*/(.*)" "\\1" CPP_FOLDER ${test_dir})

  # Set the path of executable file
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/${CPP_FOLDER})
  add_executable(${test_name}  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                               ${test_src})
  target_link_libraries(${test_name} torch_mlu gtest_shared
                        torch_cpu c10 torch torch_python pthread ${DEPENDENT_LIBRARIES} ${PYTHON_LIBRARIES} dl util)
endforeach()

install(TARGETS gtest_shared DESTINATION "${GTEST_SHARED_INSTALL_LIB_DIR}")
