cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(kineto_mlu_gtest)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/")
set(LIBKINETO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

include_directories(${LIBKINETO_DIR})
include_directories(${THIRD_PARTY_DIR}/fmt/include)

# googletest
set(CMAKE_GOOGLE_TEST_PATH "${THIRD_PARTY_DIR}/googletest/googletest")
list(APPEND GOOGLE_TEST_INCLUDE_DIRS ${CMAKE_GOOGLE_TEST_PATH}
                                     ${CMAKE_GOOGLE_TEST_PATH}/include)
include_directories("${GOOGLE_TEST_INCLUDE_DIRS}")
file(GLOB GTEST_SRC_DIR ${CMAKE_GOOGLE_TEST_PATH}/src/gtest-all.cc
                        ${CMAKE_GOOGLE_TEST_PATH}/src/gtest_main.cc)
add_library(kineto_gtest_shared SHARED ${GTEST_SRC_DIR})

# googlemock
set(CMAKE_GOOGLE_MOCK_PATH "${THIRD_PARTY_DIR}/googletest/googlemock")
list(APPEND GOOGLE_MOCK_INCLUDE_DIRS ${CMAKE_GOOGLE_MOCK_PATH}
                                     ${CMAKE_GOOGLE_MOCK_PATH}/include)
include_directories("${GOOGLE_MOCK_INCLUDE_DIRS}")
file(GLOB GMOCK_SRC_DIR ${CMAKE_GOOGLE_MOCK_PATH}/src/gmock-all.cc
                        ${CMAKE_GOOGLE_MOCK_PATH}/src/gmock_main.cc)
add_library(kineto_gmock_shared SHARED ${GMOCK_SRC_DIR})

set(TEST_SOURCES
    ConfigTest.cpp
    PidInfoTest.cpp
    CnperfApiTypeTest.cpp
    LoggerObserverTest.cpp
)

# kineto_test_mlu
foreach(src_file ${LIBKINETO_SRCS})
    list(APPEND LIBKINETO_SRCS_TEST "../${src_file}")
endforeach()
foreach(src_file ${LIBKINETO_API_SRCS})
    list(APPEND LIBKINETO_API_SRCS_TEST "../${src_file}")
endforeach()

add_library(kineto_test_base OBJECT ${LIBKINETO_SRCS_TEST})
add_library(kineto_test_api OBJECT ${LIBKINETO_API_SRCS_TEST})

set_target_properties(kineto_test_base kineto_test_api PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO)

target_compile_options(kineto_test_base PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_compile_options(kineto_test_api PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_include_directories(kineto_test_base PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CNPERF_API_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBPYTORCH_DIR}>)
target_include_directories(kineto_test_api PUBLIC
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBPYTORCH_DIR}>)

add_library(kineto_test_mlu SHARED
    $<TARGET_OBJECTS:kineto_test_base>
    $<TARGET_OBJECTS:kineto_test_api>)
target_link_libraries(kineto_test_mlu "${CNPERF_API_LIBRARY}")
target_link_libraries(kineto_test_mlu $<BUILD_INTERFACE:fmt::fmt-header-only>)
add_dependencies(kineto_test_mlu fmt::fmt-header-only)
target_link_libraries(kineto_test_mlu "${KINETO_LIBRARY}/libkineto.so")

foreach(TEST_SOURCE ${TEST_SOURCES})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kineto_gtest)
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_compile_options(${TEST_NAME} PRIVATE "${KINETO_COMPILE_OPTIONS}")
    target_link_libraries(${TEST_NAME} kineto_gtest_shared kineto_gmock_shared
        kineto_test_mlu "${CNRT_LIBRARIES}" stdc++fs)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
