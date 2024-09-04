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
    CnpapiStringsTest.cpp
    CnpapiActivityProfilerTest.cpp
)

# TODO(fuwenguang): remove this once kineto delete TMP_LIBKINETO_NANOSECOND
add_definitions(-DTMP_LIBKINETO_NANOSECOND)

foreach(TEST_SOURCE ${TEST_SOURCES})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kineto_gtest)
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_compile_options(${TEST_NAME} PRIVATE "${KINETO_COMPILE_OPTIONS}")
    target_link_libraries(${TEST_NAME} kineto_gtest_shared kineto_gmock_shared
                                       kineto_mlu "${CNRT_LIBRARIES}" "${TORCH_LIBRARIES}")
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
