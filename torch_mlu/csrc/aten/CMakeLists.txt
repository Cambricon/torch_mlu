cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(ATEN_DEPENDENT_SRCS ${BASIC_SRCS})
file(GLOB_RECURSE ATen_BASIC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewchain/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cnnl/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/cpu/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/cpu/internal/*.cpp
)

file(GLOB_RECURSE ATen_CNNL_OP_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/cnnl/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operators/cnnl/internal/*.cpp
)

list(APPEND ATEN_DEPENDENT_SRCS ${ATen_BASIC_SRCS} ${ATen_CNNL_OP_SRCS})

if(USE_BANG)
	file(GLOB_RECURSE ATen_BANG_OP_SRCS
	    ${CMAKE_CURRENT_SOURCE_DIR}/operators/bang/*.cpp
	)
	list(APPEND ATEN_DEPENDENT_SRCS ${ATen_BANG_OP_SRCS})
endif()

if(USE_MLUOP)
    file(GLOB_RECURSE ATen_MLU_OPS_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/operators/mluop/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mluop/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/operators/mluop/internal/*.cpp
    )
    list(APPEND ATEN_DEPENDENT_SRCS ${ATen_MLU_OPS_SRCS})
endif()

set(BASIC_SRCS ${ATEN_DEPENDENT_SRCS} PARENT_SCOPE)
