cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_resnet)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../libtorch_resnet/)

find_package(Torch REQUIRED)
#find_package(Torch_mlu REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# set the real path
if(DEFINED ENV{Torch_DIR})
    set(torch_path "$ENV{Torch_DIR}")
else()
    set(torch_path "path/to/your/pytorch/torch")
endif()
if(NOT EXISTS ${torch_path})
  message(FATAL_ERROR "please set correct torch_path")
endif()
include_directories(${torch_path}/include)
link_directories(${torch_path}/lib)

# set the real path
if(DEFINED ENV{TORCH_MLU_HOME})
    set(torch_mlu_path "$ENV{TORCH_MLU_HOME}")
else()
    set(torch_mlu_path "path/to/your/torch_mlu")
endif()
if(NOT EXISTS ${torch_mlu_path})
  message(FATAL_ERROR "please set correct torch_mlu_path")
endif()

# set the real path
if(DEFINED ENV{NEUWARE_HOME})
    set(neuware_home "$ENV{NEUWARE_HOME}")
else()
    set(neuware_home "path/to/your/neuware_home")
endif()
if(NOT EXISTS ${neuware_home})
  message(FATAL_ERROR "please set correct neuware_home")
endif()

include_directories(${torch_mlu_path}/csrc)
include_directories(${neuware_home}/include)

link_directories(${torch_mlu_path}/csrc/lib)
link_directories(${neuware_home}/lib64)

if(DEFINED ENV{_GLIBCXX_USE_CXX11_ABI})
	set(GLIBCXX_USE_CXX11_ABI "$ENV{_GLIBCXX_USE_CXX11_ABI}")
else()
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
add_executable(libtorch_resnet libtorch_resnet.cpp)
target_link_libraries(libtorch_resnet "${TORCH_LIBRARIES}")

target_link_libraries(libtorch_resnet "-Wl,--no-as-needed" torch_mlu "-Wl,--as-needed")

target_link_libraries(libtorch_resnet cnrt)
target_link_libraries(libtorch_resnet torch torch_cpu c10 dl util)

set_property(TARGET libtorch_resnet PROPERTY CXX_STANDARD 17)
