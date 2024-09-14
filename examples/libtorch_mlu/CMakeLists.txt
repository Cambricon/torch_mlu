cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(LibtorchMLUDemo CXX C)

set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define build type
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG")
ELSEIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("RelWithDebInfo build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ELSE()
  message("Release build.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
ENDIF()

SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

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
    set(torch_mlu_root "$ENV{TORCH_MLU_HOME}")
else()
    set(torch_mlu_root "path/to/your/torch_mlu")
endif()
if(NOT EXISTS ${torch_mlu_root})
  message(FATAL_ERROR "please set correct torch_mlu_root")
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

include_directories(${torch_mlu_root}/torch_mlu/csrc/include)
include_directories(${torch_mlu_root}/torch_mlu/csrc)
include_directories(${neuware_home}/include)

link_directories(${torch_mlu_root}/torch_mlu/csrc/lib)
link_directories(${neuware_home}/lib64)

if(DEFINED ENV{_GLIBCXX_USE_CXX11_ABI})
	set(GLIBCXX_USE_CXX11_ABI "$ENV{_GLIBCXX_USE_CXX11_ABI}")
else()
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

add_executable(torch_mlu_demo inference_model.cpp)

target_link_libraries(torch_mlu_demo "-Wl,--no-as-needed" torch_mlu "-Wl,--as-needed")
target_link_libraries(torch_mlu_demo pthread)
target_link_libraries(torch_mlu_demo cnrt)
target_link_libraries(torch_mlu_demo torch torch_cpu c10 dl util)

