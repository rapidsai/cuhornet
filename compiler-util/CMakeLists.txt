###################################################################################################
# - set build options -----------------------------------------------------------------------------

set(CUB_LIBRARY ON) # if set to off, some test cases using CUB will not compile.

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # no other RAPIDS projects use this

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # if on, save output of compile commands to compile_command.json

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0") # for compatibility with Arrow
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0") # for compatibility with Arrow
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0") # for compatibility with Arrow

find_package(CUDA)

# Check for aarch64 vs workstation architectures
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "CMAKE Detected aarch64 CPU architecture, selecting appropriate gencodes")
  # This is being build for Linux4Tegra or SBSA ARM64 CUDA
  set(GPU_ARCHS "62") # Default minimum CUDA GenCode for cuML embedded
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 9)
    set(GPU_ARCHS "${GPU_ARCHS};72")
  endif()
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
    # This is probably for SBSA CUDA, or a next gen Jetson
    set(GPU_ARCHS "${GPU_ARCHS};75;80")
  endif()

else()
  message(STATUS "CMAKE selecting appropriate gencodes for x86 or ppc64 CPU architectures")
  # System architecture was not aarch64,
  # this is datacenter or workstation class hardware
  set(GPU_ARCHS "60") # Default minimum supported CUDA gencode for cuML
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 9)
    set(GPU_ARCHS "${GPU_ARCHS};70")
  endif()
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 10)
    set(GPU_ARCHS "${GPU_ARCHS};75")
  endif()
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
    set(GPU_ARCHS "${GPU_ARCHS};80")
  endif()

endif()

message("-- Building for GPU_ARCHS = ${GPU_ARCHS}")
foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# set warnings as errors
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wno-reorder -Wno-error=deprecated-declarations")
endif(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror,-Wno-reorder,-Wno-error=deprecated-declarations")

# set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CUB_LIBRARY)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCUB_WRAPPER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUB_WRAPPER")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUB_WRAPPER")
endif()

###################################################################################################
# - Find and add different modules and supporting repos -------------------------------------------

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories(../primitives)
include_directories(../xlib/include)

if(CUB_LIBRARY)
    include_directories(../externals/cub-1.8.0)
endif()

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

# As of RAPIDS 0.16 RMM is header only
find_path(RMM_INCLUDE "rmm"
    HINTS
    "$ENV{RMM_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/rmm"
    "$ENV{CONDA_PREFIX}/include")

message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}") # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc

###################################################################################################
# - library targets -------------------------------------------------------------------------------

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --define-macro USE_NVTX")
endif(USE_NVTX)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

###################################################################################################
# - custom targets --------------------------------------------------------------------------------

