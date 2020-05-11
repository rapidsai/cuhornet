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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")

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

find_path(RMM_INCLUDE "rmm"
    HINTS
    "$ENV{RMM_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/rmm"
    "$ENV{CONDA_PREFIX}/include")

find_library(RMM_LIBRARY "rmm"
    HINTS
    "$ENV{RMM_ROOT}/lib"
    "$ENV{CONDA_PREFIX}/lib")

message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

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

