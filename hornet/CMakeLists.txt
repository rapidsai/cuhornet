cmake_minimum_required(VERSION 3.8) #language level CUDA support was added in CMake 3.8

set(PROJECT "Hornet")
project(${PROJECT} LANGUAGES CXX CUDA)

include(../compiler-util/CMakeLists.txt)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(include "${RMM_INCLUDE}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE SCPP_SRCS ${PROJECT_SOURCE_DIR}/../xlib/src/*.cpp)
file(GLOB_RECURSE SCU_SRCS  ${PROJECT_SOURCE_DIR}/../xlib/src/*.cu)

add_library(hornet ${CPP_SRCS} ${CU_SRCS} ${SCPP_SRCS} ${SCU_SRCS})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(hornet ${RMM_LIBRARY})
if(USE_NVTX)
target_link_libraries(hornet ${NVTX_LIBRARIES})
endif()

###################################################################################################
# - add executables -------------------------------------------------------------------------------

add_executable(hornet_mgpu_insert_test            test/HornetMultiGPUInsertTest.cu)
add_executable(hornet_insert_weighted_test        test/HornetInsertTestWeighted.cu)
add_executable(hornet_insert_test                 test/HornetInsertTest.cu)
add_executable(hornet_delete_test                 test/HornetDeleteTest.cu)

target_link_libraries(hornet_mgpu_insert_test           hornet)
target_link_libraries(hornet_insert_weighted_test       hornet)
target_link_libraries(hornet_insert_test                hornet)
target_link_libraries(hornet_delete_test                hornet)

