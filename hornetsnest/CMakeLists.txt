cmake_minimum_required(VERSION 3.8) #language level CUDA support was added in CMake 3.8

set(PROJECT "HornetAlg")
project(${PROJECT} LANGUAGES CXX CUDA)

include(../compiler-util/CMakeLists.txt)

#add_subdirectory(../hornet build)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(../hornet/include)
include_directories(include "${RMM_INCLUDE}")
include_directories(../primitives)
include_directories(${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

file(GLOB_RECURSE DUMMY         ${PROJECT_SOURCE_DIR}/src/Static/Dummy/Dummy.cu)
#file(GLOB_RECURSE BC_SRCS       ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/bc.cu)
#file(GLOB_RECURSE BC_SRCS2      ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/approximate_bc.cu)
#file(GLOB_RECURSE BC_SRCS3      ${PROJECT_SOURCE_DIR}/src/Static/BetweennessCentrality/exact_bc.cu)
#file(GLOB_RECURSE BUBFS_SRC     ${PROJECT_SOURCE_DIR}/src/Static/BottomUpBreadthFirstSearch/BottomUpBFS.cu)
#file(GLOB_RECURSE CC_SRCS       ${PROJECT_SOURCE_DIR}/src/Static/ConnectedComponents/CC.cu)
#file(GLOB_RECURSE CLCOEFF_SRCS  ${PROJECT_SOURCE_DIR}/src/Static/ClusteringCoefficient/cc.cu)
#file(GLOB_RECURSE SSSP_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/ShortestPath/SSSP.cu)
file(GLOB_RECURSE SPMV_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/SpMV/SpMV.cu)
#file(GLOB_RECURSE PR_SRCS       ${PROJECT_SOURCE_DIR}/src/Static/PageRank/PageRank.cu)
file(GLOB_RECURSE TRI2_SRCS     ${PROJECT_SOURCE_DIR}/src/Static/TriangleCounting/triangle2.cu)
file(GLOB_RECURSE X_SRCS        ${PROJECT_SOURCE_DIR}/../xlib/src/*)
file(GLOB_RECURSE H_SRCS        ${PROJECT_SOURCE_DIR}/../hornet/src/*)

#add_library(hornetAlg ${X_SRCS} ${H_SRCS} ${DUMMY} ${BFS_SRCS} ${BC_SRCS} ${BC_SRCS2} ${BC_SRCS3} ${BUBFS_SRC} ${CC_SRCS} ${CLCOEFF_SRCS} ${SSSP_SRCS} ${SPMV_SRCS} ${PR_SRCS} ${KCORE_SRCS} ${TRI2_SRCS})
add_library(hornetAlg ${X_SRCS} ${H_SRCS} ${DUMMY} ${SPMV_SRCS} ${TRI2_SRCS})

target_link_libraries(hornetAlg ${RMM_LIBRARY})

###################################################################################################
# - add executables -------------------------------------------------------------------------------

add_executable(dummy        test/DummyTest.cu)
#add_executable(bfs2         test/BFSTest2.cu)
#add_executable(bc	        test/BCTest.cu)
#add_executable(bubfs        test/BUBFSTest2.cu)
#add_executable(con-comp     test/CCTest.cu)
add_executable(core_number  test/CoreNumberTest.cu)
add_executable(spmv         test/SpMVTest.cu)
#add_executable(sssp         test/SSSPTest.cu)
add_executable(katz         test/KatzTest.cu)
add_executable(katzApprox   test/KatzTopKTest.cu)
add_executable(ktruss       test/KTrussTest.cu)
add_executable(triangle2    test/TriangleTest2.cu)
#add_executable(clus-coeff   test/ClusCoeffTest.cu)
#add_executable(pr           test/PageRankTest.cu)


target_link_libraries(dummy         hornetAlg)
#target_link_libraries(bfs2          hornetAlg)
#target_link_libraries(bc            hornetAlg)
#target_link_libraries(bubfs         hornetAlg)
#target_link_libraries(con-comp      hornetAlg)
target_link_libraries(core_number   hornetAlg)
target_link_libraries(spmv          hornetAlg)
#target_link_libraries(sssp          hornetAlg)
target_link_libraries(katz          hornetAlg)
target_link_libraries(katzApprox    hornetAlg)
target_link_libraries(ktruss        hornetAlg)
target_link_libraries(triangle2     hornetAlg)
#target_link_libraries(clus-coeff    hornetAlg)
#target_link_libraries(pr            hornetAlg)

