/**
 * @brief Sparse Matrix-Vector multiplication
 * @file
 */
#include "Static/SpMV/SpMV.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
//#include <cuda_profiler_api.h> //--profile-from-start off
#include <cub/cub.cuh>

int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);
    auto h_vector = new int[graph.nV()];
    auto  h_value = new int[graph.nE()];
    std::fill(h_vector, h_vector + graph.nV(), 1);
    std::fill(h_value, h_value + graph.nE(), 1);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());
    hornet_init.insertEdgeData(h_value);

    HornetGraph hornet_matrix(hornet_init);
    SpMV spmv(hornet_matrix, h_vector);

    Timer<DEVICE> TM;
    TM.start();

    spmv.run();

    TM.stop();
    TM.print("SpMV");

    auto is_correct = spmv.validate();
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");

    TM.start();

    delete[] h_vector;
    delete[] h_value;
    return is_correct;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    ret = exec(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

