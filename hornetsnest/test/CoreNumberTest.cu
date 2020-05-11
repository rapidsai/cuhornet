
/**
 * @brief CoreNumber test program
 * @file
 */

#include "Static/CoreNumber/CoreNumber.cuh"
#include <Device/Util/Timer.cuh>
#include <Graph/GraphStd.hpp>

using namespace timer;
using namespace hornets_nest;

int exec(int argc, char **argv) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vert_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
    thrust::device_vector<int> core_number(graph.nV());
    CoreNumberStatic kcore(hornet_graph, core_number.data().get());
    kcore.run();

    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    //ret = exec<hornets_nest::HornetDynamicGraph, hornets_nest::BfsTopDown2Dynamic>(argc, argv);
    ret = exec(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}
