
/**
 * @brief CoreNumber test program
 * @file
 */

#include "Static/CoreNumber/CoreNumber.cuh"
#include <Device/Util/Timer.cuh>
#include <Graph/GraphStd.hpp>

#include <thrust/device_vector.h>

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
  {

    ret = exec(argc, argv);

  }

  return ret;
}
