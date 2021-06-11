/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

template <typename HornetGraph, typename BFS>
int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;


    // graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED );
    graph::GraphStd<vid_t, eoff_t> graph(DIRECTED );
    CommandLineParam cmd(graph, argc, argv,false);
    // ParsingProp pp;
    // graph.read(argv[1],pp);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    Timer<DEVICE> TM;
    HornetGraph hornet_graph(hornet_init);

    BFS bfs_top_down(hornet_graph);

    vid_t root = graph.max_out_degree_id();
    // if (argc==3)
    //     root = atoi(argv[2]);
    int numberRoots = 1;
    if (argc>=3)
      numberRoots = atoi(argv[2]);

    int alg = 0;
    if (argc>=4)
      alg = atoi(argv[3]);

    std::cout << "My root is " << root << std::endl;


    cudaProfilerStart();
    for(int i=0; i<numberRoots; i++){
        bfs_top_down.reset();
        bfs_top_down.set_parameters((root+i)%graph.nV(),alg);
    TM.start();
        bfs_top_down.run();
    TM.stop();
        std::cout << "Number of levels is : " << bfs_top_down.getLevels() << std::endl;
    }

    cudaProfilerStop();
    TM.print("TopDown2");

    return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;
  {
    ret = exec<hornets_nest::HornetStaticGraph,  hornets_nest::BfsTopDown2Static >(argc, argv);
  }

  return ret;
}
