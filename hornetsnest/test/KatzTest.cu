/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
 *   ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#include "Static/KatzCentrality/Katz.cuh"
#include <StandardAPI.hpp>
#include <Device/Util/Timer.cuh>
#include <Graph/GraphStd.hpp>

template <typename HornetGraph,typename Katz>
int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;
    using namespace hornets_nest;
    using namespace timer;

	// Limit the number of iteartions for graphs with large number of vertices.
    int max_iterations = 20;

	  cudaSetDevice(0);
    GraphStd<vert_t, vert_t> graph(UNDIRECTED);

    HornetInit* hornet_init;

    if(argc>1){
      graph.read(argv[1], SORT | PRINT_INFO);
      hornet_init = new HornetInit(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    }else{

      max_iterations=20;
      const vert_t tempNV = 5;
      const vert_t tempNE = 2*tempNV-2;
      vert_t tempOff[tempNV+1];// = {0,1,3,5,6};
      vert_t tempEdges[tempNE];// = {1,0,2,1,3,2};

      tempOff[0]=0;
      tempOff[1]=1;
      tempOff[tempNV] = tempNE;
      for(vert_t v=2; v<tempNV; v++)
        tempOff[v]=tempOff[v-1]+2;
      tempEdges[0]=1;
      tempEdges[tempNE-1]=tempNV-2;
      vert_t count=1;
      for(vert_t v=1; v<(tempNV-1); v++){
        printf("%d, ",count);
        tempEdges[count++]=v-1;
        tempEdges[count++]=v+1;
      }

      printf("\n");

      hornet_init = new HornetInit(tempNV, tempNE, tempOff, tempEdges);
    }


    // HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

    HornetGraph hornet_graph(*hornet_init);
     // Finding largest vertex degreemake
    degree_t max_degree_vertex = hornet_graph.max_degree();
    std::cout << "Max degree vextex is " << max_degree_vertex << std::endl;


    // Katz kcStatIc(hornet_graph, max_iterations, max_degree_vertex);
    float alpha = 1.0/(max_degree_vertex+1.0);
    Katz kcStatIc(hornet_graph, alpha, max_iterations);


    Timer<DEVICE> TM;
    TM.start();

    kcStatIc.run();

    TM.stop();

    double* h_kcArray = new double[hornet_graph.nV()];

    kcStatIc.copyKCToHost(h_kcArray);

    // for (int v=0; v<hornet_graph.nV(); v++){
    //   printf("%lf, ", h_kcArray[v]);
    // }
    // printf("\n");

    auto total_time = TM.duration();
    std::cout << "The number of iterations     : "
              << kcStatIc.get_iteration_count()
              << "\nTotal time for KC          : " << total_time
              << "\nAverage time per iteartion : "
              << total_time /
                 static_cast<float>(kcStatIc.get_iteration_count())
              << "\n";


    delete[] h_kcArray;
    delete hornet_init;

    return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;
  {
    ret = exec<hornets_nest::HornetStaticGraph,hornets_nest::KatzCentralityStatic>(argc, argv);
  }

  return ret;
}
