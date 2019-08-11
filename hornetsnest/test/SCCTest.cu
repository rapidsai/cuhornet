/*
   * Copyright (c) 2019, NVIDIA CORPORATION.
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *     http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */

// Author: Prasun Gera pgera@nvidia.com

#include "Static/StrongCC/scc.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <set>

using namespace std;
using namespace graph;
using namespace graph::structure_prop;
using namespace graph::parsing_prop;

int exec(int argc, char* argv[]) {
  using namespace hornets_nest;

  graph::GraphStd<vid_t, eoff_t> graph;
  CommandLineParam cmd(graph, argc, argv, false);

  // graph.read(argv[1], SORT | PRINT_INFO);

  HornetInit hornet_init(
      graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());

  HornetGraph hornet_graph(hornet_init);

  // TODO: Naive Transpose. Replace with actual transpose later
  std::vector<std::set<vid_t>> trans_graph(graph.nV());
  for (auto i = 0; i < graph.nV(); i++) {
    for (auto j = graph.csr_out_offsets()[i];
         j < graph.csr_out_offsets()[i + 1];
         j++) {
      auto nbr = graph.csr_out_edges()[j];
      trans_graph[nbr].insert(i);
    }
  }

  std::vector<eoff_t> trans_csr_offsets(graph.nV() + 1, 0);
  std::vector<vid_t> trans_csr_out_edges(graph.nE());

  for (auto i = 0; i < graph.nV(); i++) {
    trans_csr_offsets[i + 1] = trans_csr_offsets[i] + trans_graph[i].size();
    std::copy(trans_graph[i].begin(),
              trans_graph[i].end(),
              trans_csr_out_edges.begin() + trans_csr_offsets[i]);
  }

  HornetInit trans_hornet_init(
      graph.nV(), graph.nE(), &trans_csr_offsets[0], &trans_csr_out_edges[0]);

  HornetGraph trans_hornet_graph(trans_hornet_init);

  std::cout << "G_Reg:\n";
  hornet_graph.print();

  std::cout << "G_Tran:\n";
  trans_hornet_graph.print();

  StrongCC scc(hornet_graph, trans_hornet_graph);
  scc.run();

  return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;
#if defined(RMM_WRAPPER)
  hornets_nest::gpu::initializeRMMPoolAllocation();
  // update initPoolSize if you know your memory requirement and memory
  // availability in your system, if initial pool size is set to 0 (default
  // value), RMM currently assigns half the device memory.
  {
// scoping technique to make sure that
// hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM
// allocations.
#endif

    ret = exec(argc, argv);

#if defined(RMM_WRAPPER)
  }
  hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif

  return ret;
}
