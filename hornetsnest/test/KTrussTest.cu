/**
 * @brief
 * @author Oded Green                                                       <br>
 * NVIDIA
 * @date July, 2019
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
#include "Static/KTruss/KTruss.cuh"
#include <StandardAPI.hpp>
#include <Device/Util/Timer.cuh>
#include <Graph/GraphStd.hpp>


// #include "Hornet.hpp" // Shouldn't this be done by default?
using namespace hornets_nest;


int exec(int argc, char* argv[]) {
    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;
    using namespace hornets_nest;
    using namespace timer;


	  cudaSetDevice(0);
    GraphStd<vert_t, vert_t> graph(UNDIRECTED);

    graph.read(argv[1], SORT | PRINT_INFO );

    HornetInit hornet_init(graph.nV(), graph.nE(),
                           graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_gpu(hornet_init);

    vert_t* gpuOffset;

    gpu::allocate(gpuOffset, graph.nV()+1);
    cudaMemcpy(gpuOffset,graph.csr_out_offsets(),sizeof(vert_t)*(graph.nV()+1), cudaMemcpyHostToDevice);

    // int temp;

    // int temp2=scanf("%d",&temp);
    // printf("%d %d\n",temp+1,temp2);

    KTruss ktruss (hornet_gpu);
    ktruss.init();
    ktruss.reset();

    ktruss.copyOffsetArrayHost(graph.csr_out_offsets());
    // ktruss.setInitParameters(1, 32, 0, 64000, 32);
    // ktruss.createOffSetArray();
    ktruss.setInitParameters(4, 8, 2, 64000, 32);

    Timer<DEVICE> TM;
    ktruss.reset();
    TM.start();

    ktruss.run();

    TM.stop();

    auto total_time = TM.duration();
    TM.print("Time to find the k-truss");
    std::cout << "The Maximal K-Truss is : " << ktruss.getMaxK() << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;
  {

    ret = exec(argc, argv);

  }

  return ret;
}
