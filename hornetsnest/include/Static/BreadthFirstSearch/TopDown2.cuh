/**
 * @brief Top-Down implementation of Breadth-first Search by using C++11-Style
 *        APIs
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
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
#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>

namespace hornets_nest {

using vid_t = int;
using dist_t = int;

using HornetInit  = ::hornet::HornetInit<vid_t>;
using HornetDynamicGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetStaticGraph = ::hornet::gpu::HornetStatic<vid_t>;

//using HornetGraph = gpu::Csr<EMPTY, EMPTY>;
//using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

using dist_t = int;

template <typename HornetGraph>
class BfsTopDown2 : public StaticAlgorithm<HornetGraph> {
public:
    BfsTopDown2(HornetGraph& hornet);
    ~BfsTopDown2();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    void set_parameters(vid_t source);

    dist_t getLevels(){return current_level;}

private:
    BufferPool pool;
    TwoLevelQueue<vid_t>        queue;
    load_balancing::BinarySearch load_balancing;
    //load_balancing::VertexBased1 load_balancing;
    dist_t* d_distances   { nullptr };
    vid_t   bfs_source    { 0 };
    dist_t  current_level { 0 };
};

using BfsTopDown2Dynamic = BfsTopDown2<HornetDynamicGraph>;
using BfsTopDown2Static  = BfsTopDown2<HornetStaticGraph>;

} // namespace hornets_nest

namespace hornets_nest {

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////


struct BFSOperatorAtomic {                  //deterministic
    dist_t               current_level;
    dist_t*              d_distances;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        if (atomicCAS(d_distances + dst, INF, current_level) == INF)
            queue.insert(dst);
    }
};
//------------------------------------------------------------------------------
/////////////////
// BfsTopDown2 //
/////////////////

#define BFSTOPDOWN2 BfsTopDown2<HornetGraph>

template <typename HornetGraph>
BFSTOPDOWN2::BfsTopDown2(HornetGraph& hornet) :
                                 StaticAlgorithm<HornetGraph>(hornet),
                                 queue(hornet, 5),
                                 load_balancing(hornet) {
    pool.allocate(&d_distances, hornet.nV());
    reset();
}

template <typename HornetGraph>
BFSTOPDOWN2::~BfsTopDown2() {
}

template <typename HornetGraph>
void BFSTOPDOWN2::reset() {
    current_level = 1;
    queue.clear();

    auto distances = d_distances;

    forAllnumV(
        StaticAlgorithm<HornetGraph>::hornet,
        [=] __device__ (int i){ distances[i] = INF; } );
}

template <typename HornetGraph>
void BFSTOPDOWN2::set_parameters(vid_t source) {
    bfs_source = source;
    queue.insert(bfs_source);               // insert bfs source in the frontier
    gpu::memsetZero(d_distances + bfs_source);  //reset source distance
}

template <typename HornetGraph>
void BFSTOPDOWN2::run() {
    while (queue.size() > 0) {

        forAllEdges(
            StaticAlgorithm<HornetGraph>::hornet,
            queue,
                    BFSOperatorAtomic { current_level, d_distances, queue },
                    load_balancing);
        queue.swap();
        current_level++;
    }
}

template <typename HornetGraph>
void BFSTOPDOWN2::release() {
    d_distances = nullptr;
}

template <typename HornetGraph>
bool BFSTOPDOWN2::validate() {
    return true;
}

} // namespace hornets_nest
