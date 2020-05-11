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
#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>


namespace hornets_nest {

// using vert_t = int;
using HornetInit  = ::hornet::HornetInit<vert_t>;
using HornetDynamicGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetStaticGraph = ::hornet::gpu::HornetStatic<vert_t>;


using ulong_t = long long unsigned;

struct KatzTopKData {
    ulong_t*  num_paths_data;
    ulong_t** num_paths; // Will be used for dynamic graph algorithm which
                          // requires storing paths of all iterations.

    ulong_t*  num_paths_curr;
    ulong_t*  num_paths_prev;

    double*   KC;
    double*   lower_bound;
    double*   upper_bound;

    double alpha;
    double alphaI; // Alpha to the power of I  (being the iteration)

    double lower_bound_const;
    double upper_bound_const;

    int K;

    int max_degree;
    int iteration;
    int max_iteration;

    int num_active;    // number of active vertices at each iteration
    int num_prev_active;
    int nV;

    bool*   is_active;
    double* lower_bound_unsorted;
    double* lower_bound_sorted;
    int*    vertex_array_unsorted; // Sorting
    int*    vertex_array_sorted;   // Sorting
};

// Label propogation is based on the values from the previous iteration.
template <typename HornetGraph>
class KatzCentralityTopK : public StaticAlgorithm<HornetGraph> {
  BufferPool pool;
public:
    KatzCentralityTopK(HornetGraph& hornet, int max_iteration,
                   int K, int max_degree, bool is_static = true);
    ~KatzCentralityTopK();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    int get_iteration_count();

    void copyKCToHost(double* host_array);
    void copyNumPathsToHost(ulong_t* host_array);

    KatzTopKData katz_data();

private:
    load_balancing::BinarySearch load_balancing;
    HostDeviceVar<KatzTopKData>     hd_katzdata;
    ulong_t**                   h_paths_ptr;
    bool                        is_static;

    void printKMostImportant();
};




using KatzCentralityTopKDynamicH = KatzCentralityTopK<HornetDynamicGraph>;
using KatzCentralityTopKStatic   = KatzCentralityTopK<HornetStaticGraph>;

// #include "KatzOperators.cuh"


struct Init {
    HostDeviceVar<KatzTopKData> kd;

    // Used at the very beginning
    OPERATOR(vert_t src) {
        kd().num_paths_prev[src] = 1;
        kd().num_paths_curr[src] = 0;
        kd().KC[src]             = 0.0;
        kd().is_active[src]      = true;
    }
};

//------------------------------------------------------------------------------

struct InitNumPathsPerIteration {
    HostDeviceVar<KatzTopKData> kd;

    OPERATOR(vert_t src) {
        kd().num_paths_curr[src] = 0;
    }
};

//------------------------------------------------------------------------------

struct UpdatePathCount {
    HostDeviceVar<KatzTopKData> kd;

    OPERATOR(Vertex& src, Edge& edge){
        auto src_id = src.id();
        auto dst_id = edge.dst_id();
        atomicAdd(kd().num_paths_curr + src_id,
                  kd().num_paths_prev[dst_id]);
    }
};

//------------------------------------------------------------------------------

struct UpdateKatzAndBounds {
    HostDeviceVar<KatzTopKData> kd;

    OPERATOR(vert_t src) {
        kd().KC[src] = kd().KC[src] + kd().alphaI *
                        static_cast<double>(kd().num_paths_curr[src]);
        kd().lower_bound[src] = kd().KC[src] + kd().lower_bound_const *
                                static_cast<double>(kd().num_paths_curr[src]);
        kd().upper_bound[src] = kd().KC[src] + kd().upper_bound_const *
                                static_cast<double>(kd().num_paths_curr[src]);

        if (kd().is_active[src]) {
            int pos = atomicAdd(&(kd.ptr()->num_active), 1);
            kd().vertex_array_unsorted[pos] = src;
            kd().lower_bound_unsorted[pos]  = kd().lower_bound[src];
        }
    }
};

//------------------------------------------------------------------------------

struct CountActive {
    HostDeviceVar<KatzTopKData> kd;

    OPERATOR(vert_t src) {
        auto index = kd().vertex_array_sorted[kd().num_prev_active - kd().K];
        if (kd().upper_bound[src] > kd().lower_bound[index])
            atomicAdd(&(kd.ptr()->num_active), 1);
        else
            kd().is_active[src] = false;
    }
};





#define KATZCENTRALITYTOPK KatzCentralityTopK<HornetGraph>

using length_t = int;

template <typename HornetGraph>
KATZCENTRALITYTOPK::KatzCentralityTopK(HornetGraph& hornet, int max_iteration, int K,
                               int max_degree, bool is_static) :
                                       StaticAlgorithm<HornetGraph>(hornet),
                                       load_balancing(hornet),
                                       is_static(is_static) {
    if (max_iteration <= 0)
        ERROR("Number of max iterations should be greater than zero")

    hd_katzdata().nV            = hornet.nV();
    hd_katzdata().K             = K;
    hd_katzdata().max_degree    = max_degree;
    hd_katzdata().alpha         = 1.0 / (static_cast<double>(max_degree) + 1.0);
    hd_katzdata().max_iteration = max_iteration;

    auto nV = hornet.nV();

    if (is_static) {
        pool.allocate(&hd_katzdata().num_paths_data, nV * 2);
        hd_katzdata().num_paths_prev = hd_katzdata().num_paths_data;
        hd_katzdata().num_paths_curr = hd_katzdata().num_paths_data + nV;
        hd_katzdata().num_paths      = nullptr;
        h_paths_ptr                  = nullptr;
    }
    else {
        pool.allocate(&hd_katzdata().num_paths_data, nV * max_iteration);
        pool.allocate(&hd_katzdata().num_paths, max_iteration);

        host::allocate(h_paths_ptr, max_iteration);
        for(int i = 0; i < max_iteration; i++)
            h_paths_ptr[i] = hd_katzdata().num_paths_data + nV * i;

        hd_katzdata().num_paths_prev = h_paths_ptr[0];
        hd_katzdata().num_paths_curr = h_paths_ptr[1];
        host::copyToDevice(h_paths_ptr, max_iteration, hd_katzdata().num_paths);
    }
    pool.allocate(&hd_katzdata().KC,          nV);
    pool.allocate(&hd_katzdata().lower_bound, nV);
    pool.allocate(&hd_katzdata().upper_bound, nV);

    pool.allocate(&hd_katzdata().is_active,             nV);
    pool.allocate(&hd_katzdata().vertex_array_sorted,   nV);
    pool.allocate(&hd_katzdata().vertex_array_unsorted, nV);
    pool.allocate(&hd_katzdata().lower_bound_sorted,    nV);
    pool.allocate(&hd_katzdata().lower_bound_unsorted,  nV);

    reset();
}

template <typename HornetGraph>
KATZCENTRALITYTOPK::~KatzCentralityTopK() {
    release();
}

template <typename HornetGraph>
void KATZCENTRALITYTOPK::reset() {
    hd_katzdata().iteration = 1;

    if (is_static) {
        hd_katzdata().num_paths_prev = hd_katzdata().num_paths_data;
        hd_katzdata().num_paths_curr = hd_katzdata().num_paths_data +
                                        StaticAlgorithm<HornetGraph>::hornet.nV();
    }
    else {
        hd_katzdata().num_paths_prev = h_paths_ptr[0];
        hd_katzdata().num_paths_curr = h_paths_ptr[1];
    }
}

template <typename HornetGraph>
void KATZCENTRALITYTOPK::release(){
    host::free(h_paths_ptr);
}


template <typename HornetGraph>
void KATZCENTRALITYTOPK::run() {
    forAllnumV(StaticAlgorithm<HornetGraph>::hornet, Init { hd_katzdata });

    hd_katzdata().iteration  = 1;
    hd_katzdata().num_active = StaticAlgorithm<HornetGraph>::hornet.nV();

    while (hd_katzdata().num_active > hd_katzdata().K &&
           hd_katzdata().iteration < hd_katzdata().max_iteration) {

        hd_katzdata().alphaI            = std::pow(hd_katzdata().alpha,
                                                   hd_katzdata().iteration);
        hd_katzdata().lower_bound_const = std::pow(hd_katzdata().alpha,
                                                  hd_katzdata().iteration + 1) /
                                        (1.0 - hd_katzdata().alpha);
        hd_katzdata().upper_bound_const = std::pow(hd_katzdata().alpha,
                                                  hd_katzdata().iteration + 1) /
                                        (1.0 - hd_katzdata().alpha *
                                 static_cast<double>(hd_katzdata().max_degree));
        hd_katzdata().num_active = 0; // Each iteration the number of active
                                     // vertices is set to zero.

        forAllnumV (StaticAlgorithm<HornetGraph>::hornet, InitNumPathsPerIteration { hd_katzdata } );
        forAllEdges(StaticAlgorithm<HornetGraph>::hornet, UpdatePathCount          { hd_katzdata },
                    load_balancing);
        forAllnumV (StaticAlgorithm<HornetGraph>::hornet, UpdateKatzAndBounds      { hd_katzdata } );

        hd_katzdata.sync();

        hd_katzdata().iteration++;
        if(is_static) {
            std::swap(hd_katzdata().num_paths_curr,
                      hd_katzdata().num_paths_prev);
        }
        else {
            auto                    iter = hd_katzdata().iteration;
            hd_katzdata().num_paths_prev = h_paths_ptr[iter - 1];
            hd_katzdata().num_paths_curr = h_paths_ptr[iter - 0];
        }
        auto         old_active_count = hd_katzdata().num_active;
        hd_katzdata().num_prev_active = hd_katzdata().num_active;
        hd_katzdata().num_active      = 0; // Resetting active vertices for
                                           // sorting

        // Notice that the sorts the vertices in an incremental order based on
        // the lower bounds.
        // The algorithms requires the vertices to be sorted in an decremental
        // fashion.
        // As such, we use the num_prev_active variables to store the number of
        // previous active vertices and are able to find the K-th from last
        // vertex (which is essentially going from the tail of the array).
        xlib::CubSortByKey<double, vert_t>::srun
            (hd_katzdata().lower_bound_unsorted,
             hd_katzdata().vertex_array_unsorted,
             old_active_count, hd_katzdata().lower_bound_sorted,
             hd_katzdata().vertex_array_sorted);

        forAllnumV(StaticAlgorithm<HornetGraph>::hornet, CountActive { hd_katzdata } );
        hd_katzdata.sync();
    }
}

template <typename HornetGraph>
void KATZCENTRALITYTOPK::copyKCToHost(double* d) {
    gpu::copyToHost(hd_katzdata().KC, StaticAlgorithm<HornetGraph>::hornet.nV(), d);
}

// This function should only be used directly within run() and is currently
// commented out due to to large execution overheads.
template <typename HornetGraph>
void KATZCENTRALITYTOPK::printKMostImportant() {
    ulong_t* num_paths_curr;
    ulong_t* num_paths_prev;
    int*     vertex_array;
    int*     vertex_array_unsorted;
    double*  KC;
    double*  lower_bound;
    double*  upper_bound;

    auto nV = StaticAlgorithm<HornetGraph>::hornet.nV();
    host::allocate(num_paths_curr, nV);
    host::allocate(num_paths_prev, nV);
    host::allocate(vertex_array,   nV);
    host::allocate(vertex_array_unsorted, nV);
    host::allocate(KC,          nV);
    host::allocate(lower_bound, nV);
    host::allocate(upper_bound, nV);

    gpu::copyToHost(hd_katzdata().lower_bound, nV, lower_bound);
    gpu::copyToHost(hd_katzdata().upper_bound, nV, upper_bound);
    gpu::copyToHost(hd_katzdata().KC, nV, KC);
    gpu::copyToHost(hd_katzdata().vertex_array_sorted, nV, vertex_array);
    gpu::copyToHost(hd_katzdata().vertex_array_unsorted, nV,
                    vertex_array_unsorted);

    if (hd_katzdata().num_prev_active > hd_katzdata().K) {
        for (int i = hd_katzdata().num_prev_active - 1;
                i >= hd_katzdata().num_prev_active - hd_katzdata().K; i--) {
            vert_t j = vertex_array[i];
            std::cout << j << "\t\t" << KC[j] << "\t\t" << upper_bound[j]
                      << upper_bound[j] - lower_bound[j] << "\n";
        }
    }
    std::cout << std::endl;

    host::free(num_paths_curr);
    host::free(num_paths_prev);
    host::free(vertex_array);
    host::free(vertex_array_unsorted);
    host::free(KC);
    host::free(lower_bound);
    host::free(upper_bound);
}

template <typename HornetGraph>
int KATZCENTRALITYTOPK::get_iteration_count() {
    return hd_katzdata().iteration;
}

template <typename HornetGraph>
bool KATZCENTRALITYTOPK::validate() {
    return true;
}





} // hornetAlgs namespace
