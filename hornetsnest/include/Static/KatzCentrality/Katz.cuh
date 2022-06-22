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

#include <HornetAlg.hpp>
#include <BufferPool.cuh>

#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>


namespace hornets_nest {

namespace katz_detail {

struct Scale : public thrust::unary_function<double, double> {
  double scale;

  Scale(double scale_) : scale(scale_) {}

  __host__ __device__ double operator()(const double &x) {
    return x*scale;
  }
};

}

// using vert_t = int;
using HornetInit  = ::hornet::HornetInit<vert_t>;
using HornetDynamicGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetStaticGraph = ::hornet::gpu::HornetStatic<vert_t>;


using ulong_t = long long unsigned;

struct KatzData {
    ulong_t*  num_paths_data;
    ulong_t** num_paths; // Will be used for dynamic graph algorithm which
                          // requires storing paths of all iterations.

    ulong_t*  num_paths_curr;
    ulong_t*  num_paths_prev;

    double*   KC;

    double alpha;
    double alphaI; // Alpha to the power of I  (being the iteration)

    int iteration;
    int max_iteration;

    int nV;
    bool normalized;
};

// Label propogation is based on the values from the previous iteration.
template <typename HornetGraph>
class KatzCentrality : public StaticAlgorithm<HornetGraph> {
protected:
  BufferPool pool;
public:
    KatzCentrality(HornetGraph& hornet,
                   double alpha_ = 0.1,
                   int max_iteration = 100,
                   double tolerance = 1e-5,
                   bool normalized = true,
                   bool is_static = true,
                   double * kc_out = nullptr);
    ~KatzCentrality();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    int get_iteration_count();

    void copyKCToHost(double* host_array);
    void copyKCToDevice(double* device_array);  // Deep copy
    void copyNumPathsToHost(ulong_t* host_array);

    KatzData katz_data();

    double getAlpha();
    bool hasConverged();

private:
    load_balancing::BinarySearch load_balancing;
    HostDeviceVar<KatzData>     hd_katzdata;
    ulong_t**                   h_paths_ptr;
    bool                          is_static;
    bool                          converged;

    double tol;
    double * kc_out_ptr;

};



using KatzCentralityDynamicH = KatzCentrality<HornetDynamicGraph>;
using KatzCentralityStatic  = KatzCentrality<HornetStaticGraph>;


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//                          Algorithm Operators
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


struct Init {
    HostDeviceVar<KatzData> kd;

    // Used at the very beginning
    OPERATOR(vert_t src) {
        kd().num_paths_prev[src] = 1;
        kd().num_paths_curr[src] = 0;
        kd().KC[src]             = 0;
    }
};

//------------------------------------------------------------------------------

struct InitNumPathsPerIteration {
    HostDeviceVar<KatzData> kd;

    OPERATOR(vert_t src) {
        kd().num_paths_curr[src] = 0;
    }
};

//------------------------------------------------------------------------------

struct UpdatePathCount {
    HostDeviceVar<KatzData> kd;

    OPERATOR(Vertex& src, Edge& edge){
        auto src_id = src.id();
        auto dst_id = edge.dst_id();
        atomicAdd(kd().num_paths_curr + src_id, kd().num_paths_prev[dst_id]);
    }
};


//------------------------------------------------------------------------------

struct UpdateKatz {
    HostDeviceVar<KatzData> kd;

    OPERATOR(vert_t src) {
        kd().KC[src] = kd().KC[src] + kd().alphaI *
                        static_cast<double>(kd().num_paths_curr[src]);
    }
};

struct Saxpy {
  double alphaI;

  Saxpy(double alphaI_) : alphaI(alphaI_) {}

  __host__ __device__
    double operator()(const ulong_t& path, const double& katz) {
      return path*alphaI + katz;
    }
};



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//                          Implementation details
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------




#define KATZCENTRALITY KatzCentrality<HornetGraph>



using length_t = int;

// Constructor. User needs to define maximal number of iterations that algorithm
// should be executed.

template <typename HornetGraph>
KATZCENTRALITY::KatzCentrality(HornetGraph& hornet,
                               double alpha_,
                               int max_iteration,
                               double tolerance,
                               bool normalized,
                               bool is_static,
                               double * kc_out) :
                                 StaticAlgorithm<HornetGraph>(hornet),
                                 load_balancing(hornet),
                                 is_static(is_static) {
    if (max_iteration <= 0)
        ERROR("Number of max iterations should be greater than zero")

    tol = tolerance;

    // All alpha values need to be smaller than this value to ensure convergene/
    double minimalAlpha = 1.0 / (static_cast<double>(hornet.max_degree()));

    // If default alpha value was not set by user, then set default value
    if(minimalAlpha<alpha_){
        hd_katzdata().alpha = 1.0 / (static_cast<double>(hornet.max_degree()+1));
    } else{
        hd_katzdata().alpha         = alpha_;
    }

    hd_katzdata().nV            = hornet.nV();
    hd_katzdata().max_iteration = max_iteration;

    hd_katzdata().normalized    = normalized;

    auto nV = hornet.nV();

    // Two different control paths for static vs. dynamic algorithm.
    if (is_static) {
        // Static algorithm uses two arrays for storing number of paths.
        // One array is for the current frontier. One array is for the next frontier.
        // These will alternate every iteration.
        pool.allocate(&hd_katzdata().num_paths_data, nV * 2);
        hd_katzdata().num_paths_prev = hd_katzdata().num_paths_data;
        hd_katzdata().num_paths_curr = hd_katzdata().num_paths_data + nV;
        hd_katzdata().num_paths      = nullptr;
        h_paths_ptr                  = nullptr;
    }
    else {
        // Dynamic algorithm stores the number of paths per iteration.
        // As such the number of iterations might be limited by the amount of available
        // system memory.
        pool.allocate(&hd_katzdata().num_paths_data, nV * max_iteration);
        pool.allocate(&hd_katzdata().num_paths, max_iteration);

        // The host manages the pointers to starting location of the current and next frontiers
        host::allocate(h_paths_ptr, max_iteration);
        for(int i = 0; i < max_iteration; i++)
            h_paths_ptr[i] = hd_katzdata().num_paths_data + nV * i;

        hd_katzdata().num_paths_prev = h_paths_ptr[0];
        hd_katzdata().num_paths_curr = h_paths_ptr[1];
        host::copyToDevice(h_paths_ptr, max_iteration, hd_katzdata().num_paths);
    }
    if (kc_out == nullptr) {
      pool.allocate(&hd_katzdata().KC,          nV);
    } else {
      hd_katzdata().KC = kc_out;
    }
    kc_out_ptr = kc_out;

    reset();
}

template <typename HornetGraph>
KATZCENTRALITY::~KatzCentrality() {
    release();
}

template <typename HornetGraph>
void KATZCENTRALITY::reset() {
    hd_katzdata().iteration = 1;

    // Reseting the values so we can restart the execution
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


// Free all allocated resources
template <typename HornetGraph>
void KATZCENTRALITY::release(){
    host::free(h_paths_ptr);
}


template <typename HornetGraph>
bool KATZCENTRALITY::hasConverged(void) {
  return converged;
}

template <typename HornetGraph>
double KATZCENTRALITY::getAlpha(void) {
  return hd_katzdata().alpha;
}

template <typename HornetGraph>
void KATZCENTRALITY::run() {
    // Initialized the paths and set the number of paths to 1 for all vertices
    // (Each vertex has a path to itself). This is equivalent to iteration 0.
    forAllnumV(StaticAlgorithm<HornetGraph>::hornet, Init { hd_katzdata });
    tol = tol * StaticAlgorithm<HornetGraph>::hornet.nV();

    // Update Kataz Centrality scores for the given number of iterations
    hd_katzdata().iteration  = 1;
    double err = 0;
    while (hd_katzdata().iteration <= hd_katzdata().max_iteration) {
        // Alpha^I is computed at the beginning of every iteration.
        hd_katzdata().alphaI            = std::pow(hd_katzdata().alpha,hd_katzdata().iteration);
        forAllnumV (StaticAlgorithm<HornetGraph>::hornet, InitNumPathsPerIteration { hd_katzdata } );
        forAllEdges(StaticAlgorithm<HornetGraph>::hornet, UpdatePathCount          { hd_katzdata }, load_balancing);
        forAllnumV (StaticAlgorithm<HornetGraph>::hornet, UpdateKatz               { hd_katzdata } );

        hd_katzdata().iteration++;

        {
          auto curr_path = thrust::device_pointer_cast(hd_katzdata().num_paths_curr);
          err = thrust::transform_reduce(
              curr_path,
              curr_path + StaticAlgorithm<HornetGraph>::hornet.nV(),
              katz_detail::Scale(hd_katzdata().alphaI),
              static_cast<double>(0),
              thrust::plus<double>());
        }

        if(is_static) {
            std::swap(hd_katzdata().num_paths_curr,hd_katzdata().num_paths_prev);
        }
        else {
            auto                    iter = hd_katzdata().iteration;
            hd_katzdata().num_paths_prev = h_paths_ptr[iter - 1];
            hd_katzdata().num_paths_curr = h_paths_ptr[iter - 0];
        }
        if (err < tol) { break; }

    }
    hd_katzdata().iteration--;
    converged = (err < tol);

    if(hd_katzdata().normalized == true){

        double h_normFactor = thrust::inner_product(thrust::device,
            hd_katzdata().KC, hd_katzdata().KC + StaticAlgorithm<HornetGraph>::hornet.nV(),
            hd_katzdata().KC,
            static_cast<double>(0),
            thrust::plus<double>(),
            thrust::multiplies<double>());

        if(h_normFactor>0){
            h_normFactor = 1.0/std::sqrt(h_normFactor);

        thrust::transform(thrust::device,
                        hd_katzdata().KC,
                        hd_katzdata().KC+StaticAlgorithm<HornetGraph>::hornet.nV(),
                        hd_katzdata().KC,
                        [h_normFactor] __device__ (double kc)
                        {
                            return h_normFactor*kc;
                        });

        }

    }

}

template <typename HornetGraph>
void KATZCENTRALITY::copyKCToHost(double* d) {
    gpu::copyToHost(hd_katzdata().KC, StaticAlgorithm<HornetGraph>::hornet.nV(), d);
}

void copyKCToDevice(double* device_array);  // Deep copy
template <typename HornetGraph>
void KATZCENTRALITY::copyKCToDevice(double* device_array) {
    gpu::copyToDevice(hd_katzdata().KC, StaticAlgorithm<HornetGraph>::hornet.nV(), device_array);
}



template <typename HornetGraph>
int KATZCENTRALITY::get_iteration_count() {
    return hd_katzdata().iteration;
}

template <typename HornetGraph>
bool KATZCENTRALITY::validate() {
    return true;
}





} // hornetAlgs namespace
