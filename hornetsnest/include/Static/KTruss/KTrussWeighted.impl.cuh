/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "Static/KTruss/KTruss.cuh"
#include "Static/KTruss/KTrussOperators.cuh"
#include "Static/KTruss/KTrussSupport.cuh"

#include <iostream>
#include <Device/Util/Timer.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

namespace hornets_nest {

template <typename HornetGraphType>
void kTrussOneIterationWeighted(HornetGraphType& hornet,
                        triangle_t*  __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data);

//==============================================================================

template <typename T>
KTrussWeighted<T>::KTrussWeighted(HornetGraphWeighted<T>& hornet) : StaticAlgorithm<HornetGraphWeighted<T>>(hornet), hnt(hornet){
    hd_data().active_queue.initialize(hnt);
    originalNE = hnt.nE();
    originalNV = hnt.nV();

}

template <typename T>
KTrussWeighted<T>::~KTrussWeighted() {
    release();
}

template <typename T>
void KTrussWeighted<T>::setInitParameters(int tsp, int nbl, int shifter,
                               int blocks, int sps) {
    hd_data().tsp     = tsp;
    hd_data().nbl     = nbl;
    hd_data().shifter = shifter;
    hd_data().blocks  = blocks;
    hd_data().sps     = sps;
}

template <typename T>
void KTrussWeighted<T>::init(){
    pool.allocate(&hd_data().is_active,            originalNV);
    pool.allocate(&hd_data().offset_array,         originalNV + 1);
    pool.allocate(&hd_data().triangles_per_vertex, originalNV);
    pool.allocate(&hd_data().triangles_per_edge,   originalNE);
    pool.allocate(&hd_data().src,                  originalNE);
    pool.allocate(&hd_data().dst,                  originalNE);
    pool.allocate(&hd_data().counter,              1);
    pool.allocate(&hd_data().active_vertices,      1);
    reset();
}

template <typename T>
void KTrussWeighted<T>::createOffSetArray(){

    gpu::memsetZero(hd_data().offset_array, originalNV+1);

    int *tempSize;
    pool.allocate(&tempSize, originalNV+1);

    forAllVertices(hnt, getVertexSizes {tempSize});

    cudaStream_t stream{nullptr};
    thrust::inclusive_scan(rmm::exec_policy(stream), tempSize, tempSize + originalNV, hd_data().offset_array+1);
}

template <typename T>
void KTrussWeighted<T>::copyOffsetArrayHost(const vert_t* host_offset_array) {
    cudaMemcpy(hd_data().offset_array,host_offset_array,(originalNV + 1)*sizeof(vert_t), cudaMemcpyHostToDevice);
}

template <typename T>
void KTrussWeighted<T>::copyOffsetArrayDevice(vert_t* device_offset_array){
    cudaMemcpy(hd_data().offset_array,device_offset_array,(originalNV + 1)*sizeof(vert_t), cudaMemcpyDeviceToDevice);
}

template <typename T>
vert_t KTrussWeighted<T>::getMaxK() {
    return hd_data().max_K;
}

template <typename T>
void KTrussWeighted<T>::sortHornet(){
  hnt.sort();
}

//==============================================================================

template <typename T>
void KTrussWeighted<T>::reset() {
    cudaMemset(hd_data().counter,0, sizeof(int));
    hd_data().num_edges_remaining      = originalNE;
    hd_data().full_triangle_iterations = 0;

    resetEdgeArray();
    resetVertexArray();
}

template <typename T>
void KTrussWeighted<T>::resetVertexArray() {
    gpu::memsetZero(hd_data().triangles_per_vertex, originalNV);
}

template <typename T>
void KTrussWeighted<T>::resetEdgeArray() {
    gpu::memsetZero(hd_data().triangles_per_edge, originalNE);
}

template <typename T>
void KTrussWeighted<T>::release() {
    hd_data().is_active            = nullptr;
    hd_data().offset_array         = nullptr;
    hd_data().triangles_per_edge   = nullptr;
    hd_data().triangles_per_vertex = nullptr;
    hd_data().counter              = nullptr;
    hd_data().active_vertices      = nullptr;
    hd_data().src                  = nullptr;
    hd_data().dst                  = nullptr;

}

//==============================================================================

template <typename T>
void KTrussWeighted<T>::run() {
    hd_data().max_K = 3;
    int  iterations = 0;

    while (true) {
        findTrussOfK();

        if (hd_data().num_edges_remaining <= 0) {
            hd_data().max_K--;
            break;
        }
        hd_data().max_K++;

        iterations++;
    }
}

template <typename T>
void KTrussWeighted<T>::runForK(int max_K) {
    hd_data().max_K = max_K;

    findTrussOfK();
}

template <typename T>
int
KTrussWeighted<T>::getGraphEdgeCount(void) {
  return hnt.nE();
}

template <typename T>
void
KTrussWeighted<T>::copyGraph(vert_t * src, vert_t * dst, T * weight) {
    auto coo = hnt.getCOO(false);
    cudaMemcpy(src, coo.srcPtr(), sizeof(vert_t)*coo.size(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst, coo.dstPtr(), sizeof(vert_t)*coo.size(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(weight, coo.template edgeMetaPtr<0>(), sizeof(T)*coo.size(), cudaMemcpyDeviceToDevice);
}

template <typename T>
void KTrussWeighted<T>::findTrussOfK() {
    forAllVertices(hnt, Init { hd_data });
    resetEdgeArray();
    resetVertexArray();

    cudaMemset(hd_data().counter,0, sizeof(int));

    int h_active_vertices = originalNV;

    while (h_active_vertices > 0) {

        hd_data().full_triangle_iterations++;
        kTrussOneIterationWeighted(hnt, hd_data().triangles_per_vertex,
                           hd_data().tsp, hd_data().nbl,
                           hd_data().shifter,
                           hd_data().blocks, hd_data().sps,
                           hd_data);

        forAllVertices(hnt, FindUnderK { hd_data });

        int h_counter;
        cudaMemcpy(&h_counter,hd_data().counter, sizeof(int),cudaMemcpyDeviceToHost);

        if (h_counter != 0) {
              UpdatePtrWeighted<T> ptr(h_counter, hd_data().src, hd_data().dst, nullptr);
              UpdateWeighted<T> batch_update(ptr);
              hnt.erase(batch_update);
              hd_data().num_edges_remaining -= h_counter;
    CHECK_CUDA_ERROR
        }
        else{
            return;
        }


        // Resetting the number of active vertices before check
        cudaMemset(hd_data().active_vertices,0, sizeof(int));

        forAllVertices(hnt, CountActive { hd_data });

        sortHornet();

        // Getting the number of active vertices
        cudaMemcpy(&h_active_vertices, hd_data().active_vertices,sizeof(int),cudaMemcpyDeviceToHost);

        resetEdgeArray();
        resetVertexArray();

        cudaMemset(hd_data().counter,0, sizeof(int));
    }
}


} // hornet_alg namespace
