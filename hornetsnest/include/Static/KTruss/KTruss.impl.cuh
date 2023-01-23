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

void kTrussOneIteration(HornetGraph& hornet,
                        triangle_t*  __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data);

//==============================================================================

KTruss::KTruss(HornetGraph& hornet) : StaticAlgorithm(hornet) {
    hd_data().active_queue.initialize(hornet);
    originalNE = hornet.nE();
    originalNV = hornet.nV();

}

KTruss::~KTruss() {
    release();
}

void KTruss::setInitParameters(int tsp, int nbl, int shifter,
                               int blocks, int sps) {
    hd_data().tsp     = tsp;
    hd_data().nbl     = nbl;
    hd_data().shifter = shifter;
    hd_data().blocks  = blocks;
    hd_data().sps     = sps;
}

void KTruss::init(){
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

void KTruss::createOffSetArray(){

    gpu::memsetZero(hd_data().offset_array, originalNV+1);

    int *tempSize;
    pool.allocate(&tempSize, originalNV+1);

    forAllVertices(hornet, getVertexSizes {tempSize});

    cudaStream_t stream{nullptr};
    thrust::inclusive_scan(rmm::exec_policy(stream), tempSize, tempSize + originalNV, hd_data().offset_array+1);
}

void KTruss::copyOffsetArrayHost(const vert_t* host_offset_array) {
    cudaMemcpy(hd_data().offset_array,host_offset_array,(originalNV + 1)*sizeof(vert_t), cudaMemcpyHostToDevice);
}

void KTruss::copyOffsetArrayDevice(vert_t* device_offset_array){
    cudaMemcpy(hd_data().offset_array,device_offset_array,(originalNV + 1)*sizeof(vert_t), cudaMemcpyDeviceToDevice);
}

vert_t KTruss::getMaxK() {
    return hd_data().max_K;
}

void KTruss::sortHornet(){
  hornet.sort();
}

//==============================================================================

void KTruss::reset() {
    cudaMemset(hd_data().counter,0, sizeof(int));
    hd_data().num_edges_remaining      = originalNE;
    hd_data().full_triangle_iterations = 0;

    resetEdgeArray();
    resetVertexArray();
}

void KTruss::resetVertexArray() {
    gpu::memsetZero(hd_data().triangles_per_vertex, originalNV);
}

void KTruss::resetEdgeArray() {
    gpu::memsetZero(hd_data().triangles_per_edge, originalNE);
}

void KTruss::release() {
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

void KTruss::run() {
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

void KTruss::runForK(int max_K) {
    hd_data().max_K = max_K;

    findTrussOfK();
}

int
KTruss::getGraphEdgeCount(void) {
  return hornet.nE();
}

void
KTruss::copyGraph(vert_t * src, vert_t * dst) {
    auto coo = hornet.getCOO(false);
    cudaMemcpy(src, coo.srcPtr(), sizeof(vert_t)*coo.size(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst, coo.dstPtr(), sizeof(vert_t)*coo.size(), cudaMemcpyDeviceToDevice);
}

void KTruss::findTrussOfK() {
    forAllVertices(hornet, Init { hd_data });
    resetEdgeArray();
    resetVertexArray();

    cudaMemset(hd_data().counter,0, sizeof(int));

    int h_active_vertices = originalNV;

    while (h_active_vertices > 0) {

        hd_data().full_triangle_iterations++;
        kTrussOneIteration(hornet, hd_data().triangles_per_vertex,
                           hd_data().tsp, hd_data().nbl,
                           hd_data().shifter,
                           hd_data().blocks, hd_data().sps,
                           hd_data);

        forAllVertices(hornet, FindUnderK { hd_data });

        int h_counter;
        cudaMemcpy(&h_counter,hd_data().counter, sizeof(int),cudaMemcpyDeviceToHost);

        if (h_counter != 0) {
              UpdatePtr ptr(h_counter, hd_data().src, hd_data().dst);
              Update batch_update(ptr);
              hornet.erase(batch_update);
              hd_data().num_edges_remaining -= h_counter;
    CHECK_CUDA_ERROR
        }
        else{
            return;
        }


        // Resetting the number of active vertices before check
        cudaMemset(hd_data().active_vertices,0, sizeof(int));

        forAllVertices(hornet, CountActive { hd_data });

        sortHornet();

        // Getting the number of active vertices
        cudaMemcpy(&h_active_vertices, hd_data().active_vertices,sizeof(int),cudaMemcpyDeviceToHost);

        resetEdgeArray();
        resetVertexArray();

        cudaMemset(hd_data().counter,0, sizeof(int));
    }
}


} // hornet_alg namespace
