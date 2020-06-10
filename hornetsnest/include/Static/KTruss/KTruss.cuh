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

#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>

namespace hornets_nest {

// const bool _FORCE_SOA = true;

using triangle_t = vert_t;

using HornetGraph = hornet::gpu::Hornet<vert_t>;

template <typename T>
using HornetGraphWeighted = hornet::gpu::Hornet<vert_t, hornet::EMPTY, hornet::TypeList<T>>;

using HornetInit  = ::hornet::HornetInit<vert_t>;

using UpdatePtr   = ::hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update      = ::hornet::gpu::BatchUpdate<vert_t>;

template <typename T>
using UpdatePtrWeighted  = ::hornet::BatchUpdatePtr<vert_t, hornet::TypeList<T>, hornet::DeviceType::DEVICE>;

template <typename T>
using UpdateWeighted = ::hornet::gpu::BatchUpdate<vert_t, hornet::TypeList<T>>;


struct KTrussData {
    int max_K;

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int* is_active;
    int* offset_array;
    int* triangles_per_edge;
    int* triangles_per_vertex;

    vert_t* src;
    vert_t* dst;
    int*    counter;
    int*    active_vertices;

    TwoLevelQueue<vert_t> active_queue; // Stores all the active vertices

    int full_triangle_iterations;

    vert_t nv;
    off_t ne;                  // undirected-edges
    off_t num_edges_remaining; // undirected-edges
};

//==============================================================================

// Label propogation is based on the values from the previous iteration.
class KTruss : public StaticAlgorithm<HornetGraph> {
  BufferPool pool;
public:
    KTruss(HornetGraph& hornet);
    ~KTruss();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    //--------------------------------------------------------------------------
    void setInitParameters(int tsp, int nbl, int shifter,
                           int blocks, int sps);
    void init();

    void findTrussOfK();
    void runForK(int max_K);
    int  getGraphEdgeCount(void);
    void copyGraph(vert_t * src, vert_t * dst);

    void createOffSetArray();
    void copyOffsetArrayHost(const vert_t* host_offset_array);
    void copyOffsetArrayDevice(vert_t* device_offset_array);
    void resetEdgeArray();
    void resetVertexArray();

    vert_t getIterationCount();
    vert_t getMaxK();

    void sortHornet();

private:
    HostDeviceVar<KTrussData> hd_data;

    vert_t originalNE;
    vert_t originalNV;
};

template <typename T>
class KTrussWeighted : public StaticAlgorithm<HornetGraphWeighted<T>> {
  BufferPool pool;
public:
    KTrussWeighted(HornetGraphWeighted<T>& hornet);
    ~KTrussWeighted();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    //--------------------------------------------------------------------------
    void setInitParameters(int tsp, int nbl, int shifter,
                           int blocks, int sps);
    void init();

    void findTrussOfK();
    void runForK(int max_K);
    int  getGraphEdgeCount(void);
    void copyGraph(vert_t * src, vert_t * dst, T * weight);

    void createOffSetArray();
    void copyOffsetArrayHost(const vert_t* host_offset_array);
    void copyOffsetArrayDevice(vert_t* device_offset_array);
    void resetEdgeArray();
    void resetVertexArray();

    vert_t getIterationCount();
    vert_t getMaxK();

    void sortHornet();

private:
    HornetGraphWeighted<T> &hnt;

    HostDeviceVar<KTrussData> hd_data;

    vert_t originalNE;
    vert_t originalNV;
};

} // namespace hornets_nest


#include "Static/KTruss/KTruss.impl.cuh"
#include "Static/KTruss/KTrussWeighted.impl.cuh"
