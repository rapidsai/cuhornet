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
#pragma once

#include "HornetAlg.hpp"
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <thrust/functional.h>
#include <BufferPool.cuh>


namespace hornets_nest {

#define CORENUMBER CoreNumber<HornetGraph>

using HornetGraph = ::hornet::gpu::HornetStatic<vert_t>;
using HornetInit  = ::hornet::HornetInit<vert_t>;
using UpdatePtr   = ::hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update      = ::hornet::gpu::BatchUpdate<vert_t>;

template <typename HornetGraph>
class CoreNumber : public StaticAlgorithm<HornetGraph> {
public:
    CoreNumber(HornetGraph &hornet, int *core_number_ptr);
    ~CoreNumber();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }
    void set_hcopy(HornetGraph *h_copy);

private:
    BufferPool pool;
    long edge_vertex_count;

    load_balancing::BinarySearch load_balancing;

    TwoLevelQueue<vert_t> peel_vqueue;
    TwoLevelQueue<vert_t> active_queue;
    TwoLevelQueue<vert_t> iter_queue;

    vert_t *vertex_pres { nullptr };
    vert_t *vertex_deg { nullptr };
    int *core_number { nullptr };
};

using CoreNumberStatic = CoreNumber<HornetGraph>;

}

namespace hornets_nest {

template <typename HornetGraph>
CORENUMBER::CoreNumber(HornetGraph &hornet, int *core_number_ptr) : 
                        StaticAlgorithm<HornetGraph>(hornet),
                        peel_vqueue(hornet),
                        active_queue(hornet),
                        iter_queue(hornet),
                        load_balancing(hornet),
                        core_number(core_number_ptr)
                        {

    pool.allocate(&vertex_pres, hornet.nV());
    pool.allocate(&vertex_deg, hornet.nV());
}

template <typename HornetGraph>
CORENUMBER::~CoreNumber() {
}

struct ActiveVertices {
    vert_t *vertex_pres;
    vert_t *deg;
    TwoLevelQueue<vert_t> active_queue;

    OPERATOR(Vertex &v) {
        vert_t id = v.id();
        if (v.degree() > 0) {
            vertex_pres[id] = 1;
            active_queue.insert(id);
            deg[id] = v.degree();
        }
    }
};

struct PeelVertices {
    vert_t *vertex_pres;
    vert_t *deg;
    int peel;
    TwoLevelQueue<vert_t> peel_queue;
    TwoLevelQueue<vert_t> iter_queue;
    
    //mark vertices with degrees less than peel
    OPERATOR(Vertex &v) {
        vert_t id = v.id();
        if (vertex_pres[id] == 1 && deg[id] <= peel) {
            vertex_pres[id] = 2;
            peel_queue.insert(id);
            iter_queue.insert(id);
        }
    }
};

struct RemovePres {
    vert_t *vertex_pres;
    int * core_number;
    int peel;
    
    OPERATOR(Vertex &v) {
        vert_t id = v.id();
        if (vertex_pres[id] == 2) {
            vertex_pres[id] = 0;
            core_number[id] = peel;
        }
    }
};

struct DecrementDegree {
    vert_t *deg;

    OPERATOR(Vertex &v, Edge &e) {
        vert_t src = v.id();
        vert_t dst = e.dst_id();
        atomicAdd(&deg[src], -1);
        atomicAdd(&deg[dst], -1);
    }
};

template <typename HornetGraph>
void CORENUMBER::reset() {
    peel_vqueue.swap();
    active_queue.swap();
    iter_queue.swap();
}

template <typename HornetGraph>
void CORENUMBER::run() {
    HornetGraph& hornet = StaticAlgorithm<HornetGraph>::hornet;
    forAllVertices(hornet, ActiveVertices { vertex_pres, vertex_deg, active_queue });

    active_queue.swap();
    int n_active = active_queue.size();
    int peel = 0;

    while (n_active > 0) {
      forAllVertices(hornet, active_queue, 
          PeelVertices { vertex_pres, vertex_deg, peel, peel_vqueue, iter_queue} );
        iter_queue.swap();

      n_active -= iter_queue.size();

      if (iter_queue.size() == 0) {
        peel++;
        peel_vqueue.swap();
        forAllVertices(hornet, active_queue, RemovePres { vertex_pres, core_number, peel-1 });
      } else {
        forAllEdges(hornet, iter_queue, DecrementDegree { vertex_deg }, load_balancing);
      }
    }
    forAllVertices(hornet, active_queue, RemovePres { vertex_pres, core_number, peel });
}

template <typename HornetGraph>
void CORENUMBER::release() {
}
}
