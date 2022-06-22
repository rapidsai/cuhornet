#pragma once

#include "HornetAlg.hpp"
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <BufferPool.cuh>


namespace hornets_nest {

#define KCORE KCore<HornetGraph>

//using vert_t = int;
using HornetGraph = ::hornet::gpu::Hornet<vert_t>;
using HornetInit  = ::hornet::HornetInit<vert_t>;
using UpdatePtr   = ::hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update      = ::hornet::gpu::BatchUpdate<vert_t>;

struct KCoreData {
    vert_t *src;
    vert_t *dst;
    int   *counter;
};

template <typename HornetGraph>
class KCore : public StaticAlgorithm<HornetGraph> {
public:
    KCore(HornetGraph &hornet);
    ~KCore();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }
    void set_hcopy(HornetGraph *h_copy);

private:
    BufferPool pool;
    HostDeviceVar<KCoreData> hd_data;
    long edge_vertex_count;

    load_balancing::BinarySearch load_balancing;

    TwoLevelQueue<vert_t> vqueue;
    TwoLevelQueue<vert_t> peel_vqueue;
    TwoLevelQueue<vert_t> active_queue;
    TwoLevelQueue<vert_t> iter_queue;

    vert_t *vertex_pres { nullptr };
    vert_t *vertex_color { nullptr };
    vert_t *vertex_deg { nullptr };
    thrust::device_vector<uint32_t> core_number;
};

using KCoreDynamic = KCore<HornetGraph>;

}

namespace hornets_nest {

template <bool device>
void print_ptr(vert_t* src, vert_t* dst, int count, bool sort = false) {
  vert_t * s = nullptr;
  vert_t * d = nullptr;
  std::vector<vert_t> S, D;
  if (!device) {
    s = src; d = dst;
  } else {
    S.resize(count);
    D.resize(count);
    s = S.data();
    d = D.data();
    cudaMemcpy(s, src, sizeof(vert_t)*count, cudaMemcpyDeviceToHost);
    cudaMemcpy(d, dst, sizeof(vert_t)*count, cudaMemcpyDeviceToHost);
  }
  std::vector<std::pair<vert_t, vert_t>> v;
  for (int i = 0; i < count; ++i) {
    v.push_back(std::make_pair(s[i], d[i]));
  }
  if (sort) { std::sort(v.begin(), v.end()); }
  for (unsigned i = 0; i < v.size(); ++i) {
    std::cout<<i<<"\t"<<v[i].first<<"\t"<<v[i].second<<"\n";
  }
}

template <bool device>
void print_ptr(vert_t* src, int count, bool sort = false) {
  vert_t * s = nullptr;
  std::vector<vert_t> S;
  if (!device) {
    s = src;
  } else {
    S.resize(count);
    s = S.data();
    cudaMemcpy(s, src, sizeof(vert_t)*count, cudaMemcpyDeviceToHost);
  }
  if (sort) { std::sort(S.begin(), S.end()); }
  for (unsigned i = 0; i < S.size(); ++i) {
    std::cout<<i<<"\t"<<S[i]<<"\n";
  }
}

template <typename HornetGraph>
KCORE::KCore(HornetGraph &hornet) :
                        StaticAlgorithm<HornetGraph>(hornet),
                        vqueue(hornet),
                        peel_vqueue(hornet),
                        active_queue(hornet),
                        iter_queue(hornet),
                        load_balancing(hornet)
                        {

    pool.allocate(&vertex_pres, hornet.nV());
    pool.allocate(&vertex_color, hornet.nV());
    pool.allocate(&vertex_deg, hornet.nV());
    pool.allocate(&hd_data().src,    hornet.nE());
    pool.allocate(&hd_data().dst,    hornet.nE());
    pool.allocate(&hd_data().counter, 1);
    core_number.resize(hornet.nV());
}

template <typename HornetGraph>
KCORE::~KCore() {
}

struct Comp {
    using Tuple = thrust::tuple<vert_t, vert_t, uint32_t>;
    __host__
        bool operator()(Tuple a, Tuple b) {
            if ( (thrust::get<0>(a) < thrust::get<0>(b)) ||
                    ( (thrust::get<1>(a) == thrust::get<1>(b)) && (thrust::get<1>(a) < thrust::get<1>(b)) ) ) {
                return true;
            } else {
                return false;
            }
        }
};

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
    uint32_t peel;
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
    uint32_t * core_number;
    uint32_t peel;

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

struct ExtractSubgraph {
    HostDeviceVar<KCoreData> hd;
    vert_t *vertex_pres;

    OPERATOR(Vertex &v, Edge &e) {
        vert_t src = v.id();
        vert_t dst = e.dst_id();
        if (vertex_pres[src] == 2 && vertex_pres[dst] == 2) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;
        }
    }
};

struct GetDegOne {
    TwoLevelQueue<vert_t> vqueue;
    vert_t *vertex_color;

    OPERATOR(Vertex &v) {
        vert_t id = v.id();
        if (v.degree() == 1) {
            vqueue.insert(id);
            vertex_color[id] = 1;
        }
    }
};

struct DegOneEdges {
    HostDeviceVar<KCoreData> hd;
    vert_t *vertex_color;

    OPERATOR(Vertex &v, Edge &e) {
        vert_t src = v.id();
        vert_t dst = e.dst_id();

        if (vertex_color[src] || vertex_color[dst]) {
            int spot = atomicAdd(hd().counter, 1);
            hd().src[spot] = src;
            hd().dst[spot] = dst;

            if (!vertex_color[src] || !vertex_color[dst]) {
                int spot_rev = atomicAdd(hd().counter, 1);
                hd().src[spot_rev] = dst;
                hd().dst[spot_rev] = src;
            }
        }
    }
};

template <typename HornetGraph>
void KCORE::reset() {
    vqueue.swap();
    peel_vqueue.swap();
    active_queue.swap();
    iter_queue.swap();
}

void oper_bidirect_batch(HornetGraph &hornet, vert_t *src, vert_t *dst, int size, bool print = false) {
  UpdatePtr ptr(size, src, dst);
  Update batch_update(ptr);
  hornet.erase(batch_update);
}

void kcores_new(HornetGraph &hornet,
            uint32_t *core_number,
            TwoLevelQueue<vert_t> &peel_queue,
            TwoLevelQueue<vert_t> &active_queue,
            TwoLevelQueue<vert_t> &iter_queue,
            load_balancing::BinarySearch& load_balancing,
            vert_t *deg,
            vert_t *vertex_pres) {
    forAllVertices(hornet, ActiveVertices { vertex_pres, deg, active_queue });

    active_queue.swap();
    int n_active = active_queue.size();
    uint32_t peel = 0;

    while (n_active > 0) {
      forAllVertices(hornet, active_queue,
          PeelVertices { vertex_pres, deg, peel, peel_queue, iter_queue} );
        iter_queue.swap();

      n_active -= iter_queue.size();

      if (iter_queue.size() == 0) {
        peel++;
        peel_queue.swap();
        if (n_active > 0) {
          forAllVertices(hornet, active_queue, RemovePres { vertex_pres, core_number, peel });
        }
      } else {
        forAllEdges(hornet, iter_queue, DecrementDegree { deg }, load_balancing);
      }
    }
}

template <typename HornetGraph>
void KCORE::run() {
    HornetGraph& hornet = StaticAlgorithm<HornetGraph>::hornet;
    kcores_new(hornet, core_number.data().get(), peel_vqueue, active_queue, iter_queue,
               load_balancing, vertex_deg, vertex_pres);
    thrust::copy(core_number.begin(), core_number.end(), std::ostream_iterator<uint32_t>(std::cout, "\n"));
}

void kcores_new(HornetGraph &hornet,
            HostDeviceVar<KCoreData>& hd,
            TwoLevelQueue<vert_t> &peel_queue,
            TwoLevelQueue<vert_t> &active_queue,
            TwoLevelQueue<vert_t> &iter_queue,
            load_balancing::BinarySearch& load_balancing,
            vert_t *deg,
            vert_t *vertex_pres,
            uint32_t *max_peel,
            int *batch_size) {
    forAllVertices(hornet, ActiveVertices { vertex_pres, deg, active_queue });

    active_queue.swap();
    int n_active = active_queue.size();
    uint32_t peel = 0;

    while (n_active > 0) {
      forAllVertices(hornet, active_queue,
          PeelVertices { vertex_pres, deg, peel, peel_queue, iter_queue} );
        iter_queue.swap();

      n_active -= iter_queue.size();

      if (iter_queue.size() == 0) {
        peel++;
        peel_queue.swap();
        if (n_active > 0) {
          forAllVertices(hornet, active_queue, RemovePres { vertex_pres });
        }
      } else {
        forAllEdges(hornet, iter_queue, DecrementDegree { deg }, load_balancing);
      }

    }

    gpu::memsetZero(hd().counter);  // reset counter.
    peel_queue.swap();
    forAllEdges(hornet, peel_queue,
                    ExtractSubgraph { hd, vertex_pres }, load_balancing);

    *max_peel = peel;
    int size = 0;
    cudaMemcpy(&size, hd().counter, sizeof(int), cudaMemcpyDeviceToHost);
    //print<true>(hd().src, hd().dst, size, true);
    *batch_size = size;
}

void json_dump(vert_t *src, vert_t *dst, uint32_t *peel, uint32_t peel_edges, bool sort_output = false) {
    if (sort_output) {
        auto iter = thrust::make_zip_iterator(thrust::make_tuple(src, dst, peel));
        thrust::sort(thrust::host, iter, iter + peel_edges, Comp());
    }
    std::ofstream output_file;
    output_file.open("edge_core_number.txt");

    output_file << "{\n";
    for (uint32_t i = 0; i < peel_edges; i++) {
        output_file << "\"" << src[i] << "," << dst[i] << "\": " << peel[i];
        if (i < peel_edges - 1) {
            output_file << ",";
        }
        output_file << "\n";
    }
    output_file << "}";
    output_file.close();
}

//template <typename HornetGraph>
//void KCORE::run() {
//    HornetGraph& hornet = StaticAlgorithm<HornetGraph>::hornet;
//    omp_set_num_threads(72);
//    thrust::host_vector<uint32_t> core_number(hornet.nV());
//    vert_t *src     = new vert_t[hornet.nE()];
//    vert_t *dst     = new vert_t[hornet.nE()];
//    uint32_t len = hornet.nE() / 2 + 1;
//    uint32_t *peel = new uint32_t[hornet.nE()];
//    uint32_t peel_edges = 0;
//    uint32_t ne = hornet.nE();
//    std::cout << "nv: " << hornet.nV() << std::endl;
//    std::cout << "ne: " << ne << std::endl;
//
//    auto pres = vertex_pres;
//    auto deg = vertex_deg;
//    auto color = vertex_color;
//
//    forAllnumV(hornet, [=] __device__ (int i){ pres[i] = 0; } );
//    forAllnumV(hornet, [=] __device__ (int i){ deg[i] = 0; } );
//    forAllnumV(hornet, [=] __device__ (int i){ color[i] = 0; } );
//
//    //Timer<DEVICE> TM;
//    //TM.start();
//
//    /* Begin degree 1 vertex preprocessing optimization */
//
//    // Find vertices of degree 1.
//    forAllVertices(hornet, GetDegOne { vqueue, vertex_color });
//    vqueue.swap();
//
//    // Find the edges incident to these vertices.
//    gpu::memsetZero(hd_data().counter);  // reset counter.
//    forAllEdges(hornet, vqueue,
//                    DegOneEdges { hd_data, vertex_color }, load_balancing);
//
//    // Mark edges with peel 1.
//    int peel_one_count = 0;
//    cudaMemcpy(&peel_one_count, hd_data().counter, sizeof(int), cudaMemcpyDeviceToHost);
//    #pragma omp parallel for
//    for (int i = 0; i < peel_one_count; i++) {
//        peel[i] = 1;
//    }
//
//    cudaMemcpy(src, hd_data().src, peel_one_count * sizeof(vert_t),
//                    cudaMemcpyDeviceToHost);
//    cudaMemcpy(dst, hd_data().dst, peel_one_count * sizeof(vert_t),
//                    cudaMemcpyDeviceToHost);
//
//    peel_edges = (uint32_t)peel_one_count;
//
//    // Delete peel 1 edges.
//    oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, peel_one_count);
//
//    /* Begin running main kcore algorithm */
//    while (peel_edges < ne) {
//        uint32_t max_peel = 0;
//        int batch_size = 0;
//
//        kcores_new(hornet, hd_data, peel_vqueue, active_queue, iter_queue,
//                   load_balancing, vertex_deg, vertex_pres, &max_peel, &batch_size);
//        std::cout << "max_peel: " << max_peel << "\n";
//
//        if (batch_size > 0) {
//            cudaMemcpy(src + peel_edges, hd_data().src,
//                       batch_size * sizeof(vert_t), cudaMemcpyDeviceToHost);
//
//            cudaMemcpy(dst + peel_edges, hd_data().dst,
//                       batch_size * sizeof(vert_t), cudaMemcpyDeviceToHost);
//
//            #pragma omp parallel for
//            for (int i = 0; i < batch_size; i++) {
//                peel[peel_edges + i] = max_peel;
//            }
//
//            peel_edges += batch_size;
//        }
//        oper_bidirect_batch(hornet, hd_data().src, hd_data().dst, batch_size);
//    }
//    //TM.stop();
//    //TM.print("KCore");
//    json_dump(src, dst, peel, peel_edges, true);
//    {
//      std::ofstream output_file;
//      output_file.open("vertex_core_number.txt");
//      thrust::host_vector<vert_t> out_src(peel_edges);
//      thrust::host_vector<uint32_t> out_peel(peel_edges);
//      thrust::reduce_by_key(thrust::host,
//          src, src + out_src.size(),
//          peel,
//          out_src.begin(), out_peel.begin(),
//          thrust::equal_to<vert_t>(),
//          thrust::maximum<uint32_t>());
//      int length = hornet.nV();
//      out_src.resize(length);
//      out_peel.resize(length);
//      for (int i = 0; i < length; ++i) {
//        output_file<<out_src[i]<<" "<<out_peel[i]<<"\n";
//      }
//    }
//}

template <typename HornetGraph>
void KCORE::release() {
    hd_data().src = nullptr;
    hd_data().dst = nullptr;
    hd_data().counter = nullptr;
}
}
