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

#define INT_SIZE (sizeof(int) * CHAR_BIT)
#include "Static/StrongCC/scc.cuh"

namespace hornets_nest {
using UpdatePtr =
    ::hornet::BatchUpdatePtr<vid_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
using Update = ::hornet::gpu::BatchUpdate<vid_t>;

struct InitRootData {
  int* visited;
  vid_t root;

  OPERATOR(int i) { visited[root / INT_SIZE] = 1 << (root % INT_SIZE); }
};

struct GetSrcIds {
  vid_t* dst;
  const vid2_t* data_ptr;

  OPERATOR(int i) { dst[i] = data_ptr[i].x; }
};

struct GetDstIds {
  vid_t* dst;
  const vid2_t* data_ptr;

  OPERATOR(int i) { dst[i] = data_ptr[i].y; }
};

struct PrintBatch {
  vid_t* src;
  vid_t* dst;

  OPERATOR(int i) { printf("src %d dst %d\n", src[i], dst[i]); }
};

struct LabelZeroDegreeNodes {
  vid_t* d_num_components;
  vid_t* d_component_labels;
  TwoLevelQueue<vid_t> queue;

  OPERATOR(Vertex& vertex) {
    // Zero OutDegree and Unlabelled
    if (!vertex.degree() && d_component_labels[vertex.id()] == -1) {
      vid_t old_comp_val = atomicAdd(d_num_components, 1);
      d_component_labels[vertex.id()] = old_comp_val;
      queue.insert(vertex.id());
    }
  }
};

struct LabelNodes {
  vid_t comp_id;
  vid_t* d_component_labels;

  OPERATOR(Vertex& vertex) { d_component_labels[vertex.id()] = comp_id; }
};

struct GetTotalDeg {
  degree_t* sum;

  OPERATOR(Vertex& vertex) { atomicAdd(sum, vertex.degree()); }
};

struct MakeEdgePairs {
  TwoLevelQueue<vid2_t> queue_pair;

  OPERATOR(Vertex& src, Edge& edge) {
    queue_pair.insert({src.id(), edge.dst_id()});
  }
};

struct BFSOperator {  // deterministic
  int* visited;
  TwoLevelQueue<vid_t> queue;

  OPERATOR(Vertex& vertex, Edge& edge) {
    vid_t dst_id = edge.dst_id();
    int old_val =
        atomicOr(&visited[dst_id / INT_SIZE], 1 << (dst_id % INT_SIZE));
    bool already_visited = (old_val >> (dst_id % INT_SIZE)) & 1;

    if (!already_visited)
      queue.insert(dst_id);
  }
};
}
