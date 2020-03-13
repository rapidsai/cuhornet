#include <limits>

namespace hornet {

namespace gpu {

//TODO InvalidEdgeCount
template <typename degree_t>
struct InvalidEdgeCount {
  __device__
  degree_t operator()(degree_t deg) {
    if (deg == 0) return 0;
    else return (1<<xlib::ceil_log2(deg)) - deg;
  }
};

template <int BLOCK_SIZE, typename HornetDeviceT, typename vid_t, typename degree_t>
__global__
void invalidateEdges(
    HornetDeviceT hornet,
    degree_t * offsets,
    size_t offsets_count,
    vid_t max_vertex) {
  const int ITEMS_PER_BLOCK = xlib::smem_per_block<degree_t, BLOCK_SIZE>();
  __shared__ degree_t smem[ITEMS_PER_BLOCK];
  const auto& lambda = [&] (int pos, degree_t edge_offset) {
    auto vertex = hornet.vertex(pos);
    vid_t * dst = vertex.neighbor_ptr() + vertex.degree();
    dst[edge_offset] = max_vertex;
  };
  xlib::binarySearchLB<BLOCK_SIZE>(offsets, offsets_count, smem, lambda);

}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNET::
sort(void) {
  if (_nE == 0) { return; }
  cudaStream_t stream{nullptr};

  rmm::device_vector<degree_t> offsets(_nV + 1);
  degree_t * vertex_degrees = _vertex_data.get_soa_ptr().template get<0>();
  thrust::transform(rmm::exec_policy(stream)->on(stream),
      vertex_degrees, vertex_degrees + _nV,
      offsets.begin(),
      InvalidEdgeCount<degree_t>());
    CHECK_CUDA_ERROR

  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
      offsets.begin(), offsets.end(), offsets.begin());
    CHECK_CUDA_ERROR



  degree_t number_of_edges = offsets[_nV];
  HornetDeviceT hornet_device = device();
  const int BLOCK_SIZE = 256;
  int smem = xlib::DeviceProperty::smem_per_block<degree_t>(BLOCK_SIZE);
  int num_blocks = xlib::ceil_div(number_of_edges, smem);

  if (num_blocks == 0) { return; }
  vid_t max = std::numeric_limits<vid_t>::max();
  invalidateEdges<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(hornet_device,
      offsets.data().get(), offsets.size(), max);
    CHECK_CUDA_ERROR

  _ba_manager.sort();
  CHECK_CUDA_ERROR
}

}//namespace gpu

}//namespace hornet
