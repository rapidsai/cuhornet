#ifndef SOA_DATA_SORT_CUH
#define SOA_DATA_SORT_CUH

#include <cub/cub.cuh>
#include <rmm/device_buffer.hpp>

namespace hornet {

namespace detail {

template <typename T>
__device__ void bubble_sort(int32_t size, T *edges){
  T temp;
  for(int32_t i=0; i<(size-1); i++){
    int32_t min_idx=i;
    for(int32_t j=i+1; j<(size); j++){
      if(edges[j]<edges[min_idx])
        min_idx=j;
    }
    temp            = edges[min_idx];
    edges[min_idx]  = edges[i];
    edges[i]        = temp;
  }
}

template <typename T0, typename T1>
__device__ void bubble_sort(int32_t size, T0 *key, T1 *val){
  T0 temp0;
  T1 temp1;
  for(int32_t i=0; i<(size-1); i++){
    int32_t min_idx=i;
    for(int32_t j=i+1; j<(size); j++){
      if(key[j]<key[min_idx])
        min_idx=j;
    }
    temp0        = key[min_idx];
    key[min_idx] = key[i];
    key[i]       = temp0;
    temp1        = val[min_idx];
    val[min_idx] = val[i];
    val[i]       = temp1;
  }
}

template <typename degree_t, typename... EdgeTypes>
__launch_bounds__ (32)
__global__
typename std::enable_if<(2 < sizeof...(EdgeTypes)), void>::type
small_segmented_sort_kernel(
    CSoAPtr<EdgeTypes...> edges,
    degree_t * global_index,
    const degree_t nE,
    const degree_t segment_length) {
  int i = threadIdx.x+blockIdx.x*blockDim.x;

  using T = typename xlib::SelectType<0, EdgeTypes...>::type;
  T local_edges[32];
  degree_t local_index[32];
  T * global_edges = edges.template get<0>();

  for (int j = 0; j < segment_length; ++j) {
    local_edges[j] = global_edges[segment_length*i + j];
    local_index[j] = segment_length*i + j;
  }
  bubble_sort(segment_length, local_edges, local_index);
  for (int j = 0; j < segment_length; ++j) {
    global_edges[segment_length*i + j] = local_edges[j];
    global_index[segment_length*i + j] = local_index[j];
  }
}

template <typename degree_t, typename... EdgeTypes>
__launch_bounds__ (32)
__global__
typename std::enable_if<(2 == sizeof...(EdgeTypes)), void>::type
small_segmented_sort_kernel(
    CSoAPtr<EdgeTypes...> edges,
    const degree_t nE,
    const degree_t segment_length) {
  int i = threadIdx.x+blockIdx.x*blockDim.x;

  using T0 = typename xlib::SelectType<0, EdgeTypes...>::type;
  using T1 = typename xlib::SelectType<1, EdgeTypes...>::type;
  T0 local_edges0[32];
  T1 local_edges1[32];

  for (int j = 0; j < segment_length; ++j) {
    local_edges0[j] = edges.template get<0>()[segment_length*i + j];
    local_edges1[j] = edges.template get<1>()[segment_length*i + j];
  }
  bubble_sort(segment_length, local_edges0, local_edges1);
  for (int j = 0; j < segment_length; ++j) {
    edges.template get<0>()[segment_length*i + j] = local_edges0[j];
    edges.template get<1>()[segment_length*i + j] = local_edges1[j];
  }
}

template <typename degree_t, typename... EdgeTypes>
__launch_bounds__ (32)
__global__
typename std::enable_if<(1 == sizeof...(EdgeTypes)), void>::type
small_segmented_sort_kernel(
    CSoAPtr<EdgeTypes...> edges,
    const degree_t nE,
    const degree_t segment_length) {
  int i = threadIdx.x+blockIdx.x*blockDim.x;

  using T = typename xlib::SelectType<0, EdgeTypes...>::type;
  T local_edges[32];
  T * global_edges = edges.template get<0>();

  for (int j = 0; j < segment_length; ++j) {
    local_edges[j] = global_edges[segment_length*i + j];
  }
  bubble_sort(segment_length, local_edges);
  for (int j = 0; j < segment_length; ++j) {
    global_edges[segment_length*i + j] = local_edges[j];
  }
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 >= sizeof...(EdgeTypes)), void>::type
small_segmented_sort(CSoAPtr<EdgeTypes...> soa, degree_t capacity, degree_t segment_length) {
  detail::small_segmented_sort_kernel<<<capacity/(32*segment_length), 32>>>(soa, capacity, segment_length);
  CHECK_CUDA_ERROR
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 < sizeof...(EdgeTypes)), void>::type
small_segmented_sort(CSoAPtr<EdgeTypes...> &soa, degree_t capacity, degree_t segment_length) {
  rmm::device_vector<degree_t> index(capacity);

  CSoAData<TypeList<EdgeTypes...>, DeviceType::DEVICE> temp_data(capacity);
  temp_data.copy(soa, DeviceType::DEVICE, capacity);
  CSoAPtr<EdgeTypes...> temp_soa = temp_data.get_soa_ptr();

  detail::small_segmented_sort_kernel<<<capacity/(32*segment_length), 32>>>(temp_soa,
      index.data().get(),
      capacity, segment_length);
  //TODO : Check correctness
  RecursiveCopy<0, 0>::copy(temp_soa, DeviceType::DEVICE, soa, DeviceType::DEVICE, capacity);
  RecursiveGather<1, sizeof...(EdgeTypes)>::assign(temp_soa, soa, index, capacity);
}

template <
    typename    T,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
  __global__ void CubBlockSortKernel(T *edges) {
    using namespace cub;
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockRadixSort type for our thread block
    typedef BlockRadixSort<T, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Shared memory
    __shared__ union TempStorage
    {
      typename BlockLoadT::TempStorage        load;
      typename BlockRadixSortT::TempStorage   sort;
    } temp_storage;

    T items[ITEMS_PER_THREAD];
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Our current block's offset
    int block_offset = blockIdx.x * TILE_SIZE;
    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(edges + block_offset, items);
    // Barrier for smem reuse
    __syncthreads();

    // Sort keys
    BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items);

    // Store output in striped fashion
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, edges + block_offset, items);
}

template <
    typename    T0,
    typename    T1,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__ (BLOCK_THREADS)
  __global__ void CubBlockSortKernel(T0 *edge_keys, T1 *edge_vals) {
    using namespace cub;
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockLoad<T0, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT0;
    typedef BlockLoad<T1, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT1;
    // Specialize BlockRadixSort type for our thread block
    typedef BlockRadixSort<T0, BLOCK_THREADS, ITEMS_PER_THREAD, T1> BlockRadixSortT;
    // Shared memory
    __shared__ union TempStorage
    {
      typename BlockLoadT0::TempStorage        load0;
      typename BlockLoadT1::TempStorage        load1;
      typename BlockRadixSortT::TempStorage    sort;
    } temp_storage;

    T0 key[ITEMS_PER_THREAD];
    T1 val[ITEMS_PER_THREAD];
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Our current block's offset
    int block_offset = blockIdx.x * TILE_SIZE;
    // Load keys into a blocked arrangement
    BlockLoadT0(temp_storage.load0).Load(edge_keys + block_offset, key);
    // Barrier for smem reuse
    __syncthreads();

    // Load vals into a blocked arrangement
    BlockLoadT1(temp_storage.load1).Load(edge_vals + block_offset, val);
    // Barrier for smem reuse
    __syncthreads();

    // Sort keys
    BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(key, val);

    // Store output in striped fashion
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, edge_keys + block_offset, key);

    // Store output in striped fashion
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, edge_vals + block_offset, val);
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(1 == sizeof...(EdgeTypes)), void>::type
cub_block_segmented_sort(CSoAPtr<EdgeTypes...> soa, degree_t capacity, degree_t segment_length) {
  using T = typename xlib::SelectType<0, EdgeTypes...>::type;
  T * edges = soa.template get<0>();
  int blocks = capacity/segment_length;
  if (segment_length ==  32)  CubBlockSortKernel<T,32,  1><<<blocks, 32>>>(edges);
  if (segment_length ==  64)  CubBlockSortKernel<T,32,  2><<<blocks, 32>>>(edges);
  if (segment_length ==  128) CubBlockSortKernel<T,32,  4><<<blocks, 32>>>(edges);
  if (segment_length ==  256) CubBlockSortKernel<T,32,  8><<<blocks, 32>>>(edges);
  if (segment_length ==  512) CubBlockSortKernel<T,32, 16><<<blocks, 32>>>(edges);
  if (segment_length == 1024) CubBlockSortKernel<T,256, 4><<<blocks,256>>>(edges);
  if (segment_length == 2048) CubBlockSortKernel<T,256, 8><<<blocks,256>>>(edges);
  if (segment_length == 4096) CubBlockSortKernel<T,256,16><<<blocks,256>>>(edges);
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 == sizeof...(EdgeTypes)), void>::type
cub_block_segmented_sort(CSoAPtr<EdgeTypes...> soa, degree_t capacity, degree_t segment_length) {
  using T0 = typename xlib::SelectType<0, EdgeTypes...>::type;
  T0 * key = soa.template get<0>();
  using T1 = typename xlib::SelectType<1, EdgeTypes...>::type;
  T1 * val = soa.template get<1>();
  int blocks = capacity/segment_length;
  if (segment_length ==  32)  CubBlockSortKernel<T0,T1,32,  1><<<blocks, 32>>>(key, val);
  if (segment_length ==  64)  CubBlockSortKernel<T0,T1,32,  2><<<blocks, 32>>>(key, val);
  if (segment_length ==  128) CubBlockSortKernel<T0,T1,32,  4><<<blocks, 32>>>(key, val);
  if (segment_length ==  256) CubBlockSortKernel<T0,T1,32,  8><<<blocks, 32>>>(key, val);
  if (segment_length ==  512) CubBlockSortKernel<T0,T1,32, 16><<<blocks, 32>>>(key, val);
  if (segment_length == 1024) CubBlockSortKernel<T0,T1,256, 4><<<blocks,256>>>(key, val);
  if (segment_length == 2048) CubBlockSortKernel<T0,T1,256, 8><<<blocks,256>>>(key, val);
  if (segment_length == 4096) CubBlockSortKernel<T0,T1,256,16><<<blocks,256>>>(key, val);
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 < sizeof...(EdgeTypes)), void>::type
cub_block_segmented_sort(CSoAPtr<EdgeTypes...> &soa, degree_t capacity, degree_t segment_length) {
  CSoAData<TypeList<EdgeTypes...>, DeviceType::DEVICE> temp_data(capacity);
  temp_data.copy(soa, DeviceType::DEVICE, capacity);
  CSoAPtr<EdgeTypes...> temp_soa = temp_data.get_soa_ptr();

  cudaStream_t stream{nullptr};
  rmm::device_vector<degree_t> index(capacity);
  thrust::sequence(rmm::exec_policy(stream)->on(stream), index.begin(), index.end());
  using T0 = typename xlib::SelectType<0, EdgeTypes...>::type;
  T0 * key = temp_soa.template get<0>();
  using T1 = degree_t;
  T1 * val = index.data().get();
  int blocks = capacity/segment_length;
  if (segment_length ==  32)  CubBlockSortKernel<T0,T1,32,  1><<<blocks, 32>>>(key, val);
  if (segment_length ==  64)  CubBlockSortKernel<T0,T1,32,  2><<<blocks, 32>>>(key, val);
  if (segment_length ==  128) CubBlockSortKernel<T0,T1,32,  4><<<blocks, 32>>>(key, val);
  if (segment_length ==  256) CubBlockSortKernel<T0,T1,32,  8><<<blocks, 32>>>(key, val);
  if (segment_length ==  512) CubBlockSortKernel<T0,T1,32, 16><<<blocks, 32>>>(key, val);
  if (segment_length == 1024) CubBlockSortKernel<T0,T1,256, 4><<<blocks,256>>>(key, val);
  if (segment_length == 2048) CubBlockSortKernel<T0,T1,256, 8><<<blocks,256>>>(key, val);
  if (segment_length == 4096) CubBlockSortKernel<T0,T1,256,16><<<blocks,256>>>(key, val);
  RecursiveCopy<0, 0>::copy(temp_soa, DeviceType::DEVICE, soa, DeviceType::DEVICE, capacity);
  RecursiveGather<1, sizeof...(EdgeTypes)>::assign(temp_soa, soa, index, capacity);
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(1 == sizeof...(EdgeTypes)), void>::type
cub_segmented_sort(CSoAPtr<EdgeTypes...> &soa, degree_t capacity, degree_t segment_length) {
  CSoAData<TypeList<EdgeTypes...>, DeviceType::DEVICE> temp_data(capacity);
  temp_data.copy(soa, DeviceType::DEVICE, capacity);
  CSoAPtr<EdgeTypes...> temp_soa = temp_data.get_soa_ptr();

  cudaStream_t stream{nullptr};
  using T = typename xlib::SelectType<0, EdgeTypes...>::type;
  T * in_edges  = temp_soa.template get<0>();
  T * out_edges = soa.template get<0>();

  degree_t offset_count = capacity/segment_length;
  rmm::device_vector<degree_t> offsets(offset_count + 1);
  thrust::transform(rmm::exec_policy(stream)->on(stream),
      offsets.begin(), offsets.end(),
      thrust::make_constant_iterator(segment_length),
      offsets.begin(),
      thrust::multiplies<degree_t>());

  size_t tempStorageBytes = 0;

  cub::DeviceSegmentedRadixSort::SortKeys(
      NULL, tempStorageBytes, in_edges, out_edges, capacity,
      offsets.size() - 1, offsets.data().get(), offsets.data().get() + 1);
  rmm::device_buffer tempStorage(tempStorageBytes);
  cub::DeviceSegmentedRadixSort::SortKeys(
      tempStorage.data(), tempStorageBytes, in_edges, out_edges, capacity,
      offsets.size() - 1, offsets.data().get(), offsets.data().get() + 1);
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 == sizeof...(EdgeTypes)), void>::type
cub_segmented_sort(CSoAPtr<EdgeTypes...> &soa, degree_t capacity, degree_t segment_length) {
  CSoAData<TypeList<EdgeTypes...>, DeviceType::DEVICE> temp_data(capacity);
  temp_data.copy(soa, DeviceType::DEVICE, capacity);
  CSoAPtr<EdgeTypes...> temp_soa = temp_data.get_soa_ptr();
  cudaStream_t stream{nullptr};

  using T0 = typename xlib::SelectType<0, EdgeTypes...>::type;
  using T1 = typename xlib::SelectType<1, EdgeTypes...>::type;
  T0 * in_key  = temp_soa.template get<0>();
  T0 * out_key = soa.template get<0>();
  T1 * in_val  = temp_soa.template get<1>();
  T1 * out_val = soa.template get<1>();

  degree_t offset_count = capacity/segment_length;
  rmm::device_vector<degree_t> offsets(offset_count + 1);
  thrust::transform(rmm::exec_policy(stream)->on(stream),
      offsets.begin(), offsets.end(),
      thrust::make_constant_iterator(segment_length),
      offsets.begin(),
      thrust::multiplies<degree_t>());

  size_t tempStorageBytes = 0;

  cub::DeviceSegmentedRadixSort::SortPairs(
      NULL, tempStorageBytes, in_key, out_key, in_val, out_val, capacity,
      offsets.size() - 1, offsets.data().get(), offsets.data().get() + 1);
  rmm::device_buffer tempStorage(tempStorageBytes);
  cub::DeviceSegmentedRadixSort::SortPairs(
      tempStorage.data(), tempStorageBytes, in_key, out_key, in_val, out_val, capacity,
      offsets.size() - 1, offsets.data().get(), offsets.data().get() + 1);
}

template <typename degree_t, typename... EdgeTypes>
typename std::enable_if<(2 < sizeof...(EdgeTypes)), void>::type
cub_segmented_sort(CSoAPtr<EdgeTypes...> &soa, degree_t capacity, degree_t segment_length) {
  CSoAData<TypeList<EdgeTypes...>, DeviceType::DEVICE> temp_data(capacity);
  temp_data.copy(soa, DeviceType::DEVICE, capacity);
  CSoAPtr<EdgeTypes...> temp_soa = temp_data.get_soa_ptr();
  cudaStream_t stream{nullptr};

  rmm::device_vector<degree_t> index(capacity);
  rmm::device_vector<degree_t> out_index(capacity);
  thrust::sequence(index.begin(), index.end());

  using T0 = typename xlib::SelectType<0, EdgeTypes...>::type;
  using T1 = degree_t;
  T0 * in_key  = temp_soa.template get<0>();
  T0 * out_key = soa.template get<0>();
  T1 * in_val  = index.data().get();
  T1 * out_val = out_index.data().get();

  degree_t offset_count = capacity/segment_length;
  rmm::device_vector<degree_t> offsets(offset_count + 1);
  thrust::transform(rmm::exec_policy(stream)->on(stream),
      offsets.begin(), offsets.end(),
      thrust::make_constant_iterator(segment_length),
      offsets.begin(),
      thrust::multiplies<degree_t>());

  size_t tempStorageBytes = 0;

  cub::DeviceSegmentedRadixSort::SortPairs(
      NULL, tempStorageBytes, in_key, out_key, in_val, out_val, capacity,
      offsets.size() - 1, offsets.data().get(), offsets.data().get() + 1);
  rmm::device_buffer tempStorage(tempStorageBytes);
  cub::DeviceSegmentedRadixSort::SortPairs(
      tempStorage.data(), tempStorageBytes, in_key, out_key, in_val, out_val, capacity,
      offsets.size() - 1, offsets.data().get(), offsets.data().get() + 1);

  RecursiveGather<1, sizeof...(EdgeTypes)>::assign(temp_soa, soa, out_index, capacity);
}

}//namespace detail

template<typename... Ts, DeviceType device_t>
void
CSoAData<TypeList<Ts...>, device_t>::
segmented_sort(int segment_length_log2) {
  //static_assert(device_t == DeviceType::DEVICE, "CSoAData<TypeList<Ts...>, device_t>::segmented_sort called with device_t != DeviceType::DEVICE");
  int segment_length = 1<<segment_length_log2;
  if (segment_length == 1) { return; }
  if (segment_length <= 32) {
    detail::small_segmented_sort(_soa, _capacity, segment_length);
  } else if (segment_length <= 4096) {
    detail::cub_block_segmented_sort(_soa, _capacity, segment_length);
  } else {
    detail::cub_segmented_sort(_soa, _capacity, segment_length);
  }
}

}//namespace hornet
#endif
