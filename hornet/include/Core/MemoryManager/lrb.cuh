#ifndef LRB_CUH
#define LRB_CUH

#include <rmm/thrust_rmm_allocator.h>
#include <vector>

////////////////////////////////////////////////////////////////
//LRB//
////////////////////////////////////////////////////////////////
template <typename degree_t>
constexpr int BitsPWrd = sizeof(degree_t)*8;

template <typename degree_t>
constexpr int NumberBins = sizeof(degree_t)*8 + 1;

template <typename degree_t>
__device__ inline
typename std::enable_if<(sizeof(degree_t) == 4), int>::type
ceilLog2_p1(degree_t val) {
  return BitsPWrd<degree_t> - __clz(val) + (__popc(val) > 1);
}

template <typename degree_t>
__device__ inline
typename std::enable_if<(sizeof(degree_t) == 8), int>::type
ceilLog2_p1(degree_t val) {
  return BitsPWrd<degree_t> - __clzll(val) + (__popcll(val) > 1);
}

template <typename degree_t>
__global__
void binDegrees(
    degree_t * bins,
    degree_t const * deg,
    degree_t count) {
  constexpr int BinCount = NumberBins<degree_t>;
  __shared__ degree_t lBin[BinCount];
  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    lBin[i] = 0;
  }
  __syncthreads();

  for (degree_t i = threadIdx.x + (blockIdx.x*blockDim.x);
      i < count; i += gridDim.x*blockDim.x) {
    atomicAdd(lBin + ceilLog2_p1(deg[i]), 1);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    atomicAdd(bins + i, lBin[i]);
  }
}

template <typename vid_t, typename degree_t>
__global__
void rebinIds(
    vid_t * reorgV,
    degree_t * reorgD,
    vid_t const * v,
    degree_t * prefixBins,
    degree_t const * deg,
    degree_t count) {
  constexpr int BinCount = NumberBins<degree_t>;
  __shared__ degree_t lBin[BinCount];
  __shared__ int lPos[BinCount];
  if (threadIdx.x < BinCount) {
    lBin[threadIdx.x] = 0; lPos[threadIdx.x] = 0;
  }
  __syncthreads();

  degree_t tid = threadIdx.x + blockIdx.x*blockDim.x;
  int threadBin;
  degree_t threadPos;
  if (tid < count) {
    threadBin = ceilLog2_p1(deg[tid]);
    threadPos = atomicAdd(lBin + threadBin, 1);
  }
  __syncthreads();

  if (threadIdx.x < BinCount) {
    lPos[threadIdx.x] = atomicAdd(prefixBins + threadIdx.x, lBin[threadIdx.x]);
  }
  __syncthreads();

  if (tid < count) {
    reorgV[lPos[threadBin] + threadPos] = v[tid];
    reorgD[lPos[threadBin] + threadPos] = deg[tid];
  }
}

template <typename D>
__global__ void
exclusive_scan(D const* data, D* out) {
  constexpr int BinCount = NumberBins<D>;
  D lData[BinCount];
  lData[0] = 0;
  for (int i = 0; i < BinCount - 1; ++i) {
    lData[i+1] = lData[i] + data[i];
  }
  for (int i = 0; i < BinCount; ++i) {
    out[i] = lData[i];
  }
}

template <typename V, typename D>
void lrb(
    rmm::device_vector<V> &rV,
    rmm::device_vector<D> &rD,
    rmm::device_vector<D> &rBins,
    rmm::device_vector<V> &v,
    rmm::device_vector<D> &d,
    rmm::device_vector<D> &tempBins) {
  const unsigned BLOCK_SIZE = 512;
  unsigned blocks = (v.size() + BLOCK_SIZE - 1)/BLOCK_SIZE;
  binDegrees<D><<<blocks, BLOCK_SIZE>>>(
      rBins.data().get(),
      d.data().get(),
      d.size());
  exclusive_scan<<<1,1>>>(rBins.data().get(), tempBins.data().get());
  rebinIds<V, D><<<blocks, BLOCK_SIZE>>>(
      rV.data().get(), rD.data().get(), v.data().get(),
      tempBins.data().get(), d.data().get(), v.size());
}

template <typename D>
struct DegreeHist
{
  D count;
  D ceilLogDegree;
};

template <typename D>
std::vector<DegreeHist<D>> degreeHist(D * deg, D count) {
  const unsigned BLOCK_SIZE = 512;
  unsigned blocks = (count + BLOCK_SIZE - 1)/BLOCK_SIZE;
  rmm::device_vector<D> bins(NumberBins<D>, 0);
  binDegrees<D><<<blocks, BLOCK_SIZE>>>(
      bins.data().get(),
      deg,
      count);
  std::vector<D> hBins(bins.size());
  thrust::copy(bins.begin(), bins.end(), hBins.begin());
  std::vector<DegreeHist<D>> hist; hist.reserve(bins.size());
  for (unsigned i = 1; i < hBins.size(); ++i) {
    if (hBins[i] != 0) {
      DegreeHist<D> c;
      c.count = hBins[i];
      c.ceilLogDegree = (1<<(i-1));
      hist.push_back(c);
    }
  }
  return hist;
}

template <typename D>
D maxEBSize(D * deg, D count) {
  auto hist = degreeHist<D>(deg, count);
  D max_eb_size = 0;
  int i = 0;
  for (auto &h : hist) {
    max_eb_size = std::max(max_eb_size, h.count*h.ceilLogDegree);
    D val = h.count*h.ceilLogDegree;
    float lg = std::ceil(std::log2(val));
    std::cerr<<i++<<" "<<h.count<<" "<<h.ceilLogDegree<<" "<<(1<<static_cast<int>(lg))<<"\n";
  }
  return max_eb_size;
}
////////////////////////////////////////////////////////////////

#endif
