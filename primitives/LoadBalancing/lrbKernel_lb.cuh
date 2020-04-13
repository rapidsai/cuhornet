 
/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


/*
Details of Logarthmic Radix Binning (LRB for short) can be found in additional detail in

[1] Fox, James, Alok Tripathy, and Oded Green
    "Improving Scheduling for Irregular Applications with Logarithmic Radix Binning."
    IEEE High Performance Extreme Computing Conference (HPEC), 2019

[2] Green, Oded, James Fox, Alex Watkins, Alok Tripathy, Kasimir Gabert, Euna Kim,
    Xiaojing An, Kumar Aatish, and David A. Bader.
    "Logarithmic radix binning and vectorized triangle counting."
    IEEE High Performance extreme Computing Conference (HPEC), 2018.

This implementation of LRB follows the details in [1]. More details of LRB can be found in [2].
*/


#pragma once

namespace hornets_nest {
/**
 * @brief
 */
namespace load_balancing {
namespace kernel {




template<typename HornetDevice, typename vid_t>
__global__
void computeWorkKernelLRB(HornetDevice              hornet,
                       const vid_t* __restrict__ d_input,
                       int                       num_vertices,
                       int*         __restrict__ bins) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    __shared__ int32_t localBins[33];

    if(threadIdx.x==0){
        for (int i=0; i<33; i++)
        localBins[i]=0;
    }
    __syncthreads();

    for (auto i = id; i < num_vertices; i += stride) {
        auto deg = hornet.vertex(d_input[i]).degree();
        auto myBin  = __clz(deg);
        if(myBin!=0)
            atomicAdd(localBins+myBin, 1);
    }

    __syncthreads();    

    if(threadIdx.x==0){
        for (int i=0; i<33; i++){
            atomicAdd(bins+i, localBins[i]);
        }
    }
}


template<typename HornetDevice>
__global__
void computeWorkKernelLRB(HornetDevice              hornet,
                       int                       num_vertices,
                       int*         __restrict__ bins) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    __shared__ int32_t localBins[33];

    if(threadIdx.x==0){
        for (int i=0; i<33; i++)
        localBins[i]=0;
    }
    __syncthreads();

    for (auto i = id; i < num_vertices; i += stride) {
        auto deg = hornet.vertex(i).degree();
        auto myBin  = __clz(deg);
        if(myBin!=0)
            atomicAdd(localBins+myBin, 1);
    }

    __syncthreads();    

    if(threadIdx.x==0){
        for (int i=0; i<33; i++){
            atomicAdd(bins+i, localBins[i]);
        }
    }
}

template <typename vid_t>
__global__ void  binPrefixKernelLRB(vid_t *bins, vid_t *d_binsPrefix){
    vid_t i = threadIdx.x + blockIdx.x *blockDim.x;
    if(i>=1)
        return;
    d_binsPrefix[0]=0;
    for(vid_t b=0; b<33; b++){
        d_binsPrefix[b+1]=d_binsPrefix[b]+bins[b];
    }
}


template<bool useAllHornet,typename HornetDevice>
__global__ void  rebinKernelLRB(
  HornetDevice hornet ,
  const int    *original,
  int32_t    *d_binsPrefix,
  int     *d_reOrg,
  int N){

    int i = threadIdx.x + blockIdx.x *blockDim.x;

    __shared__ int32_t localBins[33];
    __shared__ int32_t localPos[33];

    // __shared__ int32_t prefix[33];    
    int id = threadIdx.x;
    if(id<33){
      localBins[id]=0;
      localPos[id]=0;
    }

    __syncthreads();

    int myBin,myPos;
    if(i<N){
        int32_t adjSize;
        if(useAllHornet==false)
            adjSize= hornet.vertex(original[i]).degree();
        else
            adjSize= hornet.vertex(i).degree();

        myBin  = __clz(adjSize);
        myPos = atomicAdd(localBins+myBin, 1);
    }


  __syncthreads();
    if(id<33){
        localPos[id]=atomicAdd(d_binsPrefix+id, localBins[id]);
    }
  __syncthreads();

    if(i<N){
        int pos = localPos[myBin]+myPos;
        if(useAllHornet==false)
            d_reOrg[pos]=original[i];
        else
            d_reOrg[pos]=i;
    }

}



template<typename HornetDevice,typename Operator, typename vid_t>
__global__ void skinnyWorkerLRB(
  HornetDevice hornet , 
  vid_t* d_lrbRelabled,
  Operator op,
  int N,
  int start){
    int k = threadIdx.x + blockIdx.x *blockDim.x;
    if(k>=N)
        return;
    k+=start;

    vid_t src = d_lrbRelabled[k];
    // vid_t* neighPtr = hornet.vertex(src).neighbor_ptr();
    int length = hornet.vertex(src).degree();
    const auto& vertex = hornet.vertex(src);

    for (int i=0; i<length; i++) {
        const auto&   edge = vertex.edge(i);
        op(vertex,edge);
    }
}

template<typename HornetDevice,typename Operator, typename vid_t>
__global__ void fatWorkerLRB(
  HornetDevice hornet , 
  vid_t* d_lrbRelabled,
  Operator op,
  int N,
  int start){
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if(k>=N)
        return;

    k+=start;
    vid_t src = d_lrbRelabled[k];
    int length = hornet.vertex(src).degree();
    const auto& vertex = hornet.vertex(src);

    for (int i=tid; i<length; i+=blockDim.x) {
        const auto&   edge = vertex.edge(i);
        op(vertex,edge);
    }
}


template<typename HornetDevice,typename Operator, typename vid_t>
__global__ void extraFatWorkerLRB(
  HornetDevice hornet , 
  vid_t* d_lrbRelabled,
  Operator op,
  int N){
    int k = 0;
    int tid = threadIdx.x + blockIdx.x *blockDim.x;
    int stride = blockDim.x*gridDim.x;

    while(k<N){
        vid_t src = d_lrbRelabled[k];
        int length = hornet.vertex(src).degree();

        for (int i=tid; i<length; i+=stride) {
            const auto& vertex = hornet.vertex(src);
            const auto&   edge = vertex.edge(i);
            op(vertex,edge);
        }
        k++;
    }

}



} // namespace kernel
} // namespace load_balancing
} // namespace hornets_nest
