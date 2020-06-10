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
#ifndef KTRUSS_SUPPORT_CUH
#define KTRUSS_SUPPORT_CUH

namespace hornets_nest {


template <typename V, typename D>
__device__ __forceinline__
void initialize(D diag_id,
                D u_len,
                D v_len,
                V* __restrict__ u_min,
                V* __restrict__ u_max,
                V* __restrict__ v_min,
                V* __restrict__ v_max,
                int*   __restrict__ found) {
    if (diag_id == 0) {
        *u_min = *u_max = *v_min = *v_max = 0;
        *found = 1;
    }
    else if (diag_id < u_len) {
        *u_min = 0;
        *u_max = diag_id;
        *v_max = diag_id;
        *v_min = 0;
    }
    else if (diag_id < v_len) {
        *u_min = 0;
        *u_max = u_len;
        *v_max = diag_id;
        *v_min = diag_id - u_len;
    }
    else {
        *u_min = diag_id - v_len;
        *u_max = u_len;
        *v_min = diag_id - u_len;
        *v_max = v_len;
    }
}

template <typename D>
__device__ __forceinline__
void workPerThread(D uLength,
                   D vLength,
                   int threadsPerIntersection,
                   int threadId,
                   int* __restrict__ outWorkPerThread,
                   int* __restrict__ outDiagonalId) {
  int      totalWork = uLength + vLength;
  int  remainderWork = totalWork % threadsPerIntersection;
  int  workPerThread = totalWork / threadsPerIntersection;

  int longDiagonals  = threadId > remainderWork ? remainderWork : threadId;
  int shortDiagonals = threadId > remainderWork ? threadId - remainderWork : 0;

  *outDiagonalId     = (workPerThread + 1) * longDiagonals +
                        workPerThread * shortDiagonals;
  *outWorkPerThread  = workPerThread + (threadId < remainderWork);
}

template <typename D>
__device__ __forceinline__
void bSearch(unsigned found,
             D    diagonalId,
             const vert_t*  __restrict__ uNodes,
             const vert_t*  __restrict__ vNodes,
             const D*  __restrict__ uLength,
             vert_t* __restrict__ outUMin,
             vert_t* __restrict__ outUMax,
             vert_t* __restrict__ outVMin,
             vert_t* __restrict__ outVMax,
             vert_t* __restrict__ outUCurr,
             vert_t* __restrict__ outVCurr) {
    vert_t length;
    while (!found){
        *outUCurr = (*outUMin + *outUMax) >> 1;
        *outVCurr = diagonalId - *outUCurr;
        if (*outVCurr >= *outVMax){
            length = *outUMax - *outUMin;
            if (length == 1){
                found = 1;
                continue;
            }
        }

        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (comp1 && !comp2)
            found = 1;
        else if (comp1){
            *outVMin = *outVCurr;
            *outUMax = *outUCurr;
        }
        else{
            *outVMax = *outVCurr;
            *outUMin = *outUCurr;
        }
    }

    if (*outVCurr >= *outVMax && length == 1 && *outVCurr > 0 &&
            *outUCurr > 0 && *outUCurr < *uLength - 1)
    {
        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (!comp1 && !comp2)
        {
            (*outUCurr)++;
            (*outVCurr)--;
        }
    }
}

template <typename V, typename D>
__device__ __forceinline__
int fixStartPoint(D uLength, D vLength,
                  V* __restrict__ uCurr,
                  V* __restrict__ vCurr,
                  const V* __restrict__ uNodes,
                  const V* __restrict__ vNodes) {

    unsigned uBigger = (*uCurr > 0) && (*vCurr < vLength) &&
                       (uNodes[*uCurr - 1] == vNodes[*vCurr]);
    unsigned vBigger = (*vCurr > 0) && (*uCurr < uLength) &&
                       (vNodes[*vCurr - 1] == uNodes[*uCurr]);
    *uCurr += vBigger;
    *vCurr += uBigger;
    return uBigger + vBigger;
}


template <typename V>
__device__ __forceinline__
void indexBinarySearch(V* data, V arrLen, V key, int& pos) {
    int low = 0;
    int high = arrLen - 1;
    while (high >= low)
    {
        int middle = (low + high) / 2;
        if (data[middle] == key)
        {
            pos = middle;
            return;
        }
        if (data[middle] < key)
            low = middle + 1;
        if (data[middle] > key)
            high = middle - 1;
    }
}

template<typename HornetDevice, typename V, typename D>
__device__ __forceinline__
void intersectCount(const HornetDevice& hornet,
        D uLength, D vLength,
        const V*  __restrict__ uNodes,
        const V*  __restrict__ vNodes,
        V*  __restrict__ uCurr,
        V*  __restrict__ vCurr,
        int*    __restrict__ workIndex,
        const int*    __restrict__ workPerThread,
        triangle_t*    __restrict__ triangles,
        int found,
        triangle_t*  __restrict__ outPutTriangles,
        V src, V dest,
    V u, V v) {
    if (*uCurr < uLength && *vCurr < vLength) {
        int comp;
        int vmask;
        int umask;
        while (*workIndex < *workPerThread)
        {
            vmask = umask = 0;
            comp = uNodes[*uCurr] - vNodes[*vCurr];

            *triangles += (comp == 0);

            *uCurr += (comp <= 0 && !vmask) || umask;
            *vCurr += (comp >= 0 && !umask) || vmask;
            *workIndex += (comp == 0 && !umask && !vmask) + 1;

            if (*vCurr >= vLength || *uCurr >= uLength)
                break;
        }
        *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && found);
    }
}

template<typename HornetDevice, typename V, typename D>
__device__ __forceinline__
triangle_t count_triangles(const HornetDevice& hornet,
                           V u,
                           const V* __restrict__ u_nodes,
                           D u_len,
                           V v,
                           const V* __restrict__ v_nodes,
                           D v_len,
                           int   threads_per_block,
                           volatile triangle_t* __restrict__ firstFound,
                           int    tId,
                           triangle_t* __restrict__ outPutTriangles,
                           const V*      __restrict__ uMask,
                           const V*      __restrict__ vMask,
                           triangle_t multiplier,
                           V      src,
                           V      dest) {

    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //Tersect - this number will be off by 1.
    int work_per_thread, diag_id;
    workPerThread(u_len, v_len, threads_per_block, tId,
                  &work_per_thread, &diag_id);
    triangle_t triangles = 0;
    int       work_index = 0;
    int            found = 0;
    V u_min, u_max, v_min, v_max, u_curr, v_curr;

    firstFound[tId] = 0;

    if (work_per_thread > 0) {
        // For the binary search, we are figuring out the initial poT of search.
        initialize(diag_id, u_len, v_len, &u_min, &u_max,
                   &v_min, &v_max, &found);
        u_curr = 0;
        v_curr = 0;

        bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max,
                &v_min, &v_max, &u_curr, &v_curr);

        int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr,
                                u_nodes, v_nodes);
        work_index += sum;
        if (tId > 0)
           firstFound[tId - 1] = sum;
        triangles += sum;
        intersectCount
            (hornet, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
            &work_index, &work_per_thread, &triangles, firstFound[tId],
            outPutTriangles, src, dest, u, v);

    }
    return triangles;
}


template<typename V>
__device__ __forceinline__
void workPerBlock(V numVertices,
                  V* __restrict__ outMpStart,
                  V* __restrict__ outMpEnd,
                  int blockSize) {
    V       verticesPerMp = numVertices / gridDim.x;
    V     remainderBlocks = numVertices % gridDim.x;
    V   extraVertexBlocks = (blockIdx.x > remainderBlocks) ? remainderBlocks
                                                               : blockIdx.x;
    V regularVertexBlocks = (blockIdx.x > remainderBlocks) ?
                                    blockIdx.x - remainderBlocks : 0;

    V mpStart = (verticesPerMp + 1) * extraVertexBlocks +
                     verticesPerMp * regularVertexBlocks;
    *outMpStart   = mpStart;
    *outMpEnd     = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}


//==============================================================================
//==============================================================================


template<typename HornetDevice>
__global__
void devicecuHornetKTruss(HornetDevice hornet,
                           triangle_t* __restrict__ outPutTriangles,
                           int threads_per_block,
                           int number_blocks,
                           int shifter,
                           HostDeviceVar<KTrussData> hd_data) {
    KTrussData* __restrict__ devData = hd_data.ptr();
    vert_t nv = hornet.nV();
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements
    //to intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vert_t this_mp_start, this_mp_stop;

    const int blockSize = blockDim.x;
    workPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

    __shared__ vert_t      firstFound[1024];

    vert_t     adj_offset = tx >> shifter;
    vert_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vert_t src = this_mp_start; src < this_mp_stop; src++) {
        auto vertex = hornet.vertex(src);
        vert_t srcLen = vertex.degree();

        for(int k = adj_offset; k < srcLen; k += number_blocks) {
            vert_t dest = vertex.edge(k).dst_id();
            degree_t destLen = hornet.vertex(dest).degree();

            bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
            if (avoidCalc)
                continue;

            bool sourceSmaller = srcLen < destLen;
            vert_t        small = sourceSmaller ? src : dest;
            vert_t        large = sourceSmaller ? dest : src;
            degree_t    small_len = sourceSmaller ? srcLen : destLen;
            degree_t    large_len = sourceSmaller ? destLen : srcLen;

            const vert_t* small_ptr = hornet.vertex(small).neighbor_ptr();
            const vert_t* large_ptr = hornet.vertex(large).neighbor_ptr();

            triangle_t triFound = count_triangles<HornetDevice, vert_t, degree_t>
                (hornet, small, small_ptr, small_len, large, large_ptr,
                 large_len, threads_per_block, (triangle_t*)firstFoundPos,
                 tx % threads_per_block, outPutTriangles,
                 nullptr, nullptr, 1, src, dest);


            int pos = hd_data().offset_array[src] + k;
            atomicAdd(hd_data().triangles_per_edge + pos,triFound);
        }
    }
}



//==============================================================================

template <typename HornetGraphType>
void kTrussOneIterationWeighted(HornetGraphType& hornet,
                        triangle_t* __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data) {
    devicecuHornetKTruss <<< thread_blocks, blockdim >>>
        (hornet.device(), output_triangles, threads_per_block,
         number_blocks, shifter, hd_data);

}

void kTrussOneIteration(HornetGraph& hornet,
                        triangle_t*  __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data) {
    devicecuHornetKTruss <<< thread_blocks, blockdim >>>
        (hornet.device(), output_triangles, threads_per_block,
         number_blocks, shifter, hd_data);
}


} // namespace hornets_nest

#endif
